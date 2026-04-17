"""
@c7af8b26-2c30-4b17-a679-18444e1b2097/ThreadManager.py
@brief Distributed sensor simulation with a prioritized thread pool and synchronized location-level locking.
* Algorithm: Event-driven worker thread pool with atomic script consumption and condition-variable barriers.
* Functional Utility: Orchestrates simulation timepoints across multiple devices by managing concurrent script execution and cluster-wide state consistency.
"""

from Queue import Queue
from threading import Thread

class ThreadManager(object):
    """
    @brief Manager class for a pool of worker threads that execute sensor scripts.
    """
    
    def __init__(self, threads_count):
        """
        @brief Initializes the manager and bootstraps the requested number of worker threads.
        """
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = None
        self.initialize_workers(threads_count)
    
    def create_workers(self, threads_count):
        """
        @brief Instantiates worker threads targeting the main execution loop.
        """
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)
    
    def start_workers(self):
        """
        @brief Signals all worker threads to begin polling the task queue.
        """
        for thread in self.threads:
            thread.start()
    
    def initialize_workers(self, threads_count):
        """
        @brief Unified workflow for worker pool creation and startup.
        """
        self.create_workers(threads_count)
        self.start_workers()
    
    def set_device(self, device):
        """
        @brief Affiliates the manager with a specific device instance for data context.
        """
        self.device = device
    
    def execute(self):
        """
        @brief Main execution loop for each worker thread in the pool.
        Algorithm: Blocks on the queue for tasks; handles termination via "poison pill" (None, None, None).
        """
        while True:
            neighbours, script, location = self.queue.get()
            no_neighbours = neighbours is None
            no_scripts = script is None
            
            # Logic: Graceful termination check.
            if no_neighbours and no_scripts:
                self.queue.task_done()
                return
            
            self.run_script(neighbours, script, location)
            self.queue.task_done()
    
    @staticmethod
    def is_not_empty(given_object):
        """
        @brief Utility for validating data presence before processing.
        """
        return given_object is not None
    
    def run_script(self, neighbours, script, location):
        """
        @brief Orchestrates the execution of a single script unit.
        Logic: Aggregates data from the neighborhood and updates shared state across peers.
        """
        script_data = []
        
        # Distributed Data Aggregation Phase.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if ThreadManager.is_not_empty(data):
                    script_data.append(data)
        
        # Logic: Includes local data in the processing batch.
        data = self.device.get_data(location)
        if ThreadManager.is_not_empty(data):
            script_data.append(data)
            
        if script_data:
            # Execution and Propagation Phase.
            result = script.run(script_data)
            
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    continue
                # Logic: Updates peer state with computed result.
                device.set_data(location, result)
            
            # Logic: Updates local state.
            self.device.set_data(location, result)
    
    def submit(self, neighbours, script, location):
        """
        @brief Enqueues a new script task for the worker pool.
        """
        self.queue.put((neighbours, script, location))
    
    def wait_threads(self):
        """
        @brief Blocks until all currently enqueued tasks in the manager are finished.
        """
        self.queue.join()

    def end_threads(self):
        """
        @brief Coordinates a clean shutdown of all worker threads.
        """
        self.wait_threads()
        
        # Logic: Dispatches a termination signal for each worker in the pool.
        for _ in xrange(len(self.threads)):
            self.submit(None, None, None)
            
        for thread in self.threads:
            thread.join()


from threading import Condition

class ConditionalBarrier(object):
    """
    @brief Implementation of a reusable synchronization barrier using monitors.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes barrier for a fixed number of participating threads.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
    
    def wait(self):
        """
        @brief Blocks until the barrier condition (all threads arrived) is satisfied.
        Invariant: Resets the arrival counter upon release to allow re-use in the next phase.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Logic: Last thread to arrive releases the entire group.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


from threading import Event, Thread, Lock

class Device(object):
    """
    @brief Encapsulates a sensor device's state and its coordination thread.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device and starts its management lifecycle.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        self.script_received = Event()
        self.timepoint_done = Event()
        
        self.scripts = []
        self.scripts_arrived = False
        
        self.barrier = None
        # Logic: Per-location locks to ensure thread-safe updates to sensor readings.
        self.location_locks = {location: Lock() for location in sensor_data}
        
        self.thread = DeviceThread(self)
        self.thread.start()
    
    def __str__(self):
        return "Device %d" % self.device_id
    
    def assign_barrier(self, barrier):
        """
        @brief Links the device to a cluster-wide synchronization point.
        """
        self.barrier = barrier
    
    def setup_devices(self, devices):
        """
        @brief Global initialization of barriers and locks across the network.
        Invariant: Coordinator node (ID 0) manages the creation and broadcast of the barrier.
        """
        number_of_devices = len(devices)
        if self.device_id == 0:
            self.assign_barrier(ConditionalBarrier(number_of_devices))
            self.broadcast_barrier(devices, self.barrier)
    
    @staticmethod
    def broadcast_barrier(devices, barrier):
        """
        @brief Distributes a shared barrier instance to all peer devices.
        """
        for device in devices:
            if device.device_id == 0:
                continue
            device.assign_barrier(barrier)
    
    def accept_script(self, script, location):
        """
        @brief Buffers an incoming script for the current execution phase.
        """
        self.scripts.append((script, location))
        self.scripts_arrived = True
    
    def assign_script(self, script, location):
        """
        @brief Top-level interface for script arrival or phase completion signal.
        """
        if script is not None:
            self.accept_script(script, location)
        else:
            self.timepoint_done.set()
    
    def get_data(self, location):
        """
        @brief Synchronized getter for sensor data.
        Pre-condition: Acquisition of the location-specific lock ensures atomicity.
        """
        data_is_valid = location in self.sensor_data
        if data_is_valid:
            self.location_locks[location].acquire()
        return self.sensor_data[location] if data_is_valid else None
    
    def set_data(self, location, data):
        """
        @brief Synchronized setter for sensor data.
        Post-condition: Release of location-specific lock completes the update cycle.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
    
    def shutdown(self):
        """
        @brief Gracefully terminates the device's main coordination thread.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief Management thread coordinating simulation timepoints and worker dispatching.
    """
    
    def __init__(self, device):
        """
        @brief Initializes the coordinator thread and its local worker pool (8 threads).
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadManager(8)
    
    def run(self):
        """
        @brief Core lifecycle loop of the device node.
        Algorithm: Phased execution with event-driven task submission and global barrier alignment.
        """
        self.thread_pool.set_device(self.device)
        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Logic: Shutdown path.
                break
            
            # Block Logic: Internal phase loop to handle continuous task delivery.
            while True:
                scripts_ready = self.device.scripts_arrived
                done_waiting = self.device.timepoint_done.wait()
                
                if scripts_ready or done_waiting:
                    if done_waiting and not scripts_ready:
                        # Logic: All scripts assigned for the current timepoint.
                        self.device.timepoint_done.clear()
                        self.device.scripts_arrived = True
                        break
                    
                    self.device.scripts_arrived = False
                    
                    # Dispatch Phase: Offloads scripts to the worker pool.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit(neighbours, script, location)
            
            # Synchronization Phase: Wait for local pool completion and global barrier alignment.
            self.thread_pool.wait_threads()
            self.device.barrier.wait()
        
        # Cleanup Phase: Terminate worker threads.
        self.thread_pool.end_threads()
