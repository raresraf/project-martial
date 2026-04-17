"""
@c7af8b26-2c30-4b17-a679-18444e1b2097/ThreadManager.py
@brief Distributed sensor network simulation with a managed parallel worker pool.
This module implements a coordinated processing framework where a central 'ThreadManager' 
supervises a set of persistent worker threads. It utilizes a stateful locking 
protocol ('sticky locks') to manage shared sensor data and a monitor-based 
synchronization barrier to maintain temporal consistency across the network.

Domain: Thread Pool Management, Distributed State Synchronization, Monitor Patterns.
"""

from Queue import Queue
from threading import Thread


class ThreadManager(object):
    """
    Management layer for a persistent pool of computational workers.
    Functional Utility: Provides an asynchronous submission interface for tasks 
    and handles the graceful startup and termination of worker threads.
    """
    
    def __init__(self, threads_count):
        """
        Initializes the manager and spawns workers.
        @param threads_count: Size of the worker pool.
        """
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = None
        self.initialize_workers(threads_count)
    
    def create_workers(self, threads_count):
        """Spawns the specified number of worker threads."""
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)
    
    def start_workers(self):
        """Activates all threads in the pool."""
        for thread in self.threads:
            thread.start()
    
    def initialize_workers(self, threads_count):
        """Composite operation for pool setup."""
        self.create_workers(threads_count)
        self.start_workers()
    
    def set_device(self, device):
        """Injects the parent device context into the manager."""
        self.device = device
    
    def execute(self):
        """
        Main execution loop for worker threads.
        Logic: Continuously pulls tasks from the queue and applies the 
        computational logic until a termination signal (poison pill) is received.
        """
        while True:
            # Block until a task is available.
            neighbours, script, location = self.queue.get()
            no_neighbours = neighbours is None
            no_scripts = script is None
            
            # Check for poison pill (None, None, None).
            if no_neighbours and no_scripts:
                self.queue.task_done()
                return
            
            self.run_script(neighbours, script, location)
            # Signal task completion to support queue.join().
            self.queue.task_done()
    
    @staticmethod
    def is_not_empty(given_object):
        """Safety check for data validity."""
        return given_object is not None
    
    def run_script(self, neighbours, script, location):
        """
        Executes a single computational script.
        Logic: Implements the sticky lock protocol where get_data() triggers a 
        lock acquisition and set_data() triggers a release.
        """
        script_data = []
        
        # Aggregate neighborhood state.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                # Functional Utility: Atomic acquisition of neighbor state.
                data = device.get_data(location)
                if ThreadManager.is_not_empty(data):
                    script_data.append(data)
        
        # Include local sensor data.
        data = self.device.get_data(location)
        if ThreadManager.is_not_empty(data):
            script_data.append(data)

        if script_data:
            # Execute script logic.
            result = script.run(script_data)
            
            # Propagate results and release spatial locks.
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    continue
                device.set_data(location, result)
            
            self.device.set_data(location, result)
    
    def submit(self, neighbours, script, location):
        """Enqueues a new computational task."""
        self.queue.put((neighbours, script, location))
    
    def wait_threads(self):
        """Blocks until the internal queue is empty and all tasks are processed."""
        self.queue.join()

    def end_threads(self):
        """
        Orchestrates pool shutdown.
        Logic: Waits for current work to finish, then dispatches poison pills.
        """
        self.wait_threads()
        
        # Dispatch termination signal to each thread.
        for _ in xrange(len(self.threads)):
            self.submit(None, None, None)
            
        # Reclaim thread resources.
        for thread in self.threads:
            thread.join()


from threading import Condition


class ConditionalBarrier(object):
    """
    Monitor-based reusable barrier implementation.
    Functional Utility: Provides a temporal rendezvous point for a fixed number of threads.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
    
    def wait(self):
        """Blocks the caller until the arrival threshold is met."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Threshold reached: wake all participants and reset for reuse.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


from threading import Event, Thread, Lock

from ThreadManager import ThreadManager
from barriers import ConditionalBarrier


class Device(object):
    """
    Core node representation in the distributed system.
    Functional Utility: Coordinates task receipt from the supervisor and 
    manages local sensor state through a transactional locking interface.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        self.script_received = Event()
        self.timepoint_done = Event()
        
        self.scripts = []
        self.scripts_arrived = False
        
        self.barrier = None
        # Spatial lock pool to ensure per-location consistency.
        self.location_locks = {location: Lock() for location in sensor_data}
        
        self.thread = DeviceThread(self)
        self.thread.start()
    
    def __str__(self):
        return "Device %d" % self.device_id
    
    def assign_barrier(self, barrier):
        """Injects the shared network barrier."""
        self.barrier = barrier
    
    def setup_devices(self, devices):
        """
        Global synchronization initialization.
        Logic: Elects Device 0 to create and distribute the shared barrier.
        """
        number_of_devices = len(devices)
        if self.device_id == 0:
            self.assign_barrier(ConditionalBarrier(number_of_devices))
            self.broadcast_barrier(devices, self.barrier)
    
    @staticmethod
    def broadcast_barrier(devices, barrier):
        """Helper to propagate the barrier across the network group."""
        for device in devices:
            if device.device_id == 0:
                continue
            device.assign_barrier(barrier)
    
    def accept_script(self, script, location):
        """Registers a new computational script into the local buffer."""
        self.scripts.append((script, location))
        self.scripts_arrived = True
    
    def assign_script(self, script, location):
        """Interface for the supervisor to assign work or signal end-of-step."""
        if script is not None:
            self.accept_script(script, location)
        else:
            self.timepoint_done.set()
    
    def get_data(self, location):
        """
        Transactional retrieval of sensor data.
        Functional Utility: Atomically acquires the spatial lock for the location.
        """
        data_is_valid = location in self.sensor_data
        if data_is_valid:
            self.location_locks[location].acquire()
        return self.sensor_data[location] if data_is_valid else None
    
    def set_data(self, location, data):
        """
        Transactional update of sensor data.
        Functional Utility: Updates state and releases the spatial lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
    
    def shutdown(self):
        """Joins the main management thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    Simulated node management lifecycle.
    Functional Utility: Orchestrates simulation timepoints and manages the 
    interaction between the high-level supervisor and the worker thread pool.
    """
    
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadManager(8)

    def run(self):
        """
        Main execution loop for the node orchestration.
        Algorithm: Iterative timepoint processing with event-driven task dispatch.
        """
        self.thread_pool.set_device(self.device)
        while True:
            # Topology Discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Block Logic: Simulation Phase Coordination.
            while True:
                # Wait for work assignment or phase completion signal.
                scripts_ready = self.device.scripts_arrived
                done_waiting = self.device.timepoint_done.wait()
                if scripts_ready or done_waiting:
                    if done_waiting and not scripts_ready:
                        # Reset for the next simulation cycle.
                        self.device.timepoint_done.clear()
                        self.device.scripts_arrived = True
                        break
                    
                    self.device.scripts_arrived = False
                    # Offload tasks to the managed parallel pool.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit(neighbours, script, location)
            
            # Synchronize: Wait for the local worker pool to finish.
            self.thread_pool.wait_threads()
            
            # Global Rendezvous: Ensures network-wide temporal consistency.
            self.device.barrier.wait()
        
        # Cleanup worker resources on shutdown.
        self.thread_pool.end_threads()
