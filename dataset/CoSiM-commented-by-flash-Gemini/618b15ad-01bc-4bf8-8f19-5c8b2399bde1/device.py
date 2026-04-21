
"""
@file device.py
@brief Distributed device simulation framework with custom thread-pool orchestration.

Functional Intent: Implements a parallel execution environment for autonomous 
devices in a simulated network. Features a hierarchical management structure where 
each device utilizes a custom ThreadPool to execute data-processing scripts. 
Coordinates neighbor discovery and ensures global synchronization via barriers, 
while maintaining thread-safe access to shared data locations through per-location locks.

Domain: Production Systems, Distributed Computing, Parallel Processing.
"""


from threading import Event, Thread, Lock
from barrier import Barrier
from threadpool import ThreadPool

class Device(object):
    """
    @brief Represents a computational entity in the simulated distributed network.
    
    Functional Utility: Manages local sensor state, task assignments, and a dedicated 
    thread pool for local concurrency. Interfaces with a global supervisor and 
    participates in consensus-based execution steps.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device and prepares its internal orchestration state.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Synchronization: Events for coordinating script assignment and epoch completion.
        self.script_received = Event()
        self.timepoint_done = Event()
        
        self.scripts = [] # Logic: Sequence of transformation scripts for the current step.
        self.barrier = None # Logic: Shared barrier for cross-device epoch alignment.
        
        # Synchronization: Pool of mutexes for protecting specific sensor data locations.
        self.locks = {location : Lock() for location in sensor_data}

        # Control: Spawns the main management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Orchestrates the cluster-wide setup of synchronization primitives.
        
        Algorithm: Single-node master initialization (Device 0).
        Logic: The master device initializes a global barrier and triggers the 
        management threads of all participating nodes.
        """
        # Pre-condition: Setup is restricted to the first device to ensure a singleton barrier.
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))

            # Side Effect: Propagates the barrier reference and signals thread activation.
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier
                device.thread.start()

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing task for the current epoch.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Logic: Signaling None indicates the end of the assignment phase.
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a location. (Note: Caller should hold appropriate locks).
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the orchestration thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Management thread responsible for task distribution and consensus coordination.
    
    Logic: Orchestrates the transition between network-wide discovery, local 
    parallel execution via the ThreadPool, and global synchronization barriers.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Optimization: Pre-allocated worker pool to balance local resource contention.
        self.thread_pool = ThreadPool(7, device)

    def run(self):
        """
        @brief Core device operational loop.
        
        Algorithm: Discrete time-step execution.
        Logic: 
        1. Identifies current neighborhood topography.
        2. Waits for script delivery for the current epoch.
        3. Dispatches work to the ThreadPool.
        4. Synchronizes at the global barrier to align with all peer devices.
        """
        while True:
            # Block Logic: Topology discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Task Dispatch: Offloads script execution to the parallel worker pool.
            for (script, location) in self.device.scripts:
                self.thread_pool.submit(neighbours, script, location)

            # Synchronization: Global consensus point before proceeding to the next epoch.
            self.device.barrier.wait()

        # Shutdown Sequence: Triggers teardown of the internal worker pool.
        self.thread_pool.shutdown()


class ThreadPool(object):
    """
    @brief Custom worker thread manager for parallel task processing.
    
    Logic: Implements a bounded producer-consumer queue for distributing script 
    execution tasks across a fixed set of persistent worker threads.
    """
    
    def __init__(self, threads_count, device):
        """
        @brief Initializes the pool and starts the persistent worker threads.
        """
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = device

        # Block Logic: Worker instantiation.
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)
            new_thread.start()

    def execute(self):
        """
        @brief Core worker loop.
        
        Logic: Continuously polls for tasks and executes them until a shutdown 
        sentinel (None) is encountered.
        """
        while True:
            now = self.queue.get()
            if now is None:
                # Poison Pill: Sentinel value to terminate the worker thread.
                self.queue.task_done()
                return

            self.run_script(now)
            self.queue.task_done()

    def run_script(self, script_env_data):
        """
        @brief Performs the actual script execution and distributed state update.
        
        Algorithm: Neighborhood reduction and broadcast.
        Logic: 
        1. Gathers current state from all neighbors for the target location.
        2. Acquires location-specific locks on each device to ensure consistency.
        3. Executes the script logic.
        4. Broadcasts results back to all neighborhood participants.
        """
        neighbours, script, location = script_env_data
        script_data = []

        # Block Logic: Thread-safe neighborhood data gathering.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                # Synchronization: Acquires the peer's location-specific mutex.
                if location in device.sensor_data:
                    device.locks[location].acquire()
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        
        if location in self.device.sensor_data:
            self.device.locks[location].acquire()
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Functional Intent: Executes user-defined transformation logic.
            result = script.run(script_data)

            # Block Logic: Atomic state broadcast.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)
                    if location in device.sensor_data:
                        device.locks[location].release()

            self.device.set_data(location, result)
            if location in self.device.sensor_data:
                self.device.locks[location].release()

    def submit(self, neighbours, script, location):
        """
        @brief Enqueues a task for parallel execution by the pool.
        """
        self.queue.put((neighbours, script, location))

    def shutdown(self):
        """
        @brief Orchestrates a graceful termination of all pool resources.
        """
        self.queue.join()

        # Control: Broadcasts termination signal to all worker threads.
        for _ in self.threads:
            self.queue.put(None)

        for thread in self.threads:
            thread.join()
