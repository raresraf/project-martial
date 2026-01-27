"""
@file ThreadManager.py
@brief Manages a pool of worker threads for parallel execution and provides a conditional barrier.

This module defines classes for managing thread pools and synchronization mechanisms
essential for concurrent operations, likely within a distributed or multi-device simulation
or control system.

Classes:
- `ThreadManager`: Manages a fixed-size pool of worker threads to execute tasks.
- `ConditionalBarrier`: Implements a reusable barrier for synchronizing multiple threads.
- `Device`: Represents a simulated or physical device with its own thread of execution,
  sensor data, and supervisor for coordinating with other devices.
- `DeviceThread`: A dedicated thread for each `Device` to manage its operations,
  including processing scripts and interacting with the `ThreadManager`.

Functional Utility:
- `ThreadManager`: Decouples task submission from execution, allowing for parallel
  processing of scripts or operations on device data.
- `ConditionalBarrier`: Ensures that a set of threads pauses at a certain point
  until all threads have reached that point, facilitating synchronized steps
  in a concurrent algorithm.
- `Device`: Encapsulates device-specific logic, data, and communication patterns,
  providing a structured way to manage individual units in a distributed system.
- `DeviceThread`: Enables each device to operate concurrently, managing its own
  task queue and interaction with a central supervisor.

Architectural Intent:
- To provide a flexible and scalable framework for simulating or controlling
  multiple interconnected devices. The use of threads and queues suggests a
  producer-consumer pattern where tasks (scripts) are submitted and processed
  in parallel. Barriers are used for global synchronization across all devices
  at specific time points or phases of execution.

Time/Space Complexity:
- `ThreadManager`:
    - `__init__`: O(threads_count) for creating threads.
    - `submit`: O(1) (average for Queue.put).
    - `wait_threads`: O(threads_count) (worst case for Queue.join, depends on task completion).
    - `execute`: O(number_of_tasks_processed_by_thread) per thread.
- `ConditionalBarrier`:
    - `wait`: O(1) (contention dependent, but conceptually constant time for wait/notify).
- `Device`:
    - `__init__`: O(number_of_locations) for creating locks.
- `DeviceThread`:
    - `run`: O(total_timepoints * (average_scripts_per_timepoint * average_script_runtime)).
"""


from Queue import Queue
from threading import Thread


class ThreadManager(object):
    """
    @class ThreadManager
    @brief Manages a pool of worker threads to execute tasks concurrently.

    This class creates and manages a fixed number of worker threads. Tasks are
    submitted to a queue, and the worker threads pull tasks from this queue
    and execute them in parallel.
    """
    
    def __init__(self, threads_count):
        """
        @brief Initializes the ThreadManager with a specified number of threads.
        @param threads_count: The number of worker threads to create in the pool.
        
        Initializes a queue with a capacity equal to `threads_count` and
        creates an empty list to hold thread objects.
        """
        # Functional Utility: Initializes a queue with a maximum size to limit pending tasks.
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = None
        # Block Logic: Immediately initializes and starts worker threads.
        self.initialize_workers(threads_count)
    
    def create_workers(self, threads_count):
        """
        @brief Creates worker thread objects but does not start them.
        @param threads_count: The number of worker threads to create.
        
        Each worker thread is configured to run the `execute` method of this manager.
        """
        # Block Logic: Creates `threads_count` new Thread objects and appends them to `self.threads`.
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)
    
    def start_workers(self):
        """
        @brief Starts all worker threads in the pool.
        
        Each thread begins executing its target function (which is `self.execute`).
        """
        # Block Logic: Iterates through the list of created threads and starts each one.
        for thread in self.threads:
            thread.start()
    
    def initialize_workers(self, threads_count):
        """
        @brief Creates and starts the worker threads.
        @param threads_count: The number of worker threads to initialize.
        
        This method is a convenience wrapper that first creates the workers
        and then immediately starts them.
        """
        # Block Logic: Calls helper methods to create and start the worker threads.
        self.create_workers(threads_count)
        self.start_workers()
    
    def set_device(self, device):
        """
        @brief Sets the device associated with this ThreadManager.
        @param device: The device object to be associated.
        
        This method allows the ThreadManager to interact with a specific device,
        which is crucial for its worker threads to perform device-specific operations.
        """
        self.device = device
    
    def execute(self):
        """
        @brief The main loop for each worker thread.
        
        Each worker thread continuously fetches tasks from the queue. A task consists
        of `neighbours`, `script`, and `location`. If both `neighbours` and `script`
        are `None`, it's a shutdown signal for the thread. Otherwise, it runs the script.
        """
        # Block Logic: Worker thread main loop. Continuously gets tasks from the queue.
        while True:
            # Functional Utility: Retrieves a task from the queue. This call blocks until an item is available.
            neighbours, script, location = self.queue.get()
            no_neighbours = neighbours is None
            no_scripts = script is None
            # Block Logic: Checks for the shutdown signal (None, None, None).
            if no_neighbours and no_scripts:
                # Functional Utility: Signals that the task has been processed.
                self.queue.task_done()
                return
            # Block Logic: Executes the script with the provided data.
            self.run_script(neighbours, script, location)
            # Functional Utility: Signals that the task has been processed.
            self.queue.task_done()
    
    @staticmethod
    def is_not_empty(given_object):
        """
        @brief Static method to check if an object is not None.
        @param given_object: The object to check.
        @return True if the object is not None, False otherwise.
        """
        return given_object is not None
    
    def run_script(self, neighbours, script, location):
        """
        @brief Executes a given script using data from the current device and its neighbors.
        @param neighbours: A list of neighboring device objects.
        @param script: The script object to run. This object is expected to have a `run` method.
        @param location: The specific data location to retrieve from devices.
        
        This method gathers data from the current device and its neighbors (excluding itself),
        runs the provided script with this collected data, and then propagates the result
        back to the relevant devices.
        """
        script_data = []
        
        # Block Logic: Gathers data from neighboring devices, excluding the current device.
        for device in neighbours:
            # Functional Utility: Skips the current device when iterating neighbors.
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                # Block Logic: Appends data only if it's not empty.
                if ThreadManager.is_not_empty(data):
                    script_data.append(data)
        
        # Functional Utility: Gathers data from the current device.
        data = self.device.get_data(location)
        # Block Logic: Appends data from the current device if not empty.
        if ThreadManager.is_not_empty(data):
            script_data.append(data)
        
        # Block Logic: If any script data was collected, runs the script and propagates results.
        if script_data:
            # Functional Utility: Executes the script with the collected data.
            result = script.run(script_data)
            
            # Block Logic: Propagates the script result back to neighboring devices (excluding itself).
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    continue
                # Functional Utility: Sets the processed data on the neighbor device.
                device.set_data(location, result)
            
            # Functional Utility: Sets the processed data on the current device.
            self.device.set_data(location, result)
    
    def submit(self, neighbours, script, location):
        """
        @brief Submits a task to the thread pool queue.
        @param neighbours: List of neighboring devices for the task.
        @param script: The script to be executed.
        @param location: The data location relevant to the script.
        
        This method places a task tuple into the queue, which will eventually
        be picked up by an available worker thread.
        """
        self.queue.put((neighbours, script, location))
    
    def wait_threads(self):
        """
        @brief Blocks until all tasks in the queue have been processed.
        
        This method waits for the queue to become empty, indicating that all
        submitted tasks have been completed by the worker threads.
        """
        self.queue.join()
    
    def end_threads(self):
        """
        @brief Signals all worker threads to terminate and waits for their completion.
        
        First, it waits for all current tasks to finish. Then, it sends `None` signals
        to each thread to indicate termination and waits for each thread to join.
        """
        # Block Logic: Waits for all existing tasks to complete.
        self.wait_threads()
        
        # Block Logic: Submits a termination signal for each worker thread.
        for _ in xrange(len(self.threads)):
            self.submit(None, None, None)
            
        # Block Logic: Waits for all worker threads to finish executing and terminate.
        for thread in self.threads:
            thread.join()


from threading import Condition


class ConditionalBarrier(object):
    """
    @class ConditionalBarrier
    @brief Implements a reusable barrier that synchronizes multiple threads.

    Threads call `wait()` and block until `num_threads` threads have entered
    the barrier. Once all threads have arrived, they are all released, and
    the barrier resets for the next use.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the ConditionalBarrier.
        @param num_threads: The total number of threads expected to reach the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Functional Utility: Counter for threads waiting at the barrier.
        self.cond = Condition() # Functional Utility: The condition variable used for synchronization.
    
    def wait(self):
        """
        @brief Blocks the calling thread until all expected threads reach this barrier.
        
        When the last thread reaches the barrier, all waiting threads are released,
        and the barrier's internal counter is reset for reuse.
        """
        # Block Logic: Acquires the condition variable's lock.
        self.cond.acquire()
        self.count_threads -= 1
        # Block Logic: Checks if this is the last thread to reach the barrier.
        if self.count_threads == 0:
            self.cond.notify_all() # Functional Utility: Releases all waiting threads.
            self.count_threads = self.num_threads # Functional Utility: Resets the counter for the next use.
        else:
            self.cond.wait() # Functional Utility: Waits for other threads to reach the barrier.
        self.cond.release() # Block Logic: Releases the condition variable's lock.


from threading import Event, Thread, Lock

# Architectural Intent: These imports suggest a hierarchical management structure
# where ThreadManager handles task execution, and ConditionalBarrier handles synchronization.
from ThreadManager import ThreadManager
from barriers import ConditionalBarrier # Assuming barriers is a module containing ConditionalBarrier, though it's defined here.


class Device(object):
    """
    @class Device
    @brief Represents a simulated device that processes sensor data and executes scripts.

    Each device has a unique ID, sensor data, and a supervisor. It runs its
    operations in a dedicated `DeviceThread` and uses locks for data access
    and a barrier for synchronization with other devices.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data, keyed by location.
        @param supervisor: The supervisor object responsible for coordinating devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        self.script_received = Event() # Functional Utility: Event to signal when a script has been received.
        self.timepoint_done = Event() # Functional Utility: Event to signal that processing for a timepoint is done.
        
        self.scripts = []
        self.scripts_arrived = False
        
        self.barrier = None
        # Block Logic: Creates a lock for each data location to ensure thread-safe access to `sensor_data`.
        self.location_locks = {location: Lock() for location in sensor_data}
        
        self.thread = DeviceThread(self) # Functional Utility: Creates and starts a dedicated thread for this device.
        self.thread.start()
    
    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string indicating the device ID.
        """
        return "Device %d" % self.device_id
    
    def assign_barrier(self, barrier):
        """
        @brief Assigns a synchronization barrier to the device.
        @param barrier: The `ConditionalBarrier` object to use for synchronization.
        """
        self.barrier = barrier
    
    def setup_devices(self, devices):
        """
        @brief Sets up the synchronization barrier for a group of devices.
        @param devices: A list of all device objects in the system.
        
        Only device with `device_id == 0` creates the barrier and broadcasts
        it to all other devices. This ensures a single shared barrier.
        """
        number_of_devices = len(devices)
        # Block Logic: Only the device with ID 0 is responsible for creating and broadcasting the barrier.
        if self.device_id == 0:
            self.assign_barrier(ConditionalBarrier(number_of_devices))
            self.broadcast_barrier(devices, self.barrier)
    
    @staticmethod
    def broadcast_barrier(devices, barrier):
        """
        @brief Static method to broadcast a barrier to a list of devices.
        @param devices: A list of device objects.
        @param barrier: The `ConditionalBarrier` object to distribute.
        
        Each device in the list (except device 0) is assigned the same barrier object.
        """
        # Block Logic: Iterates through devices and assigns the barrier, skipping device 0.
        for device in devices:
            if device.device_id == 0:
                continue
            device.assign_barrier(barrier)
    
    def accept_script(self, script, location):
        """
        @brief Accepts a script and its associated location for execution.
        @param script: The script object.
        @param location: The data location relevant to the script.
        
        Appends the script-location pair to the device's list of scripts and
        sets a flag indicating that scripts have arrived.
        """
        self.scripts.append((script, location))
        self.scripts_arrived = True
    
    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device or signals timepoint completion.
        @param script: The script object, or `None` to signal completion.
        @param location: The data location relevant to the script.
        
        If a script is provided, it's accepted. If `script` is `None`, it
        signals that the current timepoint's processing is done for this device.
        """
        # Block Logic: If a script is provided, accept it. Otherwise, signal timepoint completion.
        if script is not None:
            self.accept_script(script, location)
        else:
            self.timepoint_done.set() # Functional Utility: Signals that this device has completed its tasks for the current timepoint.
    
    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key for the sensor data.
        @return The sensor data for the location, or `None` if the location is invalid.
        
        Acquires a lock for the specific location before retrieving data to ensure
        thread-safe access to `sensor_data`.
        """
        data_is_valid = location in self.sensor_data
        # Block Logic: Acquires a lock before accessing data if the location is valid.
        if data_is_valid:
            self.location_locks[location].acquire()
        return self.sensor_data[location] if data_is_valid else None
    
    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location and releases the corresponding lock.
        @param location: The key for the sensor data.
        @param data: The new sensor data to set.
        
        This method updates the `sensor_data` for a specific location and then
        releases the lock associated with that location.
        """
        # Block Logic: Updates data and releases the lock if the location is valid.
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
    
    def shutdown(self):
        """
        @brief Shuts down the device's dedicated thread.
        
        This method waits for the `DeviceThread` to complete its execution.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief A dedicated thread for a `Device` to manage its script execution and synchronization.

    Each `DeviceThread` runs a loop that continuously interacts with its associated
    `Device` and a `ThreadManager` pool to process scripts and synchronize with
    other devices using a `ConditionalBarrier`.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a DeviceThread.
        @param device: The `Device` object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadManager(8) # Functional Utility: Creates a ThreadManager with 8 worker threads for local task execution.
    
    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        
        This method continuously gets neighbors from the supervisor, processes
        scripts received by its device using the internal thread pool, and
        synchronizes with other DeviceThreads via a barrier.
        It breaks the loop when no more neighbors are returned by the supervisor,
        indicating the end of the simulation or process.
        """
        self.thread_pool.set_device(self.device) # Functional Utility: Associates the thread pool with this device.
        # Block Logic: Main loop for processing timepoints/iterations.
        while True:
            # Functional Utility: Retrieves current neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: If no neighbors are returned, it signifies the end of the simulation.
            if neighbours is None:
                break
            
            # Block Logic: Loop for processing scripts within a single timepoint.
            while True:
                scripts_ready = self.device.scripts_arrived
                done_waiting = self.device.timepoint_done.wait() # Block Logic: Waits for the timepoint_done event to be set.
                # Block Logic: Checks if scripts are ready or waiting is complete.
                if scripts_ready or done_waiting:
                    # Block Logic: Special handling if waiting is done but no scripts arrived (likely a signal to proceed without scripts).
                    if done_waiting and not scripts_ready:
                        self.device.timepoint_done.clear() # Functional Utility: Resets the event for the next timepoint.
                        self.device.scripts_arrived = True # Functional Utility: Ensures the script processing loop runs.
                        break
                    self.device.scripts_arrived = False
                    
                    # Block Logic: Submits each received script to the thread pool for execution.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit(neighbours, script, location)
            
            self.thread_pool.wait_threads() # Block Logic: Waits for all scripts for the current timepoint to finish.
            
            self.device.barrier.wait() # Block Logic: Waits at the global barrier, synchronizing with other devices.
        
        self.thread_pool.end_threads() # Functional Utility: Shuts down the internal thread pool.