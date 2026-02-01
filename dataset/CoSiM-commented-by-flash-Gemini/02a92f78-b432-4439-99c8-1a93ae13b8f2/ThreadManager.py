


# @file 02a92f78-b432-4439-99c8-1a93ae13b8f2/ThreadManager.py
# @brief Manages a pool of worker threads to execute scripts asynchronously, simulating device interactions.
#
# This module provides core classes for managing threads and devices within a simulated environment.
# It includes a ThreadManager for orchestrating script execution across a thread pool,
# a ConditionalBarrier for synchronizing threads at specific points, and a Device class
# to represent individual simulated devices, handling sensor data, scripts, and inter-device communication.

from Queue import Queue
from threading import Thread


class ThreadManager(object):
    """
    @brief Manages a pool of worker threads to process tasks from a queue.
    
    This class orchestrates the execution of scripts by distributing them among
    a fixed number of worker threads. It uses a thread-safe queue for task management
    and provides mechanisms for initializing, starting, and gracefully terminating
    the worker threads.
    """
    
    def __init__(self, threads_count):
        """
        @brief Initializes the ThreadManager with a specified number of worker threads.

        @param threads_count: The number of worker threads to create in the pool.
        @pre threads_count > 0.
        @post self.queue is initialized with a maximum size of threads_count.
        @post Worker threads are created and started.
        """
        # Functional Utility: Initializes a thread-safe queue to hold tasks. The size of the queue
        # is limited by the number of threads to prevent excessive memory usage.
        self.queue = Queue(threads_count)
        # Functional Utility: Stores references to the worker thread objects.
        self.threads = []
        # Functional Utility: Reference to the device object this thread manager is associated with.
        self.device = None
        # Functional Utility: Initiates the creation and starting of worker threads.
        self.initialize_workers(threads_count)
    
    def create_workers(self, threads_count):
        """
        @brief Creates the specified number of worker threads.

        @param threads_count: The number of worker threads to create.
        @pre threads_count > 0.
        @post self.threads list contains 'threads_count' new Thread objects, each targeting the 'execute' method.
        """
        # Block Logic: Iteratively creates worker threads, each configured to run the 'execute' method.
        # Invariant: Each iteration adds one new, unstarted thread to the 'threads' list.
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)
    
    def start_workers(self):
        """
        @brief Starts all created worker threads.

        @pre All worker threads have been created and are in the 'self.threads' list.
        @post All worker threads are started and actively waiting for tasks from the queue.
        """
        # Block Logic: Initiates the execution of each worker thread.
        # Invariant: Each iteration starts one thread.
        for thread in self.threads:
            thread.start()
    
    def initialize_workers(self, threads_count):
        """
        @brief Sets up and starts the worker threads.

        @param threads_count: The number of worker threads to initialize.
        @pre 'threads_count' is a positive integer.
        @post Worker threads are created and begin execution, ready to process tasks.
        """
        # Functional Utility: Delegates to create the thread objects.
        self.create_workers(threads_count)
        # Functional Utility: Delegates to start the execution of the created threads.
        self.start_workers()
    
    def set_device(self, device):
        """
        @brief Associates a device object with this ThreadManager.

        @param device: The device object to be managed by this ThreadManager's threads.
        @post self.device is set to the provided device.
        """
        # Functional Utility: Establishes a reference to the device for which scripts will be run.
        self.device = device
    
    def execute(self):
        """
        @brief The main loop for each worker thread, continuously processing tasks from the queue.

        @pre The queue is initialized.
        @post Threads continuously retrieve and execute tasks until a termination signal is received.
        """
        # Block Logic: Continuously retrieves and processes tasks from the queue.
        # Invariant: Each iteration processes one task or terminates the thread.
        while True:
            # Functional Utility: Blocks until a task is available in the queue and retrieves it.
            neighbours, script, location = self.queue.get()
            # Functional Utility: Checks if the retrieved task is a termination signal.
            no_neighbours = neighbours is None
            no_scripts = script is None
            # Block Logic: Determines if the current task is a termination signal for the thread.
            if no_neighbours and no_scripts:
                # Functional Utility: Marks the current task as done in the queue.
                self.queue.task_done()
                # Functional Utility: Exits the thread's execution loop.
                return
            # Functional Utility: Executes the received script with the provided parameters.
            self.run_script(neighbours, script, location)
            # Functional Utility: Marks the current task as done in the queue.
            self.queue.task_done()
    
    @staticmethod
    def is_not_empty(given_object):
        """
        @brief Static method to check if an object is not None.

        @param given_object: The object to check.
        @return: True if the object is not None, False otherwise.
        """
        # Functional Utility: Provides a concise check for object existence.
        return given_object is not None
    
    def run_script(self, neighbours, script, location):
        """
        @brief Executes a script, gathering data from neighboring devices and the local device,
               and then distributing the results.

        @param neighbours: A list of neighboring device objects.
        @param script: The script object to execute.
        @param location: The data location relevant to the script.
        @pre self.device is initialized and valid.
        @pre script is a callable object with a 'run' method.
        @post Script is executed, and results are propagated to relevant devices.
        """
        script_data = []
        
        # Block Logic: Iterates through neighboring devices to collect relevant sensor data.
        # Invariant: Data is collected only from devices other than the local device.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                # Functional Utility: Retrieves data from a neighboring device at the specified location.
                data = device.get_data(location)
                # Functional Utility: Appends data to 'script_data' if it is valid.
                if ThreadManager.is_not_empty(data):
                    script_data.append(data)
        
        # Functional Utility: Retrieves data from the local device at the specified location.
        data = self.device.get_data(location)
        # Functional Utility: Appends local device data to 'script_data' if it is valid.
        if ThreadManager.is_not_empty(data):
            script_data.append(data)
        # Block Logic: Executes the script if any relevant data was collected.
        if script_data:
            # Functional Utility: Executes the script with the aggregated data.
            result = script.run(script_data)
            
            # Block Logic: Distributes the script execution results to neighboring devices.
            # Invariant: Results are distributed to all neighboring devices except the local one.
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    continue
                # Functional Utility: Sets the processed data on the neighboring device.
                device.set_data(location, result)
            
            # Functional Utility: Sets the processed data on the local device.
            self.device.set_data(location, result)
    
    def submit(self, neighbours, script, location):
        """
        @brief Submits a task to the queue for processing by a worker thread.

        @param neighbours: A list of neighboring device objects.
        @param script: The script object to execute.
        @param location: The data location relevant to the script.
        @post The task is added to the internal queue.
        """
        # Functional Utility: Adds a task (tuple of neighbours, script, and location) to the queue.
        self.queue.put((neighbours, script, location))
    
    def wait_threads(self):
        """
        @brief Blocks until all tasks in the queue have been processed.

        @post The calling thread is blocked until the queue is empty and all tasks are marked as done.
        """
        # Functional Utility: Halts the calling thread until all items in the queue have been retrieved and processed.
        self.queue.join()
    
    def end_threads(self):
        """
        @brief Gracefully terminates all worker threads by sending termination signals.

        @pre All active tasks in the queue are completed.
        @post All worker threads receive termination signals and are joined.
        """
        # Functional Utility: Ensures all current tasks are processed before sending termination signals.
        self.wait_threads()
        
        # Block Logic: Sends a termination signal (None, None, None) to each worker thread.
        # Invariant: Each iteration sends one termination signal.
        for _ in xrange(len(self.threads)):
            self.submit(None, None, None)
            
        # Block Logic: Waits for each worker thread to complete its execution and terminate.
        # Invariant: Each iteration waits for one thread to join.
        for thread in self.threads:
            thread.join()


from threading import Condition


class ConditionalBarrier(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.
    
    This barrier allows a specified number of threads to wait until all of them
    have reached a certain point before proceeding. It uses a threading.Condition
    for managing thread states and notifications.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the ConditionalBarrier with the expected number of threads.

        @param num_threads: The total number of threads that must reach the barrier to release.
        @pre num_threads > 0.
        @post self.num_threads and self.count_threads are initialized.
        @post self.cond is initialized as a threading.Condition object.
        """
        # Functional Utility: Stores the total number of threads expected to reach the barrier.
        self.num_threads = num_threads
        # Functional Utility: Tracks the count of threads currently waiting at the barrier.
        self.count_threads = self.num_threads
        # Functional Utility: Provides the underlying lock and waiting mechanism for synchronization.
        self.cond = Condition()
    
    def wait(self):
        """
        @brief Blocks the calling thread until all expected threads have reached this barrier.

        @post The thread is blocked until all 'num_threads' threads call this method.
        @post Once all threads arrive, the barrier resets for future use.
        """
        # Functional Utility: Acquires the condition lock to safely modify the shared count.
        self.cond.acquire()
        # Functional Utility: Decrements the count of threads yet to reach the barrier.
        self.count_threads -= 1
        # Block Logic: Checks if the current thread is the last one to reach the barrier.
        if self.count_threads == 0:
            # Functional Utility: Notifies all waiting threads that the barrier has been met.
            self.cond.notify_all()
            # Functional Utility: Resets the barrier's count for reuse in subsequent synchronization points.
            self.count_threads = self.num_threads
        else:
            # Functional Utility: Releases the lock and blocks the current thread until notified by the last thread.
            self.cond.wait()
        # Functional Utility: Releases the condition lock.
        self.cond.release()


from threading import Event, Thread, Lock

# Functional Utility: Imports the ThreadManager for managing worker threads.
# Functional Utility: Imports the ConditionalBarrier for thread synchronization.


class Device(object):
    """
    @brief Represents a simulated device within the environment.
    
    Each device has a unique ID, sensor data, a supervisor for querying neighbors,
    and manages scripts to be executed. It also handles synchronization points
    using a barrier and ensures data consistency with location-specific locks.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device object.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary representing the device's sensor data, keyed by location.
        @param supervisor: A reference to the supervisor object for inter-device communication.
        @post Device attributes are initialized, and a DeviceThread is started for this device.
        """
        # Functional Utility: Unique identifier for the device.
        self.device_id = device_id
        # Functional Utility: Dictionary storing sensor readings specific to this device, indexed by location.
        self.sensor_data = sensor_data
        # Functional Utility: Reference to the central entity responsible for orchestrating devices.
        self.supervisor = supervisor
        
        # Functional Utility: Event flag to signal when a new script has been received by the device.
        self.script_received = Event()
        # Functional Utility: Event flag to signal that processing for the current timepoint is complete.
        self.timepoint_done = Event()
        
        # Functional Utility: List to store scripts awaiting execution on this device.
        self.scripts = []
        # Functional Utility: Boolean flag indicating if scripts have arrived for the current timepoint.
        self.scripts_arrived = False
        
        # Functional Utility: Reference to a synchronization barrier shared among devices.
        self.barrier = None
        # Functional Utility: Dictionary of locks, one for each data location, to ensure exclusive access to sensor data.
        self.location_locks = {location: Lock() for location in sensor_data}
        
        # Functional Utility: The dedicated thread for this device's operational logic.
        self.thread = DeviceThread(self)
        # Functional Utility: Starts the device's dedicated thread.
        self.thread.start()
    
    def __str__(self):
        """
        @brief Provides a string representation of the Device.

        @return: A formatted string indicating the device's ID.
        """
        # Functional Utility: Returns a human-readable string representation of the device.
        return "Device %d" % self.device_id
    
    def assign_barrier(self, barrier):
        """
        @brief Assigns a shared synchronization barrier to this device.

        @param barrier: The ConditionalBarrier object to be used for synchronization.
        @post self.barrier is set to the provided barrier object.
        """
        # Functional Utility: Links this device to a shared synchronization barrier.
        self.barrier = barrier
    
    def setup_devices(self, devices):
        """
        @brief Sets up the synchronization barrier for a group of devices.

        @param devices: A list of all device objects in the simulation.
        @pre This method is typically called only by device with ID 0.
        @post A ConditionalBarrier is created and broadcast to all devices.
        """
        number_of_devices = len(devices)
        # Block Logic: Ensures that only the device with ID 0 initializes the shared barrier.
        if self.device_id == 0:
            # Functional Utility: Creates a new shared barrier for all devices.
            self.assign_barrier(ConditionalBarrier(number_of_devices))
            # Functional Utility: Propagates the newly created barrier to all other devices.
            self.broadcast_barrier(devices, self.barrier)
    
    @staticmethod
    def broadcast_barrier(devices, barrier):
        """
        @brief Static method to broadcast a barrier to all devices in a list.

        @param devices: A list of device objects.
        @param barrier: The ConditionalBarrier object to be broadcast.
        @pre The 'barrier' object is initialized.
        @post Each device in the list (except device 0) receives the shared barrier.
        """
        # Block Logic: Iterates through all devices to assign the shared barrier.
        # Invariant: Each iteration assigns the barrier to a device, skipping device 0.
        for device in devices:
            if device.device_id == 0:
                continue
            # Functional Utility: Assigns the common barrier to the current device.
            device.assign_barrier(barrier)
    
    def accept_script(self, script, location):
        """
        @brief Accepts a script and its associated location for later execution.

        @param script: The script object to be executed.
        @param location: The data location relevant to the script.
        @post The script and location are added to the device's pending scripts.
        @post The 'scripts_arrived' flag is set to True.
        """
        # Functional Utility: Adds the incoming script and its target location to the device's processing queue.
        self.scripts.append((script, location))
        # Functional Utility: Signals that new scripts are available for execution.
        self.scripts_arrived = True
    
    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device or signals the completion of a timepoint.

        @param script: The script object to execute, or None to signal timepoint completion.
        @param location: The data location relevant to the script.
        @post If 'script' is not None, it is accepted. If 'script' is None, 'timepoint_done' is set.
        """
        # Block Logic: Differentiates between receiving a script for execution and receiving a signal for timepoint completion.
        if script is not None:
            # Functional Utility: Delegates to accept the provided script for processing.
            self.accept_script(script, location)
        else:
            # Functional Utility: Signals that the current timepoint's script assignments are complete.
            self.timepoint_done.set()
    
    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The location for which to retrieve sensor data.
        @return: The sensor data for the specified location, or None if the location is invalid.
        @pre Data access is protected by a lock if the location is valid.
        """
        # Functional Utility: Verifies if the requested location exists in the device's sensor data.
        data_is_valid = location in self.sensor_data
        # Block Logic: Acquires a lock if the data location is valid to prevent race conditions during data access.
        if data_is_valid:
            self.location_locks[location].acquire()
        # Functional Utility: Returns the sensor data if valid, otherwise returns None.
        return self.sensor_data[location] if data_is_valid else None
    
    def set_data(self, location, data):
        """
        @brief Sets sensor data for a specific location and releases the associated lock.

        @param location: The location for which to set sensor data.
        @param data: The new sensor data value.
        @post The sensor data at the specified location is updated.
        @post The lock for the specified location is released.
        """
        # Block Logic: Updates the sensor data only if the location is valid.
        if location in self.sensor_data:
            self.sensor_data[location] = data
            # Functional Utility: Releases the lock associated with the data location, allowing other threads to access it.
            self.location_locks[location].release()
    
    def shutdown(self):
        """
        @brief Joins the device's dedicated thread, effectively shutting it down.

        @post The device's thread completes its execution.
        """
        # Functional Utility: Waits for the device's operational thread to complete its tasks and terminate.
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief The dedicated thread for a simulated device, responsible for its operational lifecycle.
    
    This thread manages the device's interaction with the supervisor, processes scripts
    using a ThreadManager, and synchronizes its operations with other devices via a barrier.
    """
    
    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.

        @param device: The Device object that this thread is responsible for.
        @post The thread is initialized with a name and a ThreadManager for script execution.
        """
        # Functional Utility: Calls the base Thread class constructor, setting the thread's name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        # Functional Utility: Stores a reference to the Device object this thread operates on.
        self.device = device
        # Functional Utility: Initializes a ThreadManager for this device to handle script execution in parallel.
        self.thread_pool = ThreadManager(8)
    
    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        @post The thread continuously fetches neighbors, processes scripts, and synchronizes
              with other devices until a termination signal is received from the supervisor.
        """
        # Functional Utility: Associates the device with its thread pool for script execution.
        self.thread_pool.set_device(self.device)
        # Block Logic: Main operational loop for the device, continuing until the supervisor signals termination.
        while True:
            # Functional Utility: Retrieves a list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Checks if the supervisor has signaled the end of the simulation by returning None for neighbours.
            if neighbours is None:
                # Functional Utility: Exits the main operational loop upon receiving a termination signal.
                break
            
            # Block Logic: Inner loop to wait for and process scripts for the current timepoint.
            while True:
                # Functional Utility: Checks if scripts have arrived for processing.
                scripts_ready = self.device.scripts_arrived
                # Functional Utility: Blocks until the timepoint is explicitly marked as done.
                done_waiting = self.device.timepoint_done.wait()
                # Block Logic: Determines the state of script readiness and timepoint completion.
                if scripts_ready or done_waiting:
                    # Block Logic: Handles the scenario where the timepoint is done but no scripts were initially marked as ready.
                    if done_waiting and not scripts_ready:
                        # Functional Utility: Clears the timepoint done flag for the next cycle.
                        self.device.timepoint_done.clear()
                        # Functional Utility: Resets the scripts arrived flag to allow breaking the inner loop.
                        self.device.scripts_arrived = True
                        # Functional Utility: Exits the inner loop to proceed with script processing.
                        break
                    # Functional Utility: Resets the scripts arrived flag for the next iteration.
                    self.device.scripts_arrived = False
                    
                    # Block Logic: Submits each pending script to the thread pool for execution.
                    # Invariant: Each iteration submits one script to the thread pool.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit(neighbours, script, location)
            
            # Functional Utility: Waits for all scripts submitted to the thread pool for the current timepoint to complete.
            self.thread_pool.wait_threads()
            
            # Functional Utility: Blocks at the shared barrier until all devices have completed their processing for the timepoint.
            self.device.barrier.wait()
        
        # Functional Utility: Initiates the graceful termination of the thread pool managed by this device.
        self.thread_pool.end_threads()
