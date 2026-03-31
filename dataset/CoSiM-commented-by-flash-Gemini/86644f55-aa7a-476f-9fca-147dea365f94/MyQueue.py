
"""
This module implements a multi-threaded device simulation environment.
It includes a custom queue-based thread pool (`MyQueue`), a reusable barrier
for synchronization (`ReusableBarrier`), and classes representing `Device`
entities and their main execution threads (`DeviceThread`).

Domain: Concurrency, Multi-threading, Distributed Systems, Producer-Consumer Problem.
"""

from Queue import Queue 
from threading import Thread

class MyQueue():
    """
    @brief Implements a custom queue-based thread pool for processing tasks.

    This class manages a pool of worker threads that continuously fetch tasks
    (script execution requests) from an internal queue, process them, and
    update device data. It also provides mechanisms for graceful shutdown.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the MyQueue thread pool.

        @param num_threads: The number of worker threads to create in the pool.
        """
        self.queue = Queue(num_threads) # @brief The underlying queue for tasks.
        self.threads = [] # @brief List to hold the worker thread objects.
        self.device = None # @brief Reference to the Device instance this queue serves.

        # Block Logic: Create and start worker threads.
        for _ in xrange(num_threads):
            thread = Thread(target=self.run)
            self.threads.append(thread)
        
        # Block Logic: Start all created worker threads.
        for thread in self.threads:
            thread.start()
    
    def run(self):
        """
        @brief The main execution loop for each worker thread in the pool.

        Each worker continuously gets a task from the queue. If the task is a
        sentinel (None, None, None), the thread terminates. Otherwise, it
        executes the script, collects data from neighbors and the device,
        and updates data if the script yields a result.
        """
        while True:
            # Block Logic: Get a task from the queue. Tasks consist of neighbors, script, and location.
            neighbours, script, location = self.queue.get()

            # Block Logic: Check for sentinel value to terminate the thread.
            if neighbours is None and script is None:
                self.queue.task_done() # Inline: Mark the task as done before exiting.
                return
        
            script_data = [] # @brief List to store collected sensor data for the script.
            
            # Block Logic: Collect data from neighboring devices for the current location.
            for device in neighbours:
                if device.device_id != self.device.device_id: # Inline: Exclude self device.
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            # Block Logic: Collect data from the current device for the current location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Block Logic: Execute the script with the collected data.
                result = script.run(script_data)
                
                # Block Logic: Disseminate the script's result to neighboring devices.
                for device in neighbours:
                    if device.device_id != self.device.device_id: # Inline: Exclude self device.
                        device.set_data(location, result)
                
                # Block Logic: Update the current device's data with the script's result.
                self.device.set_data(location, result)
            
            self.queue.task_done() # @brief Mark the current task as done.
    
    def finish(self):
        """
        @brief Gracefully shuts down the worker threads in the pool.

        It blocks until all tasks in the queue are processed (`queue.join()`),
        then puts sentinel values into the queue to signal each worker thread to terminate,
        and finally waits for all worker threads to join.
        """
        # Block Logic: Block until all tasks in the queue are processed.
        self.queue.join()

        # Block Logic: Put sentinel values into the queue to signal each worker thread to terminate.
        for _ in xrange(len(self.threads)):
           self.queue.put((None, None, None))

        # Block Logic: Wait for all worker threads to terminate.
        for thread in self.threads:
            thread.join()


from threading import Thread, Event, Lock, Semaphore


class ReusableBarrier():
    """
    @brief Implements a reusable barrier for thread synchronization.

    This barrier allows a fixed number of threads to wait until all have reached a certain point
    before any are allowed to proceed. It uses a two-phase approach (implemented by the generic
    `phase` method) to ensure reusability without deadlocks. The counters are wrapped in lists
    to allow modification within nested scopes (e.g., `with` statements).
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier with a specified number of threads.

        @param num_threads: The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # @brief Counter for the first phase of the barrier, wrapped in a list.
        self.count_threads1 = [self.num_threads]
        # @brief Counter for the second phase of the barrier, wrapped in a list.
        self.count_threads2 = [self.num_threads]
        # @brief Lock to protect access to the thread counters.
        self.count_lock = Lock()                 
        # @brief Semaphore for the first phase of waiting threads.
        self.threads_sem1 = Semaphore(0)         
        # @brief Semaphore for the second phase of waiting threads.
        self.threads_sem2 = Semaphore(0)         
 
    def wait(self):
        """
        @brief Blocks until all participating threads have reached this point.

        This method orchestrates the two phases of the barrier to ensure all threads
        synchronize before proceeding, allowing for reusability.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        @brief Implements a single phase of the barrier synchronization.

        Threads decrement a shared counter. The last thread to reach zero releases all
        waiting threads in this phase and resets the counter for the next cycle.
        
        @param count_threads: A list containing the counter for the current phase.
        @param threads_sem: The semaphore associated with the current phase.
        Invariant: All threads must pass through this phase before any can proceed
                   if it's the first phase, or before the barrier can be reused
                   if it's the second phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                # Block Logic: Release all threads waiting in this phase.
                for i in range(self.num_threads):
                    threads_sem.release()        
                # Inline: Reset the counter for the next cycle.
                count_threads[0] = self.num_threads  
        threads_sem.acquire()

class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.

    This class manages sensor data, stores assigned scripts, and coordinates
    synchronization and data access using shared locks (per location) and
    a shared barrier for timepoint processing.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's sensor readings.
                            Keys are locations, values are data.
        @param supervisor: A reference to the supervisor object that manages the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # @brief Event to signal when a script has been received (not directly used as an Event in this version).
        self.script_received = Event()
        # @brief List to store assigned scripts and their locations.
        self.scripts = []
        # @brief Event to signal that processing for a timepoint is done.
        self.timepoint_done = Event()
        # @brief Synchronization barrier for coordinating timepoints across devices.
        self.barrier = None
        # @brief Dictionary of locks, one for each location, to protect sensor data access.
        self.location_locks = {location: Lock() for location in self.sensor_data}
        # @brief Flag to indicate if scripts are available for processing in the current timepoint.
        self.scripts_available = False
        # @brief The thread responsible for running the device's main logic.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up and distributes the shared synchronization barrier.

        Only the device with `device_id == 0` initializes the `ReusableBarrier`.
        Other devices then reference this shared barrier.

        @param devices: A list of all Device objects in the network.
        """
        # Block Logic: Only Device 0 initializes the shared barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            # Block Logic: Distribute the initialized barrier to other devices.
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script is provided, it's added to the list of scripts and the `scripts_available`
        flag is set to True. If no script is provided (None), it signals that script
        assignment for the current timepoint is complete by setting `timepoint_done`.

        @param script: The script object to be executed, or None to signal completion.
        @param location: The data location (index) the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True # Inline: Indicate that new scripts are available.
        else:
            self.timepoint_done.set() # Inline: Signal that script assignment for the timepoint is complete.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        Acquires a lock for the specific location before accessing data to ensure thread safety.
        Note: The lock is acquired here but released in `set_data`, implying a specific
              workflow where `set_data` follows `get_data` for the same location.

        @param location: The data location (index) to retrieve data from.
        @return The sensor data at the specified location, or None if not found.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire() # @brief Acquire lock for thread-safe data access.
            return self.sensor_data[location]     
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.

        Releases the lock for the specific location after modifying data.
        This release must correspond to a prior acquire in `get_data`.

        @param location: The data location (index) to set data for.
        @param data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release() # @brief Release lock after data modification.
        else:
            return None

    def shutdown(self):
        """
        @brief Shuts down the device's main thread, ensuring proper termination.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief The main execution thread for a Device.

    This thread manages the device's operational lifecycle, including fetching
    neighbor information, managing script execution through a `MyQueue` thread pool,
    and synchronizing with other devices using barriers.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = MyQueue(8) # @brief Custom thread pool for executing scripts, with 8 worker threads.

    def run(self):
        """
        @brief The main execution loop for the device thread.

        This loop continuously performs the following actions:
        1. Sets its associated `MyQueue`'s device reference.
        2. Fetches current neighbor information from the supervisor.
        3. Terminates if no neighbors are returned (simulation end).
        4. Enters a nested loop to handle script assignment and timepoint completion:
           - If `scripts_available` is True, it processes all assigned scripts,
             putting them into the `MyQueue` for execution.
           - If `timepoint_done` event is set, it breaks from the nested loop,
             clears the event, and resets `scripts_available`.
        5. Waits for all tasks in `MyQueue` to complete.
        6. Waits on the `device.barrier` to synchronize with other devices.
        7. On loop termination (no neighbors), it gracefully shuts down the `MyQueue`.
        Invariant: The device processes data in discrete timepoints, synchronizing with the network
                   after each timepoint.
        """
        self.queue.device = self.device # @brief Set the device reference for the MyQueue worker threads.
        while True:
            # Block Logic: Fetch updated neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Inline: Terminate thread if no more neighbors (simulation end).

            # Block Logic: Loop to handle script assignment and timepoint completion events.
            while True:
                # Pre-condition: Either new scripts are available, or the timepoint_done event is set.
                if self.device.scripts_available or self.device.timepoint_done.wait():
                    # Block Logic: If scripts are available, put them into the queue for processing.
                    if self.device.scripts_available:
                        self.device.scripts_available = False # Inline: Reset flag after processing.

                        # Block Logic: Add each assigned script as a task to the MyQueue.
                        for (script, location) in self.device.scripts:
                            self.queue.queue.put((neighbours, script, location))
            
                    # Block Logic: If timepoint_done event is set (meaning no more scripts for this timepoint).
                    else:
                        self.device.timepoint_done.clear() # Inline: Clear the event for the next timepoint.
                        self.device.scripts_available = True # Inline: Reset flag.
                        break # Inline: Exit nested loop to proceed with barrier synchronization.
            
            # Block Logic: Wait for all tasks in the MyQueue to be completed.
            self.queue.queue.join()
            # Block Logic: Wait on the barrier to synchronize with all other devices in the network.
            self.device.barrier.wait()

        # Block Logic: When the main loop terminates (neighbours is None), shut down the MyQueue.
        self.queue.finish()
