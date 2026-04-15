
"""
@bb7735cb-1a9b-45f7-8516-701f69a3ac2b/MyQueue.py
@brief Implements a thread-safe queue system for distributed device processing and a reusable barrier for synchronization.
This module provides a custom queue (`MyQueue`) designed to handle concurrent tasks across multiple devices,
along with a `ReusableBarrier` for coordinating the execution of these threads.
It is designed to facilitate distributed simulations or data processing where devices need to share data and synchronize steps.

Domain: Concurrency, Distributed Systems, Simulation, Data Processing.
"""

from Queue import Queue 
from threading import Thread

class MyQueue():
    """
    @brief Manages a pool of worker threads to process tasks from a queue.
    This class orchestrates the execution of scripts on device data by distributing
    tasks to a fixed number of worker threads. It handles data retrieval from
    neighboring devices and the local device, script execution, and result propagation.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the MyQueue with a specified number of worker threads.
        @param num_threads: The number of worker threads to create and manage.
        Functional Utility: Sets up the task queue and initializes worker threads.
        """
        
        self.queue = Queue(num_threads)
        self.threads = []
        self.device = None

        # Block Logic: Initializes and stores worker threads.
        # Each thread will execute the `run` method.
        for _ in xrange(num_threads):
            thread = Thread(target=self.run)
            self.threads.append(thread)
        
        # Block Logic: Starts all worker threads, making them ready to process tasks.
        for thread in self.threads:
            thread.start()
    
    def run(self):
        """
        @brief The main loop for each worker thread.
        Functional Utility: Continuously retrieves tasks from the queue, processes them,
        and propagates results. A task consists of a set of neighboring devices,
        a script to execute, and a location for data access.
        """
        while True:
            # Block Logic: Retrieves a task from the queue.
            # Invariant: The queue either contains a valid task or a termination signal.
            neighbours, script, location = self.queue.get()

            # Block Logic: Checks for a termination signal.
            # If a termination signal (None, None, None) is received, the thread exits.
            if neighbours is None and script is None:
                self.queue.task_done()
                return
        
            script_data = []
            
            # Block Logic: Collects data from neighboring devices.
            # It iterates through all neighbors and retrieves data for the specified location,
            # excluding the current device itself to avoid self-interaction in this loop.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            # Block Logic: Collects data from the current device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script if there is any data to process.
            # Pre-condition: `script_data` must not be empty for script execution.
            if script_data != []:
                # Functional Utility: Runs the assigned script with the collected data.
                result = script.run(script_data)
                
                # Block Logic: Propagates the script result to neighboring devices.
                # It iterates through all neighbors and updates their data for the specified location,
                # excluding the current device itself.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                # Functional Utility: Updates the current device's data with the script result.
                self.device.set_data(location, result)
            
            # Functional Utility: Marks the current task as done in the queue.
            self.queue.task_done()
    
    def finish(self):
        """
        @brief Signals all worker threads to terminate and waits for their completion.
        Functional Utility: Ensures graceful shutdown of all worker threads by
        inserting termination signals into the queue and joining each thread.
        """
        
        # Block Logic: Waits for all tasks currently in the queue to be processed.
        self.queue.join()

        # Block Logic: Inserts termination signals into the queue for each worker thread.
        for _ in xrange(len(self.threads)):
           self.queue.put((None, None, None))

        # Block Logic: Waits for all worker threads to complete their execution (after processing termination signals).
        for thread in self.threads:
            thread.join()


from threading import Thread, Event, Lock, Semaphore
from MyQueue import MyQueue

class ReusableBarrier():
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.
    This barrier allows a fixed number of threads to wait for each other at a
    specific point in their execution, and then proceeds together. It can be
    reused multiple times after all threads have passed through.
    Algorithm: Double-counting semaphore-based barrier.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier for a specified number of threads.
        @param num_threads: The total number of threads that will participate in the barrier.
        Functional Utility: Sets up internal counters and semaphores required for barrier synchronization.
        """
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Inline: Counter for the first phase of the barrier.
        self.count_threads2 = [self.num_threads] # Inline: Counter for the second phase of the barrier.
        self.count_lock = Lock()                 # Inline: A lock to protect access to the thread counters.
        self.threads_sem1 = Semaphore(0)         # Inline: Semaphore for releasing threads in the first phase.
        self.threads_sem2 = Semaphore(0)         # Inline: Semaphore for releasing threads in the second phase.
 
    def wait(self):
        """
        @brief Blocks the calling thread until all other threads have also called wait.
        Functional Utility: Orchestrates the two-phase synchronization mechanism, ensuring
        all participating threads reach this point before any can proceed.
        """
        # Block Logic: Executes the first phase of the barrier.
        self.phase(self.count_threads1, self.threads_sem1)
        # Block Logic: Executes the second phase of the barrier.
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        @brief Implements one phase of the barrier synchronization.
        @param count_threads: A list containing the current count of threads waiting for this phase.
        @param threads_sem: The semaphore used to release threads once the count reaches zero.
        Block Logic: Decrements the thread count. When the count reaches zero,
        all waiting threads are released, and the count is reset for the next use.
        Invariant: At the entry, threads are waiting for their turn in the phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # Block Logic: Checks if this is the last thread to reach the barrier phase.
            if count_threads[0] == 0:            
                # Block Logic: Releases all waiting threads from the semaphore and resets the counter.
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  # Inline: Resets the counter for the next use of the barrier.
        # Functional Utility: Acquires the semaphore, blocking the thread until all threads have reached the barrier.
        threads_sem.acquire()

class Device(object):
    """
    @brief Represents a single device in a distributed system, managing its sensor data and scripts.
    This class encapsulates device-specific state, including its ID, sensor readings,
    and a list of scripts to be executed. It interacts with a supervisor for global
    context and uses a barrier for synchronization with other devices.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: Unique identifier for the device.
        @param sensor_data: A dictionary containing sensor readings, indexed by location.
        @param supervisor: A reference to the supervisor object managing all devices.
        Functional Utility: Sets up the device's state, including synchronization primitives
        and a dedicated thread for processing.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()       # Inline: Event to signal when a script has been assigned to this device.
        self.scripts = []                    # Inline: List to hold (script, location) tuples assigned to the device.
        self.timepoint_done = Event()        # Inline: Event to signal the completion of a timepoint's processing.
        self.barrier = None                  # Inline: Synchronization barrier for coordinating with other devices.
        # Functional Utility: Creates a lock for each data location to ensure thread-safe access to sensor data.
        self.location_locks = {location: Lock() for location in self.sensor_data}
        self.scripts_available = False       # Inline: Flag indicating if new scripts are available for processing.
        self.thread = DeviceThread(self)     # Inline: The dedicated thread for this device's operations.
        self.thread.start()                  # Functional Utility: Starts the device's dedicated processing thread.

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        Functional Utility: Provides a human-readable identifier for the device.
        """
        # Functional Utility: Formats the device ID into a descriptive string.
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the synchronization barrier across multiple devices.
        @param devices: A list of all participating Device objects.
        Block Logic: If this is the master device (device_id == 0), it initializes a
        ReusableBarrier and assigns it to itself and all other devices.
        Pre-condition: Called once all Device objects have been instantiated.
        """
        # Block Logic: Checks if the current device is the master device (ID 0).
        if self.device_id == 0:
            # Functional Utility: Initializes a ReusableBarrier with the total number of devices.
            self.barrier = ReusableBarrier(len(devices))
            # Block Logic: Assigns the same barrier instance to all other devices.
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location on the device.
        @param script: The script object to execute.
        @param location: The data location on which the script will operate.
        Block Logic: Appends the script and its target location to the device's script list.
        If no script is provided, it signals that the current timepoint is done.
        """
        # Block Logic: Checks if a script has been provided.
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True  # Inline: Sets flag to indicate new scripts are ready.
        else:
            self.timepoint_done.set()      # Inline: Signals that the current timepoint has completed all script assignments.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location, acquiring a lock for thread safety.
        @param location: The key identifying the sensor data to retrieve.
        Functional Utility: Ensures exclusive access to sensor data during read operations.
        Returns: The sensor data at the specified location, or None if the location is not found.
        """
        # Block Logic: Checks if the requested location exists in the sensor data.
        if location in self.sensor_data:
            self.location_locks[location].acquire() # Inline: Acquires a lock for the specific data location.
            return self.sensor_data[location]     
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a given location and releases the associated lock.
        @param location: The key identifying the sensor data to update.
        @param data: The new data value to set.
        Functional Utility: Writes new data to a specific location and releases the lock,
        making the data available to other threads.
        Returns: None if the location is not found, otherwise implicitly returns nothing.
        """
        # Block Logic: Checks if the requested location exists in the sensor data.
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release() # Inline: Releases the lock for the specific data location.
        else:
            return None

    def shutdown(self):
        """
        @brief Joins the device's processing thread, effectively stopping its operation.
        Functional Utility: Ensures that the DeviceThread completes its execution before the program exits.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief A dedicated thread for a Device to manage its task queue and synchronize with other devices.
    This thread continuously fetches tasks (scripts) for its associated device,
    manages the MyQueue for script execution, and synchronizes with other device threads
    using a barrier at the end of each processing timepoint.
    """
    

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread with its associated device.
        @param device: The Device object that this thread will manage.
        Functional Utility: Sets up the thread name and initializes a MyQueue for tasks.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = MyQueue(8) # Inline: Initializes a MyQueue with 8 worker threads for this device.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        Functional Utility: Coordinates the fetching of scripts, processing them via MyQueue,
        and synchronizing with other devices at each timepoint. It continues until the
        supervisor signals termination.
        """
        # Functional Utility: Associates the MyQueue instance with the current device.
        self.queue.device = self.device

        while True:
            # Block Logic: Continuously retrieves neighbors from the supervisor.
            # Invariant: If neighbors are None, it signals termination.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Processes scripts for the current timepoint.
            # Invariant: Loops until all scripts for the current timepoint are processed or signaled as done.
            while True:
                # Block Logic: Waits until new scripts are available or the timepoint is signaled as done.
                if self.device.scripts_available or self.device.timepoint_done.wait():
                    # Block Logic: If scripts are available, add them to the queue for processing.
                    if self.device.scripts_available:
                        self.device.scripts_available = False # Inline: Resets the flag after processing available scripts.

                        # Block Logic: Adds each assigned script to the MyQueue for execution.
                        for (script, location) in self.device.scripts:
                            self.queue.queue.put((neighbours, script, location))
            
                    else:
                        # Block Logic: If timepoint_done is set (meaning no more scripts for this timepoint),
                        # clear the event and prepare for the next timepoint.
                        self.device.timepoint_done.clear()
                        self.device.scripts_available = True # Inline: Resets for the next timepoint.
                        break
            
            # Functional Utility: Waits for all tasks related to the current timepoint to complete in MyQueue.
            self.queue.queue.join()
            # Functional Utility: Synchronizes with other device threads using the barrier.
            self.device.barrier.wait()

        # Functional Utility: Shuts down the internal MyQueue gracefully when the DeviceThread terminates.
        self.queue.finish()
