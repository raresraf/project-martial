

from Queue import Queue 
from threading import Thread

class MyQueue():
    """
    A custom multi-threaded queue implementation designed to distribute tasks
    (scripts for devices) among a fixed pool of worker threads.

    It manages a set of threads that continuously fetch tasks from an internal
    queue, process them, and signal completion. It also provides mechanisms
    for graceful shutdown.
    """
    
    def __init__(self, num_threads):
        """
        Initializes MyQueue with a specified number of worker threads.

        Args:
            num_threads (int): The number of worker threads to create and manage.
        """
        # Initialize an internal Queue to hold tasks.
        self.queue = Queue(num_threads)
        # List to store references to the worker threads.
        self.threads = []
        # Reference to the Device object that this queue is serving.
        self.device = None

        # Create and start the specified number of worker threads.
        for _ in xrange(num_threads):
            # Each thread's target function is 'self.run'.
            thread = Thread(target=self.run)
            self.threads.append(thread)
        
        # Start all the worker threads.
        for thread in self.threads:
            thread.start()
    
    def run(self):
        """
        The main execution loop for each worker thread.

        Each thread continuously retrieves tasks from the queue. A task
        consists of (neighbours, script, location). If a sentinel value
        (None, None, None) is received, the thread terminates. Otherwise,
        it collects data, executes the script, updates relevant device data,
        and marks the task as done in the queue.
        """
        while True:
            # Get a task from the queue. This call blocks until an item is available.
            neighbours, script, location = self.queue.get()

            # Check for the sentinel value to signal thread termination.
            if neighbours is None and script is None:
                self.queue.task_done() # Mark task as done before exiting.
                return # Terminate the thread.
        
            script_data = [] # List to store data collected for the script.
            
            # Collect data from neighboring devices.
            for device in neighbours:
                # Exclude the current device itself if it's in the neighbours list.
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            # Collect data from the current device for the specific location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # If there's collected data, execute the script.
            if script_data != []:
                # Execute the script with the collected data.
                result = script.run(script_data)
                
                # Update data in neighboring devices.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                # Update data in the current device.
                self.device.set_data(location, result)
            
            # Mark the task as done in the queue, allowing queue.join() to track completion.
            self.queue.task_done()
    
    def finish(self):
        """
        Gracefully shuts down the queue and all worker threads.

        It first waits for all tasks currently in the queue to be processed,
        then puts sentinel values into the queue for each worker thread to
        signal their termination, and finally joins all worker threads.
        """
        # Block until all tasks in the queue are processed.
        self.queue.join()

        # Put sentinel values into the queue to make each worker thread terminate.
        for _ in xrange(len(self.threads)):
           self.queue.put((None, None, None))

        # Wait for all worker threads to complete their execution (i.e., process their sentinel).
        for thread in self.threads:
            thread.join()


from threading import Thread, Event, Lock, Semaphore
from MyQueue import MyQueue

class ReusableBarrier():
    """
    A reusable barrier synchronization primitive for coordinating multiple threads.
    This barrier allows a fixed number of threads to wait until all threads
    have reached a specific point, and then resets itself for reuse.
    It utilizes a two-phase semaphore-based mechanism to ensure correct
    operation across multiple waits.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that must reach the
                                barrier before any can proceed.
        """
        self.num_threads = num_threads
        # Tracks the number of threads waiting in the first phase of the barrier.
        self.count_threads1 = [self.num_threads] # Using a list to make it mutable within closures/nested scopes.
        # Tracks the number of threads waiting in the second phase of the barrier.
        self.count_threads2 = [self.num_threads] 
        # A lock to protect access to the thread count.
        self.count_lock = Lock()                 
        # Semaphore for releasing threads in the first phase.
        self.threads_sem1 = Semaphore(0)         
        # Semaphore for releasing threads in the second phase.
        self.threads_sem2 = Semaphore(0)         
 
    def wait(self):
        """
        Causes the calling thread to wait until all other threads have also
        called this method. This method orchestrates the two phases of the
        barrier to ensure proper synchronization and reusability.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        Manages a single phase of the barrier synchronization.

        Threads decrement a shared counter. The last thread to reach zero
        releases all waiting threads using a semaphore, and then resets
        the counter for the next use.

        Args:
            count_threads (list): A mutable list containing the current count
                                  of threads remaining in this phase.
            threads_sem (Semaphore): The semaphore used to release threads
                                     once the count reaches zero.
        """
        with self.count_lock:
            # Decrement the count of threads waiting in this phase.
            count_threads[0] -= 1
            # If this is the last thread to arrive in this phase.
            if count_threads[0] == 0:            
                # Release all waiting threads by calling release() num_threads times.
                for i in range(self.num_threads):
                    threads_sem.release()        
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads  
        # Acquire the semaphore, effectively waiting until all threads are released by the last thread.
        threads_sem.acquire()

class Device(object):
    """
    Represents a simulated device within a multi-device system.

    Each device manages its own sensor data, processes assigned scripts,
    and coordinates with other devices through a shared barrier and
    location-specific locks for data consistency.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor readings,
                                 where keys are locations and values are data.
            supervisor (Supervisor): A reference to the supervisor object
                                     that manages overall simulation and device interactions.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that scripts have been received for the current timepoint.
        self.script_received = Event()
        # List of scripts assigned to this device for current processing round.
        self.scripts = []
        # Event to signal that all scripts for the current timepoint are ready.
        self.timepoint_done = Event()
        # Reference to the shared reusable barrier for inter-device synchronization.
        self.barrier = None
        # Dictionary of Locks, one for each location in sensor_data, to protect data access.
        self.location_locks = {location: Lock() for location in self.sensor_data}
        # Flag indicating if new scripts are available for processing.
        self.scripts_available = False
        # The dedicated thread for this device's main operational logic.
        self.thread = DeviceThread(self)
        # Start the device's operational thread upon initialization.
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures shared resources, specifically the synchronization barrier,
        for all devices in the system.

        If this device is device_id 0, it initializes the `ReusableBarrier`
        and shares it with all other devices.

        Args:
            devices (list): A list of all `Device` instances in the system.
        """
        # Only device with device_id 0 is responsible for initializing the shared barrier.
        if self.device_id == 0:
            # Create a new reusable barrier with the total number of devices.
            self.barrier = ReusableBarrier(len(devices))
            # Share the initialized barrier with all other devices.
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be processed by this device.

        If a script is provided, it's added to the device's list of scripts,
        and the `scripts_available` flag is set to True.
        If `script` is None, it signals that script assignment for the current
        timepoint is complete, and the `timepoint_done` event is set.

        Args:
            script (object or None): The script object to assign, or None to signal
                                     the end of script assignments for a timepoint.
            location (int): The identifier for the location associated with the script.
        """
        # If a script is provided, add it to the temporary scripts list.
        if script is not None:
            self.scripts.append((script, location))
            # Indicate that there are scripts available for processing.
            self.scripts_available = True
        else:
            # If script is None, signal that all scripts for this timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location, acquiring a lock for thread safety.

        Args:
            location (int): The identifier for the location for which to retrieve data.

        Returns:
            Any: The sensor data if the location exists in sensor_data, otherwise None.
        """
        # Check if the location exists in the sensor data.
        if location in self.sensor_data:
            # Acquire the lock associated with this location to ensure exclusive access.
            self.location_locks[location].acquire()
            return self.sensor_data[location]     
        else:
            return None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location, releasing the associated lock.

        Args:
            location (int): The identifier for the location where the data should be set.
            data (Any): The new data to set for the location.
        """
        # Check if the location exists in the sensor data.
        if location in self.sensor_data:
            self.sensor_data[location] = data
            # Release the lock associated with this location after data has been updated.
            self.location_locks[location].release()
        else:
            return None

    def shutdown(self):
        """
        Shuts down the device by joining its associated thread.
        This ensures that the device's main thread completes its execution.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    Manages the script execution rounds for a Device using a `MyQueue`
    for concurrent processing.

    This thread continuously fetches neighbor information, dispatches
    assigned scripts to the `MyQueue` for worker threads to process,
    and synchronizes with other DeviceThreads using a shared barrier.
    """
    
    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Initialize a MyQueue with 8 worker threads for concurrent script execution.
        self.queue = MyQueue(8)

    def run(self):
        """
        The main execution loop for the DeviceThread.

        Each iteration represents a processing round. It performs the following steps:
        1. Sets its associated `MyQueue`'s device.
        2. Retrieves updated neighbor information from the supervisor.
        3. If no neighbors are returned (e.g., simulation end), the loop breaks.
        4. Enters a sub-loop to handle script assignment:
           - If `scripts_available` is true, it means new scripts are ready. These are
             then put into the `MyQueue` for processing.
           - If `timepoint_done` is set (meaning all scripts have been collected),
             it clears events and breaks the sub-loop.
        5. Waits for all tasks in `MyQueue` to complete.
        6. Synchronizes with other DeviceThreads using the shared `barrier`.
        7. If the main loop breaks (no neighbors), it calls `queue.finish()` to
           gracefully shut down its worker threads.
        """
        # Assign the device to the MyQueue so worker threads can access its data.
        self.queue.device = self.device
        while True:
            # Retrieve updated neighbor information from the supervisor for the current round.
            neighbours = self.device.supervisor.get_neighbours()
            # If supervisor returns None, it signals the simulation to terminate.
            if neighbours is None:
                break

            # Block Logic: Handle script assignment and dispatch to the queue.
            # Invariant: This loop continues until all scripts for the current timepoint
            # have been dispatched and the timepoint is marked as done.
            while True:
                # If scripts are available OR timepoint_done is set (meaning no more scripts
                # for this timepoint but previous ones need processing).
                if self.device.scripts_available or self.device.timepoint_done.wait():
                    if self.device.scripts_available:
                        self.device.scripts_available = False # Reset flag.

                        # Put all assigned scripts into the MyQueue for processing.
                        for (script, location) in self.device.scripts:
                            self.queue.queue.put((neighbours, script, location))
            
                    else:
            
                        # Clear the timepoint_done event for the next cycle.
                        self.device.timepoint_done.clear()
                        # Reset scripts_available flag for the next cycle (to be set by assign_script).
                        self.device.scripts_available = True
                        break # Exit inner while loop.
            
            # Wait until all tasks (scripts) currently in the queue have been processed.
            self.queue.queue.join()
            # Wait at the shared barrier to synchronize with all other devices.
            self.device.barrier.wait()

        # After the main loop breaks (simulation ends), gracefully shut down the MyQueue workers.
        self.queue.finish()
