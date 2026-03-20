"""
This module defines classes for managing distributed devices in a sensor network,
including a reusable barrier for synchronization and a multi-threaded device
implementation for script execution and data sharing among neighbors.

The core components are:
- RBarrier: Implements a reusable barrier synchronization mechanism.
- Device: Represents an individual device in the network, managing its state,
          sensor data, and interactions with a supervisor and other devices.
- DeviceThread: A dedicated thread for each device to handle script execution
                and communication with neighboring devices.
"""

from threading import Lock, Semaphore, Thread, Event
from Queue import Queue

class RBarrier(object):
    """
    A reusable barrier synchronization mechanism for multiple threads.
    This barrier ensures that all participating threads reach a specific
    point of execution before any of them are allowed to proceed.
    It uses a double-barrier pattern to allow for reuse without re-initialization.

    Algorithm: Count-based semaphore signaling with two phases for reusability.
    Time Complexity: O(num_threads) for each `wait()` operation in the worst case (signaling all semaphores).
    Space Complexity: O(1) beyond thread-local storage.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the RBarrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that will participate in this barrier.
        """
        self.num_threads = num_threads
        # count_threads1 and count_threads2 are lists to allow mutation within the Lock context.
        # They track how many threads have reached the current phase of the barrier.
        self.count_threads1 = [self.num_threads] 
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock() # Protects access to the count_threads variables.
        # Semaphores used to block threads until all have arrived.
        # Initialized to 0 so threads will block immediately upon acquiring.
        self.threads_sem1 = Semaphore(0) 
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` threads have called `wait()`.
        This method alternates between two internal phases (using threads_sem1 and threads_sem2)
        to enable reusability of the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the current count of threads for this phase.
            threads_sem (Semaphore): The semaphore associated with this phase.
        
        Block Logic: Decrements the thread count. If the count reaches zero,
                     it means all threads have arrived, and the semaphore is
                     released `num_threads` times, unblocking all waiting threads.
                     The count is then reset for the next use.
        Pre-condition: `count_threads` accurately reflects the number of threads
                       yet to arrive in this phase.
        Invariant: All threads block at `threads_sem.acquire()` until `count_threads[0]`
                   reaches 0, at which point all are released.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # All threads have arrived; release them all.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset the count for the next iteration of the barrier.
                count_threads[0] = self.num_threads
        # Block until all other threads have arrived and released the semaphore.
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device in a distributed sensor network.
    Each device has an ID, sensor data, and interacts with a supervisor
    and other devices. It can receive and execute scripts.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): Unique identifier for this device.
            sensor_data (dict): Dictionary storing sensor data for various locations.
            supervisor (Supervisor): A reference to the central supervisor object
                                     for network-wide operations.
        
        Functional Utility: Sets up the device's unique state and prepares
                            it for distributed operations.
        """
        # Dictionary to store locks for specific data locations, ensuring data integrity.
        self.location_lock = {} 
        self.barrier = None # Reference to a shared RBarrier for synchronization.
        self.all_devices = [] # List of all devices in the network.
        self.update = Lock() # A lock to protect shared updates, particularly for setup.
        self.got_data = False # Flag indicating if new data has been received for processing.
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when a new script has been assigned.
        self.scripts = [] # List to store assigned scripts (script, location) tuples.
        self.timepoint_done = Event() # Event to signal when the device has completed its operations for a timepoint.
        # Each device runs its logic in a separate thread.
        self.thread = DeviceThread(self)
        self.thread.start() # Start the dedicated device thread.

    def __str__(self):
        """
        Returns a string representation of the Device.
        Functional Utility: Provides a human-readable identifier for the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the network-wide shared resources (like the barrier and update lock).
        This method should typically be called once by a designated 'master' device (device_id 0).

        Args:
            devices (list): A list of all Device instances in the network.
        
        Functional Utility: Initializes the synchronization primitives that are
                            shared across all devices, ensuring consistent state.
        Block Logic: If this is the master device (device_id 0), it initializes
                     the RBarrier and propagates the shared barrier and update lock
                     references to all other devices.
        Pre-condition: This method is called exactly once by the master device.
        """
        self.all_devices = devices
        
        if self.device_id == 0: # Master device initializes shared resources
            self.barrier = RBarrier(len(self.all_devices))
            self.update = Lock() # Re-initialize to ensure it's the shared lock.
            for i in xrange(0, len(self.all_devices)):
                self.all_devices[i].barrier = self.barrier
                self.all_devices[i].update = self.update

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific data location on this device.

        Args:
            script (Script): The script object to be executed.
            location (str): The identifier for the data location the script pertains to.
        
        Functional Utility: Queues up new processing tasks for the device's thread.
        Block Logic: If a new location is encountered, a shared lock for that location
                     is initialized and propagated to all devices. The script is then
                     appended to the device's script list, and the `script_received`
                     event is set to wake up the `DeviceThread`.
        Pre-condition: `script` is a valid executable script object, and `location`
                       is a string identifier.
        """
        if script is not None:
            if location not in self.location_lock:
                self.got_data = True # Indicates new data might be available for processing
                self.location_lock[location] = Lock() # Create a lock for this specific data location
                
                # Propagate the new location lock to all other devices to ensure consistency.
                with self.update:
                    for i in xrange(0, len(self.all_devices)):
                        self.all_devices[i].location_lock[
                            location] = self.location_lock[location]

            if location in self.location_lock:
                self.scripts.append((script, location))
                self.script_received.set() # Signal that a new script is available
        else:
            # If no script is provided, it indicates the end of a timepoint's tasks.
            self.timepoint_done.set()


    def get_data(self, location):
        """
        Retrieves sensor data for a specified location.

        Args:
            location (str): The identifier for the data location.

        Returns:
            Any: The sensor data at the specified location, or None if not found.
        Functional Utility: Provides read access to the device's internal sensor data.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None


    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location.

        Args:
            location (str): The identifier for the data location.
            data (Any): The new sensor data to set.
        Functional Utility: Allows modification of the device's internal sensor data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Waits for the device's processing thread to complete its execution.
        Functional Utility: Ensures a clean shutdown of the device's operations.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    A dedicated thread for each Device to manage script execution and
    communication with neighboring devices. It uses a producer-consumer
    pattern with a Queue for scripts and worker threads for processing.
    """
    
    THREAD_NUMBER = 8 # Number of worker threads to use for parallel script execution.
    STOP_FLAG = "STOP" # Special flag to signal worker threads to terminate.

    def __init__(self, device):
        """
        Initializes the DeviceThread with a reference to its parent Device.

        Args:
            device (Device): The Device instance this thread belongs to.
        
        Functional Utility: Sets up the worker thread pool and the script queue
                            for asynchronous processing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_number = DeviceThread.THREAD_NUMBER
        # Queue to hold script execution tasks. Max size limits pending tasks.
        self.queue = Queue(self.thread_number) 
        # Create and start worker threads. Each worker will call `thread_func`.
        self.threads = [Thread(target=self.thread_func) for _ in [None]*self.thread_number]
        _ = [x.start() for x in self.threads]
        self.debug_timepoint = 0 # Counter for debugging timepoints.
        self.transmit = 0 # Counter for transmissions (likely data pushes).

    def run(self):
        """
        The main execution loop for the DeviceThread.
        This loop continuously fetches scripts, dispatches them to worker threads,
        and synchronizes with other device threads using a barrier.
        Functional Utility: Manages the lifecycle of script processing for the device.
        """
        while True:
            # Block Logic: Retrieves neighbors from the supervisor. If no neighbors,
            #              it indicates the end of simulation/processing.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Exit loop if no more neighbors (end of simulation)
            
            while True:
                # Block Logic: Waits for a timepoint to be done (either scripts assigned or explicitly marked).
                self.device.timepoint_done.wait()
                if not self.device.got_data:
                    # If no new data was received, clear the timepoint done event and break.
                    self.device.timepoint_done.clear()
                    self.transmit += 1
                    self.device.got_data = True # Reset flag
                    break
                else:
                    # Functional Utility: Enqueues all received scripts for processing by worker threads.
                    # Each script becomes a task (neighbours, script_obj, location) in the queue.
                    _ = [self.queue.put((neighbours, x[0], x[1])) \
                     for x in self.device.scripts]
                    self.device.got_data = False # Reset flag after queuing scripts
            
            self.debug_timepoint += 1 # Increment timepoint counter for debugging.
            self.queue.join() # Block until all tasks in the queue are processed by worker threads.
            self.device.barrier.wait() # Synchronize with other devices at the barrier.

        # Shutdown sequence:
        self.queue.join() # Ensure all pending tasks are completed.
        # Functional Utility: Puts STOP_FLAG into the queue for each worker thread to signal termination.
        _ = [self.queue.put((
            DeviceThread.STOP_FLAG, \
            DeviceThread.STOP_FLAG, \
            DeviceThread.STOP_FLAG)) \
            for _ in xrange(self.THREAD_NUMBER)]
        # Wait for all worker threads to terminate.
        _ = [x.join() for x in self.threads]

    def thread_func(self):
        """
        The main function executed by each worker thread.
        It continuously retrieves script tasks from the queue, executes them,
        and propagates results to local and neighboring devices.
        Functional Utility: Performs the actual data processing and communication for assigned scripts.
        """
        # Initially get a task from the queue.
        neighbours, script, location = self.queue.get()
        while neighbours is not DeviceThread.STOP_FLAG: # Loop until STOP_FLAG is received
            script_data = []
            # Block Logic: Acquire a lock for the specific data location to ensure exclusive access.
            # Pre-condition: `location_lock[location]` is a valid Lock object.
            with self.device.location_lock[location]:
                # Block Logic: Collects relevant data from neighboring devices for the current location.
                for device in neighbours:
                    if device.device_id != self.device.device_id: # Avoid getting data from self again
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                # Also collect local data for the script execution.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Functional Utility: Executes the script with the collected data.
                    result = script.run(script_data)
                    # Block Logic: Propagates the script's result to all neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Updates the local device's data with the script's result.
                    self.device.set_data(location, result)
            
            self.queue.task_done() # Mark the current task as done.
            # Get the next task from the queue.
            neighbours, script, location = self.queue.get()
        self.queue.task_done() # Mark the final STOP_FLAG task as done.

