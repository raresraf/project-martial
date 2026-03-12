


"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `ReusableBarrier`: A reusable barrier synchronization mechanism using semaphores.
- `Device`: Represents a single device, managing sensor data and orchestrating operations.
- `WorkerThread`: A worker thread that consumes scripts from a shared queue, executes them,
  and updates device data.
- `DeviceThread`: The main thread for a `Device`, acting as a producer of scripts for `WorkerThread`s,
  and managing overall timepoint synchronization.

The system utilizes Python's `Queue` for inter-thread communication, `threading.Lock`
for protecting shared resources, `threading.Event` for signaling, and a custom
`ReusableBarrier` for coordinating thread execution across devices and within a single device.
"""

from Queue import Queue
from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    A reusable barrier synchronization mechanism for multiple threads using semaphores.
    This barrier allows a fixed number of threads to wait at a synchronization point,
    and once all threads arrive, they are all released simultaneously. The counters
    are stored in lists (`count_threads1`, `count_threads2`) to allow for reusability
    of the barrier instance across multiple `wait` calls.
    """
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that will participate
                               in the barrier synchronization.
        """
        self.num_threads = num_threads
        # Counters for the two phases of the barrier. Stored in lists to be mutable
        # when passed to the 'phase' method, enabling reusability.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # Lock to protect the shared counters during decrements and resets.
        self.count_lock = Lock()
        # Semaphores for the two phases of threads to wait on.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait until all threads have reached this barrier.
        This method orchestrates a two-phase synchronization to ensure reusability
        without deadlocks.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the current count of threads
                                  remaining for this phase.
            threads_sem (Semaphore): The semaphore threads wait on for this phase.
        """
        with self.count_lock: # Protect shared counter access.
            count_threads[0] -= 1 # Decrement the count of threads remaining.
            if count_threads[0] == 0: # If this is the last thread to arrive:
                for _ in range(self.num_threads):
                    threads_sem.release() # Release all waiting threads by incrementing the semaphore.
                count_threads[0] = self.num_threads # Reset counter for next use.
        threads_sem.acquire() # Wait (decrement) the semaphore, blocking until released by the last thread.


class Device(object):
    """
    Represents a single device within a simulated distributed environment.
    Each device manages its own sensor data, communicates with a supervisor,
    and orchestrates multi-threaded script execution. It uses a `Queue`
    to distribute scripts to worker threads and a shared `ReusableBarrier`
    for inter-device synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for the device.
            supervisor (object): A reference to a supervisor object for inter-device communication.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to hold assigned scripts (tuples of (script, location)).
        self.timepoint_done = Event() # Event to signal that the current timepoint's processing is complete.

        
        self.barrier = None # Placeholder for the global ReusableBarrier, set in setup_devices.
        self.queue = Queue() # A queue to distribute scripts to WorkerThread instances.
        self.workers = [WorkerThread(self) for _ in range(8)] # A list of 8 WorkerThread instances.

        
        self.thread = DeviceThread(self) # The main orchestrating thread for this device.
        self.thread.start() # Start the DeviceThread.

        
        for thread in self.workers:
            thread.start() # Start all WorkerThread instances.

    def __str__(self):
        """
        Returns a string representation of the device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the global barrier and shared location-specific locks across all devices.
        This method is designed to be called only by the device with `device_id == 0`.
        It creates a single `ReusableBarrier` and a dictionary of `Lock`s for each unique
        data location found across all devices.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        # Block Logic: Only the device with device_id 0 performs this setup.
        if self.device_id == 0:
            # Inline: Creates a global ReusableBarrier for synchronization among all DeviceThreads.
            barrier = ReusableBarrier(len(devices))

            locks = {} # Dictionary to store shared locks for each data location.

            # Block Logic: Initialize a unique `Lock` for each distinct data location present across all devices.
            for device in devices:
                for location in device.sensor_data:
                    if not location in locks:
                        locks[location] = Lock()

            # Block Logic: Distributes the created global `barrier` and `locks` to all devices.
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """
        Assigns a script to the device. The script is added to the internal list of scripts.
        If `script` is None, it signals that the timepoint's script assignment is complete.

        Args:
            script (object): The script object to be executed. If None, it signals timepoint completion.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
        else:
            self.timepoint_done.set() # If no script, signal that the timepoint's script assignment is done.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's `sensor_data` dictionary.

        Args:
            location (int): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location in this device's `sensor_data` dictionary.
        The data is updated only if the location exists in the `sensor_data`.

        Args:
            location (int): The location identifier for which to set data.
            data (any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device by waiting for its main `DeviceThread` to complete.
        """
        self.thread.join() # Wait for the DeviceThread to finish its execution.


class WorkerThread(Thread):
    """
    A worker thread that continuously consumes scripts from its parent `Device`'s queue.
    For each script, it collects necessary data from the device and its neighbors,
    executes the script, and updates the devices' sensor data, using location-specific locks.
    """
    def __init__(self, device):
        """
        Initializes a `WorkerThread` instance.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
        """
        Thread.__init__(self)
        self.device = device

        def run(self):

            """

            The main execution method for the `WorkerThread`.

            It continuously fetches work items (scripts and locations) from the device's queue.

            If a `None` item is received, it signals the thread to terminate.

            Otherwise, it executes the script, collects data from relevant devices,

            and updates their sensor data, all while ensuring thread safety through locks.

            """

            while True:

                item = self.device.queue.get() # Inline: Retrieve a work item from the shared queue.

                if item is None: # Inline: `None` is a sentinel value indicating that the thread should terminate.

                    break # Exit the loop.

    

                (script, location) = item # Unpack the work item into script and location.

    

                # Block Logic: Acquire the location-specific lock to ensure exclusive access

                # to data at this `location` across all devices during script execution and data update.

                with self.device.locks[location]:

                    script_data = [] # List to collect input data for the script.

    

                    # Block Logic: Collect data from all neighboring devices at the specified location.

                    for device in self.device.neighbours:

                        data = device.get_data(location) # Get data from the neighbor.

                        if data is not None:

                            script_data.append(data) # Add to script input if available.

    

                    # Block Logic: Collect data from this worker's own parent device at the specified location.

                    data = self.device.get_data(location)

                    if data is not None:

                        script_data.append(data) # Add to script input if available.

    

                    # Block Logic: If input data is available, execute the script and update device data.

                    if script_data != []:

                        # Inline: Execute the script's `run` method with the collected data.

                        result = script.run(script_data)

    

                        # Block Logic: Update sensor data for all involved devices (neighbors and self) with the result.

                        for device in self.device.neighbours:

                            device.set_data(location, result) # Update neighbor's data.

    

                        self.device.set_data(location, result) # Update this device's own data.

    

                self.device.queue.task_done() # Inline: Signal that the current task from the queue has been completed.

    

    class DeviceThread(Thread):

    

        """

    

        The main orchestrating thread for a `Device`.

    

        This thread acts as a producer of work for `WorkerThread`s, fetching neighbor

    

        information, distributing scripts to the device's queue, and managing

    

        overall timepoint synchronization using a global barrier. It also handles

    

        the shutdown of its associated worker threads.

    

        """

    

    

    

        def __init__(self, device):

    

            """

    

            Initializes a `DeviceThread` instance.

    

    

    

            Args:

    

                device (Device): The parent `Device` object this thread belongs to.

    

            """

    

            

    

            Thread.__init__(self, name="Device Thread %d" % device.device_id)

    

            self.device = device

    def run(self):
        """
        The main execution loop for the `DeviceThread`.
        It continuously fetches neighbor data, waits for timepoint readiness,
        puts assigned scripts into the queue for `WorkerThread`s,
        waits for all scripts to be processed, and synchronizes with other devices.
        It also manages the shutdown of its associated worker threads.
        """
        while True:

            # Block Logic: Fetch neighbor information from the supervisor.
            # This is crucial for workers to collect data from adjacent devices.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # Inline: If `neighbours` is None, it signals termination for the entire device.
            if self.device.neighbours is None:
                break # Exit the loop, initiating the shutdown sequence.

            # Block Logic: Wait for the `timepoint_done` event to be set,
            # indicating that all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Clear the event for the next timepoint.

            # Block Logic: Put all assigned scripts into the queue for processing by `WorkerThread`s.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location))

            # Block Logic: Wait until all tasks (scripts) currently in the queue have been processed
            # by the `WorkerThread`s. `queue.join()` blocks until `task_done()` has been called
            # for every item previously put into the queue.
            self.device.queue.join()

            # Block Logic: Synchronize all `DeviceThread` instances across all `Device`s.
            # This ensures all devices have completed their timepoint processing before proceeding.
            self.device.barrier.wait()

        # Block Logic: Shutdown sequence for worker threads.
        # After the main loop breaks (due to `neighbours` being None),
        # put `None` into the queue for each worker to signal their termination.
        for _ in range(8):
            self.device.queue.put(None)

        # Block Logic: Wait for all `WorkerThread`s to finish their execution.
        for thread in self.device.workers:
            thread.join()
