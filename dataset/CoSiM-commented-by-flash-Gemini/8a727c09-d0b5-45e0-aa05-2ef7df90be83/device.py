"""
@file device.py
@brief Implements Device and Threading logic for a simulated distributed system.

This module defines classes for managing simulated devices (`Device`),
a reusable barrier for thread synchronization (`ReusableBarrier`),
and thread classes (`DeviceThread`, `DeviceSubThread`) to handle
concurrent operations and script execution within the simulated environment.

The system simulates devices interacting with a supervisor and exchanging
sensor data at specific locations, utilizing a barrier for synchronized
progression through "timepoints" or simulation steps.
"""

from threading import Event, Thread, Semaphore, Lock



class ReusableBarrier(object):
    """
    @brief A reusable barrier synchronization primitive.

    This barrier allows a fixed number of threads to wait until all of them
    have reached a specific synchronization point. Once all threads arrive,
    they are all released simultaneously. The barrier can then be reset
    and reused for subsequent synchronization points.

    Algorithm: Two-phase barrier using semaphores and a counter.
    Time Complexity: O(N) for each `wait` call, where N is `num_threads`, due to semaphore releases.
    Space Complexity: O(1) beyond thread-local storage.
    """
    

    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier with a specified number of threads.
        @param num_threads The total number of threads that must reach the barrier.
        """
        # num_threads: Stores the total number of threads expected to participate in the barrier.
        self.num_threads = num_threads
        # count_threads1: Counter for the first phase of the barrier, initialized to num_threads.
        self.count_threads1 = [self.num_threads]
        # count_threads2: Counter for the second phase of the barrier, also initialized to num_threads.
        self.count_threads2 = [self.num_threads]
        # count_lock: A lock to protect access to the thread counters, ensuring atomic decrements.
        self.count_lock = Lock()
        # threads_sem1: Semaphore for blocking/unblocking threads in the first phase.
        self.threads_sem1 = Semaphore(0)
        # threads_sem2: Semaphore for blocking/unblocking threads in the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` threads have reached this point.
        Orchestrates the two-phase barrier mechanism to ensure reusability.
        """
        # Block Logic: Executes the first phase of the barrier.
        # This phase synchronizes threads before proceeding to the second phase.
        self.phase(self.count_threads1, self.threads_sem1)
        # Block Logic: Executes the second phase of the barrier.
        # This phase allows threads to proceed after the first phase is complete,
        # ensuring the barrier is reset and ready for reuse.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements a single phase of the barrier synchronization.
        @param count_threads A list containing the counter for the current phase.
        @param threads_sem The semaphore associated with the current phase.

        Logic: Decrements the thread counter. If it reaches zero, all waiting
        threads are released and the counter is reset. Otherwise, the current
        thread waits on the semaphore.
        """
        # Block Logic: Acquires a lock to safely decrement the shared thread counter.
        with self.count_lock:
            # Decrement the counter for the current phase.
            count_threads[0] -= 1
            # If the counter reaches zero, all threads for this phase have arrived.
            if count_threads[0] == 0:
                # Release all waiting threads from the semaphore.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for this phase, allowing reuse of the barrier.
                count_threads[0] = self.num_threads
        # Acquire the semaphore; this will block if not all threads have arrived yet,
        # or unblock immediately if they have all arrived and been released.
        threads_sem.acquire()




class Device(object):
    """
    @brief Represents a single device in a simulated distributed system.

    Each device has a unique ID, sensor data, a reference to a supervisor,
    and manages its own thread of execution to process scripts and
    synchronize with other devices.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary holding the device's sensor data.
        @param supervisor The supervisor object responsible for managing devices.
        """
        # device_id: Unique identifier for this device.
        self.device_id = device_id
        # sensor_data: Dictionary containing sensor readings or state relevant to specific locations.
        self.sensor_data = sensor_data
        # supervisor: Reference to the central supervisor managing all devices.
        self.supervisor = supervisor
        # script_received: An Event flag signaling when a new script has been assigned to the device.
        self.script_received = Event()
        # scripts: A list to store tuples of (script, location) to be executed by the device.
        self.scripts = []
        # timepoint_done: An Event flag signaling completion of operations for the current timepoint.
        self.timepoint_done = Event()
        # barrier: Reference to a ReusableBarrier instance used for global synchronization across devices.
        self.barrier = None
        # lock: A threading Lock to protect access to the device's internal state (e.g., sensor_data).
        self.lock = Lock()
        # locationlock: A list of locks, where each lock protects a specific location's data.
        self.locationlock = []
        # thread: The main execution thread for this device.
        self.thread = DeviceThread(self)
        # Start the device's main thread.
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (barrier, location locks) for devices.
        This method is typically called by the supervisor or a coordinating device (device_id == 0).
        @param devices A list of all Device objects in the system.
        """
        # Block Logic: Only device with ID 0 is responsible for initializing shared resources.
        if self.device_id == 0:
            # barrier: Creates a new ReusableBarrier instance for all devices.
            barrier = ReusableBarrier(len(devices))
            # locationlock: Initializes a list of locks, one for each possible location (assuming 100 locations).
            locationlock = []
            for _ in xrange(100):
                locationlock.append(Lock())
            # For each device, assign the newly created shared location locks and barrier.
            for device in devices:
                device.locationlock = locationlock
                device.set_barrier(barrier)
        else:
            # Other devices do not perform setup; they will receive their barrier and locks from device 0.
            pass

    def set_barrier(self, barrier):
        """
        @brief Sets the shared barrier for this device.
        @param barrier The ReusableBarrier instance to be used for synchronization.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location, or signals completion.
        @param script The script object to execute. If None, it signals that no more scripts are coming for this timepoint.
        @param location The location ID where the script should be executed.
        """
        # Block Logic: If a script is provided, add it to the list of scripts to be executed.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If no script is provided (script is None), it signifies the end of script assignments for the current timepoint.
            # Set the script_received event to unblock any waiting threads that are expecting scripts.
            self.script_received.set()
            # Set the timepoint_done event, indicating that this device is done with script assignments for the current timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.
        @param location The ID of the location to retrieve data for.
        @return The sensor data for the given location, or None if the location is not found.
        """
        # Block Logic: Checks if the requested location exists in the sensor_data.
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data for a specified location.
        @param location The ID of the location whose data is to be updated.
        @param data The new data value to set for the location.
        """
        # Block Logic: Checks if the requested location exists in the sensor_data.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device by joining its main thread.
        """
        self.thread.join()




class DeviceThread(Thread):
    """
    @brief Represents the main execution thread for a Device.

    This thread continuously interacts with the supervisor, processes assigned
    scripts, and synchronizes with other devices using a shared barrier.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread.
        @param device The Device object that this thread will manage.
        """
        # Calls the constructor of the parent Thread class, setting the thread's name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        # Stores a reference to the Device object this thread is associated with.
        self.device = device

    def run(self):
        """
        @brief The main loop of the DeviceThread.

        Logic: Continuously fetches neighbor information from the supervisor,
        waits for the current timepoint's tasks to be ready, launches sub-threads
        to execute assigned scripts, waits for sub-threads to complete,
        and then synchronizes with other devices using the shared barrier.
        Invariant: The device processes data and synchronizes in distinct timepoints.
        """
        # Block Logic: Main loop for device operations, continues until a shutdown signal.
        while True:
            # neighbours: Fetches the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If no neighbours are returned (e.g., simulation ended), break the loop.
            if neighbours is None:
                break
            
            # Block Logic: Waits until the 'timepoint_done' event is set,
            # indicating that all scripts for the current timepoint have been assigned.
            # Pre-condition: Scripts for the current timepoint have been assigned via assign_script(None, ...).
            self.device.timepoint_done.wait()
            # subthreads: List to hold DeviceSubThread instances for parallel script execution.
            subthreads = []

            # Block Logic: Iterates through assigned scripts and launches a new sub-thread for each.
            # Invariant: Each script runs concurrently in its own sub-thread.
            for (script, location) in self.device.scripts:
                # Appends a new DeviceSubThread instance to the list.
                subthreads.append(
                    DeviceSubThread(self, neighbours, script, location))
                # Starts the newly created sub-thread.
                subthreads[len(subthreads) - 1].start()
            
            # Block Logic: Waits for all launched sub-threads to complete their execution.
            # Pre-condition: All sub-threads have been started.
            for subthread in subthreads:
                subthread.join()
            
            # Functional Utility: Clears the 'timepoint_done' event to reset it for the next timepoint.
            self.device.timepoint_done.clear()
            
            # Block Logic: Synchronizes with all other devices using the shared barrier.
            # This ensures that all devices complete their timepoint's tasks before proceeding to the next.
            self.device.barrier.wait()


class DeviceSubThread(Thread):
    """
    @brief Executes a specific script for a Device at a given location.

    This thread collects data from neighboring devices and its own device
    at a specific location, runs a provided script on that data, and then
    updates the data on its neighbors and itself. It ensures exclusive access
    to the location data using a lock.
    """
    
    def __init__(self, devicethread, neighbours, script, location):
        """
        @brief Initializes a new DeviceSubThread.
        @param devicethread The parent DeviceThread that spawned this sub-thread.
        @param neighbours A list of neighboring Device objects.
        @param script The script object to be executed.
        @param location The specific location ID to which the script pertains.
        """
        # Calls the constructor of the parent Thread class, setting the thread's name.
        Thread.__init__(self, name="Device SubThread %d"
            % devicethread.device.device_id)
        # Stores a reference to the list of neighboring devices.
        self.neighbours = neighbours
        # Stores a reference to the parent DeviceThread.
        self.devicethread = devicethread

        # Stores the script object to be run by this thread.
        self.script = script
        # Stores the location ID for which this script is relevant.
        self.location = location

    def run(self):
        """
        @brief The main execution logic for the DeviceSubThread.

        Logic: Acquires a lock for the specific location to prevent race conditions.
        It collects relevant sensor data from its own device and neighbors for that location,
        executes the assigned script with this data, and then propagates the result
        back to its own device and neighbors, finally releasing the location lock.
        Invariant: Data for a specific location is processed and updated atomically.
        """
        # Block Logic: Acquires the lock for the specific location to ensure exclusive access
        # during data collection and update, preventing race conditions.
        self.devicethread.device.locationlock[self.location].acquire()
        # script_data: List to aggregate sensor data relevant to the script.
        script_data = []
        
        # Block Logic: Collects sensor data from all neighboring devices for the current location.
        # Pre-condition: Neighbours list is populated with valid Device objects.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Collects sensor data from the current device itself for the specified location.
        data = self.devicethread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: If any data was collected, execute the script and propagate results.
        # Pre-condition: script_data contains relevant sensor readings.
        if script_data != []:
            # result: Executes the script with the collected data.
            result = self.script.run(script_data)
            # Block Logic: Updates sensor data on all neighboring devices with the script's result.
            for device in self.neighbours:
                # Uses the device's internal lock to safely update its sensor data.
                with device.lock:
                    device.set_data(self.location, result)
            
            # Block Logic: Updates sensor data on the current device with the script's result.
            # Uses the device's internal lock to safely update its own sensor data.
            with self.devicethread.device.lock:
                self.devicethread.device.set_data(self.location, result)
        
        # Functional Utility: Releases the lock for the current location, allowing other threads to access it.
        self.devicethread.device.locationlock[self.location].release()
