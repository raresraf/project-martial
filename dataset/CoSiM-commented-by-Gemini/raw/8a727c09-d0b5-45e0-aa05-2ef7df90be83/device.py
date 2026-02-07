


"""
@file device.py
@brief Implements simulated distributed device functionality with synchronization primitives and multi-threaded execution.
This module defines `Device` for individual simulated entities, `ReusableBarrier` for thread coordination,
and `DeviceThread` along with `DeviceSubThread` for managing concurrent script execution and data processing
in a simulated environment.
"""


from threading import Event, Thread, Semaphore, Lock



class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier synchronization mechanism.
    This barrier ensures that a specified number of threads wait at a synchronization point
    until all threads have arrived, then allows them to proceed, and can be reused.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier.
        @param num_threads: The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Counter for the first phase, wrapped in list for mutability
        self.count_threads2 = [self.num_threads] # Counter for the second phase, wrapped in list for mutability
        self.count_lock = Lock() # Protects access to the counters
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase of the barrier
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase of the barrier

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have completed both phases of the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements a single phase of the reusable barrier logic.
        Threads decrement a shared counter. The last thread to decrement the counter
        resets it and releases all other threads waiting on the phase's semaphore.
        @param count_threads: The shared counter (list containing an integer) for this phase.
        @param threads_sem: The semaphore for this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # Invariant: If this is the last thread to enter the phase, it releases all other threads.
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release() # Release all waiting threads
                count_threads[0] = self.num_threads # Reset counter for next reuse
        threads_sem.acquire() # Acquire semaphore, blocking until released by the last thread




class Device(object):
    """
    @brief Represents a simulated device in a distributed system, handling sensor data,
    script execution, and synchronization with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data for various locations.
        @param supervisor: A reference to the supervisor entity managing the devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal that a script has been received for processing
        self.scripts = []  # List to store assigned scripts and their locations
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's tasks
        self.barrier = None  # ReusableBarrier for inter-device synchronization
        self.lock = Lock()  # Generic lock for device-level synchronization
        self.locationlock = []  # List of Locks for protecting sensor data access by location
        self.thread = DeviceThread(self)  # Main device thread for managing timepoints and neighbors
        self.thread.start()

    def __str__(self):
        """
        @brief Provides a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the device's shared resources and synchronization mechanisms.
        Only the master device (device_id 0) initializes the barrier and location locks,
        then distributes them to other devices.
        @param devices: A list of all Device instances in the system.
        Pre-condition: This method is expected to be called by all devices during initial setup.
        Post-condition: The device's `barrier` and `locationlock` are initialized
                        or linked to the master device's instances.
        """
        # Block Logic: Master device (device_id == 0) initializes shared synchronization primitives.
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            locationlock = []
            # Functional Utility: Initializes a list of 100 locks for protecting sensor data access per location.
            for _ in xrange(100):
                locationlock.append(Lock())
            # Functional Utility: Distributes the initialized barrier and location locks to all other devices.
            for device in devices:
                device.locationlock = locationlock
                device.set_barrier(barrier)
        else:
            # Invariant: Non-master devices implicitly receive barrier and location locks
            # through the master device's setup logic.
            pass

    def set_barrier(self, barrier):
        """
        @brief Sets the `ReusableBarrier` instance for this device.
        @param barrier: The `ReusableBarrier` instance to be used for synchronization.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location on the device.
        If a script is provided, it's added to the device's script list.
        If `script` is None, it signals that all scripts for the current timepoint have been assigned,
        and that the timepoint's processing can begin.
        @param script: The script object to assign, or `None` to signal timepoint completion.
        @param location: The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Block Logic: Signals that all scripts for the current timepoint have been assigned
            # and that the timepoint processing can begin.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The location for which to retrieve sensor data.
        @return The sensor data if `location` exists, otherwise `None`.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location if it exists.
        @param location: The location for which to set sensor data.
        @param data: The new sensor data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the main device thread.
        Pre-condition: Assumes `DeviceThread` has a mechanism to gracefully exit its run loop.
        """
        self.thread.join()





class DeviceThread(Thread):
    """
    @brief A worker thread that manages the device's main execution loop across timepoints.
    It orchestrates neighbor updates, spawns `DeviceSubThread`s for script execution,
    and synchronizes with other devices using a barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread.
        @param device: The `Device` instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        This loop continuously updates neighbor information, waits for timepoint completion,
        spawns sub-threads for script execution, and synchronizes using a barrier before
        starting the next timepoint.
        """
        while True:
            # Functional Utility: Fetches the latest neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Invariant: If `neighbours` is None, it signals that the device is shutting down.
            if neighbours is None:
                break
            
            # Block Logic: Waits for the current timepoint's script assignments to be completed.
            self.device.timepoint_done.wait()
            subthreads = []

            # Block Logic: Spawns `DeviceSubThread`s for each assigned script to execute concurrently.
            for (script, location) in self.device.scripts:
                subthreads.append(
                    DeviceSubThread(self, neighbours, script, location))
                subthreads[len(subthreads) - 1].start()
            
            # Block Logic: Joins all spawned sub-threads, waiting for their completion.
            for subthread in subthreads:
                subthread.join()
            
            self.device.timepoint_done.clear() # Reset the timepoint completion event
            
            # Block Logic: Synchronizes all devices at the barrier before proceeding to the next timepoint.
            self.device.barrier.wait()



class DeviceSubThread(Thread):
    """
    @brief A sub-worker thread responsible for executing a single script for a specific location.
    It collects sensor data from neighboring devices and its parent device, executes the script,
    and then updates the sensor data on all relevant devices.
    """
    
    def __init__(self, devicethread, neighbours, script, location):
        """
        @brief Initializes a new DeviceSubThread.
        @param devicethread: A reference to the parent `DeviceThread` instance.
        @param neighbours: A list of neighboring `Device` instances.
        @param script: The script object to be executed.
        @param location: The location associated with the script.
        """
        Thread.__init__(self, name="Device SubThread %d"
            % devicethread.device.device_id)
        self.neighbours = neighbours
        self.devicethread = devicethread


        self.script = script
        self.location = location

    def run(self):
        """
        @brief The main execution logic for the DeviceSubThread.
        It acquires a lock for the specific location, collects sensor data from neighbors
        and the parent device, executes the script, and then updates the data on
        neighbors and the parent device in a thread-safe manner.
        """
        # Block Logic: Acquire a lock for the specific location to ensure exclusive access to its sensor data.
        self.devicethread.device.locationlock[self.location].acquire()
        script_data = []
        
        # Block Logic: Collect data from neighboring devices for the current location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Collect data from the parent device for the current location.
        data = self.devicethread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: If data is available, execute the script and update relevant devices.
        if script_data != []:
            result = self.script.run(script_data) # Execute the script
            # Block Logic: Update sensor data on neighboring devices in a thread-safe manner.
            for device in self.neighbours:
                with device.lock: # Acquire lock for neighbor device's data
                    device.set_data(self.location, result)
            
            # Block Logic: Update sensor data on the parent device in a thread-safe manner.
            with self.devicethread.device.lock: # Acquire lock for parent device's data
                self.devicethread.device.set_data(self.location, result)
        
        # Block Logic: Release the location lock after data processing is complete.
        self.devicethread.device.locationlock[self.location].release()

