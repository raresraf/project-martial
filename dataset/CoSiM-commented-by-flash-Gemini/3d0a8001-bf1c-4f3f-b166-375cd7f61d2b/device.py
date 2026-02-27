


"""
@file device.py
@brief This module defines a complex device simulation architecture utilizing a Singleton pattern for shared resource management.

@details It implements device behavior, concurrent script execution, and synchronization.
         The `ReusableBarrier` provides synchronization across all devices. The `Singleton`
         class ensures a single instance of shared barrier and data locks.
         `Device` represents an individual simulated entity, and `DeviceThread` manages
         its main operational loop, dynamically spawning threads for concurrent
         script processing.
"""

from threading import Thread, Event, Semaphore, Lock

class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.

    @details This barrier uses a two-phase mechanism to ensure all participating
             threads wait until every thread has reached a common synchronization point,
             and then releases them simultaneously. It is designed to be reusable
             for subsequent synchronization points without needing re-initialization.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrier instance.

        @param num_threads The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # Two counters are used to manage the two phases of the barrier, allowing reusability.
        # Stored in a list to allow modification within nested scopes (e.g., `with self.count_lock`).
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock() # A lock to protect access to the thread counters.
        # Two semaphores, one for each phase, to block and release threads.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all threads have reached this barrier.

        @details This method orchestrates the two-phase synchronization, ensuring
                 all `num_threads` complete `phase1` and `phase2` before any
                 thread proceeds past the `wait` call.
        """
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single phase of the barrier synchronization.

        @details Decrements a shared counter. When the counter reaches zero, it
                 releases all waiting threads using a semaphore. It then resets
                 the counter for the next phase.
        @param count_threads A list containing the counter for the current phase (mutable).
        @param threads_sem The semaphore associated with the current phase.
        """
        
        with self.count_lock: # Ensure atomic access to the counter.
            count_threads[0] -= 1
            if count_threads[0] == 0: # If this is the last thread to reach the barrier.
                for _ in range(self.num_threads): # Release all waiting threads.
                    threads_sem.release()
                count_threads[0] = self.num_threads # Reset counter for reusability.
        threads_sem.acquire() # Block until released by the last thread.


class Singleton(object):
    """
    @brief Implements the Singleton design pattern for managing shared resources.

    @details This class ensures that only one instance of `Singleton.RealSingleton`
             is created throughout the application's lifecycle. It provides a global
             access point to shared resources such as the `ReusableBarrier` and
             a dictionary of `Lock` objects for data locations.
    """

    class RealSingleton(object):
        """
        @brief The actual singleton class holding shared resources.

        @details This inner class holds the global `barrier` and `locks` dictionary.
                 Its `initialize` method is called only once to set up these resources.
        """

        barrier = None # Global reusable barrier for device synchronization.
        locks = None   # Dictionary to hold locks for different data locations.

        def initialize(self, devices):
            """
            @brief Initializes the shared resources of the singleton.

            @details Sets up the global `ReusableBarrier` with the number of devices
                     and initializes an empty dictionary for location-specific locks.
            @param devices The number of devices that will use the shared barrier.
            """

            self.barrier = ReusableBarrier(devices) # Initialize the global barrier.
            self.locks = {} # Initialize the dictionary for location locks.

        def get_lock(self, location):
            """
            @brief Retrieves or creates a lock for a specific data location.

            @details Ensures that a unique `threading.Lock` object exists for each
                     data `location`, creating it if it doesn't already exist.
            @param location The identifier for the data location.
            @return A `threading.Lock` object associated with the given location.
            """
            
            # Block Logic: Create a new lock for the location if it doesn't exist.
            if location not in self.locks:
                self.locks[location] = Lock()

            return self.locks[location]

    
    __instance = None # Private class variable to hold the single instance.

    def __init__(self, numberOfDevices):
        """
        @brief Initializes the Singleton instance.

        @details This constructor ensures that `RealSingleton` is initialized only once.
                 Subsequent attempts to create a Singleton will return the existing instance.
        @param numberOfDevices The number of devices to initialize the barrier with (used only on first instantiation).
        """
        
        # Precondition: Check if the singleton instance has already been created.
        if Singleton.__instance is None:
            Singleton.__instance = Singleton.RealSingleton() # Create the RealSingleton instance.
            Singleton.__instance.initialize(numberOfDevices) # Initialize its shared resources.

    def __getattr__(self, attr):
        """
        @brief Delegates attribute access to the `RealSingleton` instance.

        @param attr The name of the attribute being accessed.
        @return The attribute value from the `RealSingleton` instance.
        """
        
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """
        @brief Delegates attribute setting to the `RealSingleton` instance.

        @param attr The name of the attribute being set.
        @param value The value to set the attribute to.
        """
        
        return setattr(self.__instance, attr, value)

    def get_instance(self):
        """
        @brief Returns the single instance of `Singleton.RealSingleton`.

        @return The `Singleton.RealSingleton` instance.
        """
        
        return self.__instance

    def get_lock(self, location):
        """
        @brief Convenience method to retrieve a lock for a given location from the singleton instance.

        @param location The identifier for the data location.
        @return A `threading.Lock` object associated with the given location.
        """
        
        return self.__instance.get_lock(location)

class Device(object):
    """
    @brief Represents a single simulated device in the environment.

    @details Each device manages its own sensor data, interacts with a central supervisor,
             and executes scripts. It utilizes a `Singleton` instance for accessing
             global shared resources like the synchronization barrier and location-specific locks.
             The device's operations are handled by a dedicated `DeviceThread`.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id An integer representing the unique identifier for the device.
        @param sensor_data A dictionary containing initial sensor data for the device.
        @param supervisor An object responsible for overseeing and coordinating devices.
        """

        self.singleton = None # Reference to the global Singleton instance for shared resources.
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to store (script, location) tuples assigned to this device.
        self.timepoint_done = Event() # Event to signal when all scripts for a timepoint are assigned.
        self.thread = DeviceThread(self) # The dedicated thread for this device's main loop.
        self.thread.start() # Start the main device thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Initializes the device's access to the global shared `Singleton` instance.

        @details This method is called during the setup phase to ensure each device
                 has a reference to the global shared barrier and locks. It triggers
                 the initialization of the `Singleton` if it hasn't been initialized yet.
        @param devices A list of all Device objects in the simulation (used to get total count for Singleton initialization).
        """

        # Block Logic: Initialize the global Singleton for shared resources if it's the first time,
        # or get the existing instance for subsequent devices.
        self.singleton = Singleton(len(devices))

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific data location.

        @details If a script is provided, it's added to the device's script queue.
                 If `script` is None, it signals that all scripts for the current
                 timepoint have been assigned to this device.
        @param script The script object to assign, or None.
        @param location The data location (e.g., sensor ID) where the script will operate.
        """
        
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its target location.
        else:
            self.timepoint_done.set() # Signal that all script assignments for this timepoint are done.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location from the device's internal state.

        @param location The data location (e.g., sensor ID).
        @return The sensor data for the location, or None if the location is not found.
        """
        
        return (self.sensor_data[location] if location in self.sensor_data
                else None)

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location in the device's internal state.

        @param location The data location (e.g., sensor ID).
        @param data The new sensor data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's dedicated thread.

        @details Joins the `DeviceThread`, ensuring all ongoing operations are completed
                 before the device fully shuts down.
        """
        
        self.thread.join() # Wait for the main device thread to finish its execution.



class DeviceThread(Thread):
    """
    @brief Manages the main operational loop for a Device object, coordinating script execution.

    @details This thread is responsible for the overall lifecycle of a device's operations
             within a timepoint. It fetches neighbor information, waits for scripts to be assigned,
             and then, crucially, spawns separate threads (`run_script`) to execute each
             script in parallel. It uses the shared `Singleton`'s barrier for global synchronization.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device The `Device` object that this thread is responsible for.
        """

        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device # Reference to the parent Device object.


    def run_script(self, location, neighbours, script):
        """
        @brief Executes a single script for a specific data location.

        @details This method is designed to be run in a separate thread for parallel execution.
                 It acquires a location-specific lock, gathers data from neighbors and the
                 current device, runs the script, and then updates the sensor data
                 on both the neighbors and the current device.
        @param location The data location (e.g., sensor ID) where the script operates.
        @param neighbours A list of neighboring Device objects to gather data from and update.
        @param script The script object to execute.
        """
        
        script_data = [] # List to accumulate data for the script.
        # Block Logic: Acquire the lock for the specific data location to ensure exclusive access.
        with self.device.singleton.get_lock(location):
            # Block Logic: Gather data from neighboring devices for the current location.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Get data from the current device for the current location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)


            # Precondition: Execute the script only if there is data available.
            if script_data != []:
                # Execute the assigned script with the collected data.
                result = script.run(script_data)

                # Block Logic: Update sensor data on neighboring devices and the current device.
                for device in neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)


    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        @details This loop continuously processes timepoints in the simulation.
                 It fetches neighbor information from the supervisor, waits for
                 scripts to be assigned, and then creates and manages auxiliary
                 threads (`run_script`) for parallel script execution. After all
                 scripts are processed, it clears the `timepoint_done` event
                 and synchronizes globally using the `Singleton`'s barrier.
                 The loop terminates when the supervisor signals the end of the simulation.
        """
        while True:
            # Block Logic: Query the supervisor for updated neighbor information for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            # Precondition: If supervisor returns None, it signals the end of the simulation.
            if neighbours is None:
                break # Exit the main loop.

            # Block Logic: Wait for all scripts for the current timepoint to be assigned by the Device.
            self.device.timepoint_done.wait()

            # Block Logic: Create and start auxiliary threads for each script to run in parallel.
            threads = [Thread(target=self.run_script, args=(
                l, neighbours, s)) for (s, l) in self.device.scripts]

            # Start all auxiliary script execution threads.
            for thread in threads:
                thread.start()
            # Wait for all auxiliary script execution threads to complete.
            for thread in threads:
                thread.join()

            # Block Logic: Reset the timepoint completion event and synchronize globally.
            self.device.timepoint_done.clear() # Reset the event for the next timepoint.
            self.device.singleton.barrier.wait() # Global synchronization barrier.
