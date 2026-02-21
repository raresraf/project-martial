


from threading import Event, Thread, Lock, Semaphore

class Barrier():
    """
    @brief Implements a reusable two-phase barrier for thread synchronization.

    This barrier ensures that all participating threads complete two distinct
    phases of execution before any thread proceeds to the next iteration.
    It uses semaphores and a counter to manage synchronization.
    """

    def __init__(self):
        """
        @brief Initializes the Barrier instance.
        """
        self.threads_num = 0       # Total number of threads to synchronize
        self.count1_threads = 0    # Counter for the first phase
        self.count2_threads = 0    # Counter for the second phase
        self.counter_lock = Lock() # Lock to protect access to counters
        self.semafor1 = Semaphore(0) # Semaphore for the first phase
        self.semafor2 = Semaphore(0) # Semaphore for the second phase

    def init_devices (self, dev_nr):
        """
        @brief Initializes the barrier with the total number of devices/threads.

        @param dev_nr: The total number of devices (threads) that will participate
                       in the synchronization.
        """
        self.threads_num = dev_nr
        self.count1_threads = dev_nr
        self.count2_threads = dev_nr

    def fazaI (self):
        """
        @brief First phase of the two-phase barrier synchronization.

        Threads decrement a counter. The last thread to reach zero
        releases all other threads waiting on `semafor1`.
        """
        with self.counter_lock:
            self.count1_threads -= 1
            # Block Logic: If this is the last thread to reach the barrier in phase I,
            # release all threads waiting on semafor1 and reset the counter.
            if self.count1_threads == 0:
                for i in range (self.threads_num):
                    self.semafor1.release()
                self.count1_threads = self.threads_num # Reset for reuse
        self.semafor1.acquire() # Wait for all threads to reach this point

    def fazaII (self):
        """
        @brief Second phase of the two-phase barrier synchronization.

        Threads decrement a counter. The last thread to reach zero
        releases all other threads waiting on `semafor2`.
        """
        with self.counter_lock:
            self.count2_threads -= 1
            # Block Logic: If this is the last thread to reach the barrier in phase II,
            # release all threads waiting on semafor2 and reset the counter.
            if self.count2_threads == 0:
                for i in range (self.threads_num):
                    self.semafor2.release()
                self.count2_threads = self.threads_num # Reset for reuse
        self.semafor2.acquire() # Wait for all threads to reach this point

    def wait(self):
        """
        @brief Waits for all threads to complete both phases of the barrier.

        This method orchestrates the two phases of synchronization.
        """
        self.fazaI()
        self.fazaII()

class Device(object):
    """
    @brief Represents a simulated device in a distributed sensing network.

    Manages local sensor data, assigned scripts, and coordinates its operation
    within the simulated environment using a dedicated thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing initial sensor readings
                            for various locations.
        @param supervisor: A reference to the central supervisor managing
                           the distributed system.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when scripts are assigned for a timepoint
        self.scripts = [] # List to store assigned scripts
        self.timepoint_done = Event() # Event to signal when timepoint processing is ready to proceed
        self.thread = DeviceThread(self) # Dedicated thread for this device
        self.thread.start() # Starts the device's main thread
        # Note: 'locks' array and 'devices' and 'barrier' attributes are managed by DeviceThread.bariera
        self.locks = [None] * 100 # Array of locks for specific sensor data locations
        self.devices = None # Reference to the list of all devices in the simulation
        self.barrier = None # Global barrier for synchronizing all devices

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device synchronization for all devices.

        Initializes the shared `Barrier` instance with the total number of devices.

        @param devices: A list of all Device instances in the simulation.
        """
        # Functional Utility: Initializes the global two-phase barrier with the total number of devices.
        DeviceThread.bariera.init_devices(len(devices))

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script is provided, it's added to the device's script list.
        If no script is provided (None), it signals that the current timepoint's
        script assignments are complete.

        @param script: The script object to execute, or None to signal timepoint completion.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """
        # Block Logic: Handles script assignment and signals script availability.
        if script is not None:
            # Block Logic: Initializes a location-specific lock if one does not exist.
            # This lock is shared globally and ensures data consistency.
            if self.locks[location] is None:
                self.locks[location] = Lock()
                # Functional Utility: Ensures all devices share the same lock instance for this location.
                for i in self.devices: # This reference to self.devices might be problematic if not set in setup_devices.
                    i.locks[location] = self.locks[location]

            self.scripts.append((script, location))
            self.script_received.set() # Signals the device thread that scripts have been received
        else:
            self.timepoint_done.set() # Signals the device thread that script assignment for timepoint is done

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The identifier for the data location.
        @return The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location: The identifier for the data location.
        @param data: The new data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its main thread to complete.

        Functional Utility: Ensures proper termination and cleanup of resources
        associated with the device's dedicated execution thread.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief A dedicated thread for each Device, managing its simulation logic.

    This thread is responsible for discovering neighbors, executing assigned
    scripts, and synchronizing with other devices at each timepoint using
    a shared two-phase barrier.
    """

    # Functional Utility: A static shared barrier instance for all DeviceThread instances.
    bariera = Barrier()

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: A reference to the parent Device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Continuously performs neighborhood discovery, waits for timepoint
        completion signals, processes all assigned scripts for the timepoint,
        and then synchronizes with other devices globally using a two-phase barrier.
        """
        while True:
            # Block Logic: Discovers neighboring devices for the current timepoint.
            # Pre-condition: `self.device.supervisor` is available to provide neighborhood information.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Checks for a shutdown condition (None neighbors indicates termination).
            if neighbours is None:
                break

            # Block Logic: Waits for the supervisor to signal that script assignments for
            # the current timepoint are complete.
            # Invariant: All scripts for the current timepoint are in `self.device.scripts` after this wait.
            self.device.timepoint_done.wait()

            # Block Logic: Processes all assigned scripts for the current timepoint.
            # Each script is executed on relevant sensor data.
            for (script, location) in self.device.scripts:
                script_data = []

                # Block Logic: Gathers relevant sensor data from neighboring devices for script input.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                # Block Logic: Includes the device's own sensor data in the script input.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if there is any data to process.
                if script_data != []:
                    # Functional Utility: Executes the assigned script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Propagates the script's result back to neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    # Block Logic: Updates the device's own sensor data with the script's result.
                    self.device.set_data(location, result)

            # Block Logic: Clears the timepoint_done event, resetting it for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been processed.
            self.device.timepoint_done.clear()

            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            DeviceThread.bariera.wait()