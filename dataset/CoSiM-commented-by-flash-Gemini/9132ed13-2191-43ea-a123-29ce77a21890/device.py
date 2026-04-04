


"""
@9132ed13-2191-43ea-a123-29ce77a21890/device.py
@brief Implements a multi-threaded simulation for distributed sensor devices with dynamic script execution.

This module defines the core components for simulating a network of sensor devices,
each capable of executing scripts, managing local sensor data, and interacting
with a central supervisor. This version utilizes dynamically spawned `Thread` objects
to execute scripts via a `run_scripts` method within `DeviceThread`, and a
`ReusableBarrierSem` for timepoint synchronization. Location-specific locks are
managed dynamically.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate with neighbors under the
guidance of a supervisor.

Classes:
- Device: Represents a single simulated sensor device.
- DeviceThread: Manages the lifecycle and operation of a Device, including spawning
                and managing temporary threads for script execution.
- ReusableBarrierSem: A custom barrier implementation for thread synchronization.

Domain: Distributed Systems Simulation, Concurrent Programming, Dynamic Threading, Sensor Networks.
"""

from threading import Lock, Thread, Event, Semaphore


class Device(object):
    """
    @brief Represents a single simulated sensor device in a distributed network.

    Each device manages its own sensor data, interacts with a supervisor,
    and executes assigned scripts in a multi-threaded environment. This version
    uses a dedicated `DeviceThread` which, in turn, spawns new `Thread` instances
    for each script to be executed. Synchronization across devices is handled by a
    `ReusableBarrierSem`.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        Sets up the device's unique identifier, its initial sensor data,
        a reference to the central supervisor, and initializes various
        synchronization primitives and state variables required for
        multi-threaded operation.

        @param device_id: A unique integer identifier for the device.
        @param sensor_data: A dictionary containing the device's initial sensor readings.
                            Keys are location IDs, values are sensor data.
        @param supervisor: A reference to the Supervisor object managing the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Synchronization primitive: Event to signal that a new script has been assigned to the device.
        self.script_received = Event()
        self.scripts = [] # Stores scripts assigned to this device for execution. Each script is (script_object, location_id).
        # Synchronization primitive: Event to signal that all scripts for the current timepoint have been assigned.
        self.timepoint_done = Event()
        # Spawns a dedicated thread to manage this device's operations.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Shared resource: A barrier to synchronize all devices at specific points in the simulation.
        self.barrier = None
        # Shared resource: A dictionary of locks, one for each location, to ensure thread-safe
        # access to specific sensor data locations. Managed dynamically.
        self.location_locks = None
    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the device's awareness of all other devices in the simulation
               and initializes shared synchronization primitives.

        This method is called once at the beginning of the simulation.
        If this is the master device (device_id == 0), it initializes a shared
        `ReusableBarrierSem` and a `location_locks` dictionary. Otherwise, it
        inherits these from the master device.

        @param devices: A list of all Device objects participating in the simulation.
        """
        Device.devices_no = len(devices) # Store the total number of devices (class-level attribute).
        # Block Logic: The master device (device_id == 0) initializes shared resources,
        # while other devices inherit them from the master.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices)) # Initialize shared barrier for all devices.
            self.location_locks = {} # Initialize shared dictionary for location-specific locks.
        else:
            # Block Logic: Non-master devices inherit shared resources from the master device.
            self.barrier = devices[0].barrier
            self.location_locks = devices[0].location_locks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device to be executed at a specific location
               or signals the completion of script assignments if no script is provided.

        @param script: The script object to be executed, or None if the timepoint is done.
        @param location: The location ID associated with the script, or irrelevant if script is None.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signals that a new script has been assigned.
        else:
            # Block Logic: If no script is provided (script is None), it signifies that
            # all scripts for the current timepoint have been assigned.
            self.timepoint_done.set() # Signals that the timepoint processing is logically done (no more scripts to assign).

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The location ID for which to retrieve data.
        @return The sensor data for the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.

        This method directly updates the `sensor_data` dictionary for the given location.

        @param location: The location ID for which to set data.
        @param data: The new sensor data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown process for the device by joining its associated thread.

        This ensures that the `DeviceThread` completes its execution before
        the program exits.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the lifecycle and operation of a Device, including coordinating script execution.

    This thread is responsible for handling timepoint progression, retrieving neighbor
    information from the supervisor, and orchestrating the parallel execution of scripts
    by creating temporary `Thread` instances for each script. It uses a shared
    `ReusableBarrierSem` for global synchronization.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: The Device object that this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_scripts(self, script, location, neighbours):
        """
        @brief Executes a single script, gathering data and updating device states.

        This method is designed to be run in a separate `Thread` for each script.
        It acquires a location-specific lock, collects sensor data from the device
        and its neighbors, executes the script, and updates the sensor data
        in all relevant devices.

        @param script: The script object to be executed.
        @param location: The location ID associated with the script.
        @param neighbours: A list of neighboring Device objects from which to collect data.
        """
        # Block Logic: Dynamically get or create a lock for the specific location.
        # This ensures thread-safe access to sensor data at this location.
        lock_location = self.device.location_locks.get(location)
        if lock_location is None and location is not None:
            self.device.location_locks[location] = Lock()
            lock_location = self.device.location_locks[location]
        lock_location.acquire() # Pre-condition: Acquire lock for the specific location.
        
        script_data = [] # List to accumulate sensor data for the script.
        
        # Block Logic: Gathers sensor data from neighboring devices for the specified location.
        for device in neighbours:
            # Note: Access to neighbor's data is not protected by neighbor's lock here,
            # which could be a race condition if neighbor's `set_data` is called concurrently.
            # Assuming `get_data` is read-only or protected by `location_locks` implicitly.
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
            
        # Block Logic: Gathers sensor data from the current device for the specified location.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        
        # Pre-condition: If valid script data was collected.
        if script_data != []:
            # Functional Utility: Executes the assigned script with the collected sensor data.
            # The script's `run` method presumably contains the core logic for data processing.
            result = script.run(script_data)

            # Block Logic: Updates the sensor data for all neighboring devices with the script's result.
            for device in neighbours:
                # Note: Similar to `get_data`, `set_data` might need explicit locking if not atomic.
                device.set_data(location, result)
            
            # Block Logic: Updates the sensor data for the current device with the script's result.
            self.device.set_data(location, result)
        lock_location.release() # Post-condition: Releases the location-specific lock.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        It continuously retrieves neighbor information, waits for the timepoint
        to be marked as done (scripts assigned), spawns temporary threads for
        each script, waits for their completion, and then synchronizes with
        other devices via the shared barrier.
        """
        while True:
            # Block Logic: Retrieves information about neighboring devices from the supervisor.
            # This information is crucial for scripts that need to interact with or gather
            # data from adjacent devices in the simulated environment.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If there are no more neighbors (end of simulation), break the loop.
            if neighbours is None:
                break
            
            # Pre-condition: Waits until the `Device` signals that all scripts for the current
            # timepoint have been assigned.
            self.device.timepoint_done.wait()
            
            # Block Logic: Creates and starts temporary threads for each assigned script.
            # Each thread executes the `run_scripts` method to process a single script.
            tlist = [] # List to keep track of spawned threads.
            for (script, location) in self.device.scripts:
                thread = Thread(target=self.run_scripts, args=(script, location, neighbours))
                tlist.append(thread)
                thread.start()
            
            # Block Logic: Waits for all temporary script execution threads to complete.
            for thread in tlist:
                thread.join()
            
            # Post-condition: Clears the 'timepoint_done' event for the next cycle.
            self.device.timepoint_done.clear()
            
            # Synchronization point: All devices wait here until every other device
            # has completed its current timepoint processing. This ensures that
            # all devices are synchronized before moving to the next timepoint.
            self.device.barrier.wait()


class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier using semaphores for synchronizing a fixed number of threads.

    This barrier allows a specified number of threads to wait at a synchronization
    point and then proceed together. It is designed to be reusable for multiple
    synchronization cycles.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrierSem instance.

        @param num_threads: The total number of threads that must reach the barrier
                            before any of them can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for the first phase of the barrier.
        self.count_threads2 = self.num_threads # Counter for the second phase of the barrier.
        self.counter_lock = Lock()               # Lock to protect access to the counters.
        self.threads_sem1 = Semaphore(0)         # Semaphore for the first phase, initially blocking all threads.
        self.threads_sem2 = Semaphore(0)         # Semaphore for the second phase, initially blocking all threads.

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this point.

        This method orchestrates the two-phase synchronization, ensuring all threads
        are released only after all have arrived at the barrier.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief First phase of the barrier synchronization.

        Threads acquire the `counter_lock` to decrement `count_threads1`. The last thread
        to reach this phase (when `count_threads1` becomes 0) releases all semaphores
        for the first phase, allowing all waiting threads to proceed.
        """
        # Block Logic: Ensures atomic decrement of the counter for the first phase.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Pre-condition: If this is the last thread to reach the barrier in phase 1.
            if self.count_threads1 == 0:
                # Functional Utility: Releases all `num_threads` from the `threads_sem1` semaphore.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Resets the counter for the next use.

        # Functional Utility: Blocks until released by the last thread in phase 1.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Second phase of the barrier synchronization.

        Threads acquire the `counter_lock` to decrement `count_threads2`. The last thread
        to reach this phase (when `count_threads2` becomes 0) releases all semaphores
        for the second phase, allowing all waiting threads to proceed. This two-phase
        approach prevents issues where a fast thread might re-enter the barrier before
        all slow threads have left it from the previous cycle.
        """
        # Block Logic: Ensures atomic decrement of the counter for the second phase.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Pre-condition: If this is the last thread to reach the barrier in phase 2.
            if self.count_threads2 == 0:
                # Functional Utility: Releases all `num_threads` from the `threads_sem2` semaphore.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Resets the counter for the next use.

        # Functional Utility: Blocks until released by the last thread in phase 2.
        self.threads_sem2.acquire()
