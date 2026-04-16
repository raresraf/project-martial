

"""
@ec588f2d-562d-4fd9-9187-e15673cef537/device.py
@brief Defines core components for simulating a distributed sensor network or device system.
This module provides classes for devices, their operational threads, and a reusable
barrier synchronization mechanism, enabling simulation of concurrent operations
and data exchange across multiple simulated entities.

Domain: Distributed Systems, Concurrency, Simulation.
"""

from threading import Lock, Thread, Event, Semaphore


class Device(object):
    """
    @brief Represents a single device in the simulated distributed system.
    Each device manages its own sensor data, processes scripts, and interacts
    with a supervisor to coordinate with other devices. It encapsulates
    device-specific state and behavior.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data readings
                            keyed by location.
        @param supervisor: A reference to the supervisor object that
                           manages inter-device communication and coordination.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to this device.
        self.script_received = Event()
        # List to store assigned scripts, each paired with its execution location.
        self_scripts = []
        # Event to signal completion of a timepoint's processing for this device.
        self.timepoint_done = Event()
        # The dedicated thread responsible for this device's operations.
        self.thread = DeviceThread(self)
        # Start the operational thread for this device.
        self.thread.start()
        # Barrier for synchronizing all devices at the end of a timepoint.
        self.barrier = None
        # Locks for controlling access to shared sensor data locations across devices.
        self.location_locks = None

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        @return A string in the format "Device {device_id}".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures shared resources (barrier and location locks) across devices.
        This method ensures that all devices share the same synchronization barrier
        and a common set of locks for data locations. The first device (device_id 0)
        is responsible for initializing these shared resources.
        @param devices: A list of all Device objects in the simulation.
        """
        # Block Logic: Establish the total number of participating devices.
        Device.devices_no = len(devices)
        # Block Logic: Initialize shared resources (barrier, location locks)
        # only by the first device (device_id 0) to avoid duplication.
        if self.device_id == 0:
            # Reusable barrier for synchronizing all threads at simulation timepoints.
            self.barrier = ReusableBarrierSem(len(devices))
            # Dictionary to hold Locks for each sensor data location, preventing
            # race conditions during data access.
            self.location_locks = {}
        else:
            # Other devices reference the shared barrier and locks initialized by device 0.
            self.barrier = devices[0].barrier
            self.location_locks = devices[0].location_locks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.
        If a script is provided, it's added to the device's queue and a signal
        is set to notify the device's thread. If no script is provided (None),
        it indicates that the current timepoint's assignments are complete.
        @param script: The script object to be executed, or None to signal completion.
        @param location: The data location pertinent to the script's execution.
        """
        # Block Logic: Processes script assignment or signals timepoint completion.
        if script is not None:
            self.scripts.append((script, location))
            # Signal that a script has been received, waking up the device's thread.
            self.script_received.set()
        else:
            # Signal that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.
        @param location: The location for which to retrieve data.
        @return The sensor data at the given location, or None if the location
                does not exist in the device's sensor_data.
        """
        # Inline: Safely retrieve data using dictionary's get method to handle missing locations.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data for a specified location.
        @param location: The location whose data needs to be updated.
        @param data: The new data value to set for the location.
        """
        # Block Logic: Updates sensor data only if the location already exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread.
        Ensures proper termination by waiting for the device's thread to complete its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the execution lifecycle of a single Device.
    This thread is responsible for fetching neighbor information,
    executing assigned scripts, and synchronizing with other device threads
    at various timepoints in the simulation.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread for a given device.
        @param device: The Device object that this thread will manage.
        """
        # Functional Utility: Initializes the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_scripts(self, script, location, neighbours):
        """
        @brief Executes a specific script at a given location, handling data access
               and updates for the device and its neighbors.
        This method ensures thread-safe access to sensor data locations by
        acquiring and releasing a per-location lock. It aggregates data
        from the device and its neighbors, executes the script, and then
        propagates the results.
        @param script: The script object to execute.
        @param location: The sensor data location pertinent to the script.
        @param neighbours: A list of neighboring Device objects.
        """
        # Block Logic: Retrieves or creates a lock for the specific data location
        # to ensure exclusive access during script execution.
        lock_location = self.device.location_locks.get(location)
        if lock_location is None and location is not None:
            self.device.location_locks[location] = Lock()
            lock_location = self.device.location_locks[location]
        # Acquire the lock to prevent other threads from modifying data at this location.
        lock_location.acquire()
        script_data = []
        
        # Block Logic: Collects sensor data from neighboring devices at the specified location.
        for device in neighbours:
            data = device.get_data(location)
            # Invariant: Only valid data (non-None) is appended to script_data.
            if data is not None:
                script_data.append(data)
            
        # Block Logic: Collects sensor data from the current device itself.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        
        # Block Logic: Executes the script if there is any collected data.
        if script_data != []:
            # Functional Utility: Executes the assigned script with the aggregated data.
            result = script.run(script_data)

            # Block Logic: Propagates the script's result to neighboring devices.
            for device in neighbours:
                device.set_data(location, result)
            
            # Block Logic: Updates the current device's sensor data with the script's result.
            self.device.set_data(location, result)
        # Release the lock, allowing other threads to access this data location.
        lock_location.release()

    def run(self):
        """
        @brief The main execution loop for the device thread.
        This loop continuously runs, coordinating with the supervisor to get
        neighbor information, executing assigned scripts in parallel, and
        synchronizing with other device threads using a barrier at each timepoint.
        """
        # Block Logic: The main simulation loop, continuing until no neighbors are found
        # (indicating simulation termination or a paused state).
        while True:
            # Invariant: 'neighbours' is either a list of Device objects or None.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            # Block Logic: Waits until the supervisor signals that all scripts for
            # the current timepoint have been assigned.
            self.device.timepoint_done.wait()
            tlist = []
            # Block Logic: Iterates through assigned scripts and launches a new thread
            # for each script to run it concurrently.
            for (script, location) in self.device.scripts:
                thread = Thread(target=self.run_scripts, args=(script, location, neighbours))
                tlist.append(thread)
                thread.start()
            # Block Logic: Waits for all script execution threads for the current timepoint to complete.
            for thread in tlist:
                thread.join()
            # Reset the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()
            # Synchronize all device threads using the barrier, ensuring all
            # have completed their current timepoint's work before proceeding.
            self.device.barrier.wait()


class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier synchronization mechanism using semaphores.
    This barrier allows a fixed number of threads to synchronize multiple times,
    ensuring that no thread proceeds past the barrier until all participating
    threads have reached it. It uses a two-phase approach to allow reusability.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that must reach the
                            barrier for it to be lifted.
        """
        self.num_threads = num_threads
        # Counter for the first phase of the barrier.
        self.count_threads1 = self.num_threads
        # Counter for the second phase of the barrier, enabling reusability.
        self.count_threads2 = self.num_threads
        # Lock to protect access to the thread counters.
        self.counter_lock = Lock()
        # Semaphore for the first synchronization phase. Initialized to 0
        # so threads wait until all have arrived.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for the second synchronization phase. Initialized to 0
        # for reusability, ensures threads wait for reset.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this point.
        Orchestrates the two-phase barrier synchronization.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief The first phase of the reusable barrier.
        Threads decrement a counter and the last thread to reach zero
        releases all waiting threads via a semaphore, then resets the counter.
        """
        # Block Logic: Critical section for safely decrementing the thread counter.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Invariant: If this is the last thread, release all waiting threads.
            if self.count_threads1 == 0:
                # Release all threads waiting on threads_sem1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset the counter for the next use of the barrier.
                self.count_threads1 = self.num_threads
        # Wait until all other threads have reached phase 1 and the semaphore is released.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief The second phase of the reusable barrier.
        Similar to phase 1, but uses a different semaphore to allow
        the barrier to be reused.
        """
        # Block Logic: Critical section for safely decrementing the thread counter for phase 2.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Invariant: If this is the last thread, release all waiting threads.
            if self.count_threads2 == 0:
                # Release all threads waiting on threads_sem2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset the counter for the next use of the barrier.
                self.count_threads2 = self.num_threads
        # Wait until all other threads have reached phase 2 and the semaphore is released.
        self.threads_sem2.acquire()
