

"""
This module implements a device simulation framework, featuring:
- ReusableBarrierSem: A re-usable barrier synchronization mechanism for threads.
- Device: Represents a simulated device with sensors, scripts, and multi-threaded processing capabilities.
- DeviceThread: Individual threads managing script execution and data processing for a Device.
"""


from threading import Lock, Event, Thread, Semaphore, Condition


class ReusableBarrierSem():
    """
    Implements a reusable barrier synchronization mechanism using semaphores and a lock.
    This barrier allows a fixed number of threads to wait until all have reached a certain point
    before any are allowed to proceed, and can then be reset for subsequent synchronizations.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads


        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have reached this barrier.
        Uses a two-phase approach to allow reusability.
        """
        self.phase1()
        self.phase2()
    def phase1(self):
        """
        First phase of the barrier. Threads decrement a counter and the last thread
        releases all waiting threads for this phase.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            # Block Logic: If this is the last thread in phase 1, release all waiting threads.
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset counter for next use
        self.threads_sem1.acquire() # Wait for all threads to reach this point
    def phase2(self):
        """
        Second phase of the barrier. Threads decrement a counter and the last thread
        releases all waiting threads for this phase.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            # Block Logic: If this is the last thread in phase 2, release all waiting threads.
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset counter for next use
        self.threads_sem2.acquire() # Wait for all threads to reach this point

class Device(object):
    """
    Represents a simulated device in a distributed system. Each device
    manages its sensor data, executes scripts, and interacts with a supervisor
    and other devices. It uses multiple DeviceThread instances for concurrent operations.
    """
    
    location_locks = [] # Class-level list to store locks for different locations
    barrier = None      # Class-level barrier for synchronizing DeviceThreads across all devices
    nr_t = 8            # Number of threads per device

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for various locations.
            supervisor (Supervisor): The supervisor object responsible for managing devices
                                     and providing global information (e.g., neighbours).
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when a new script is assigned
        self.scripts = []             # List to store assigned scripts (script, location) tuples
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's processing
        self.neighbours_event = Event() # Event to signal when neighbour information is available
        self.threads = []             # List to hold DeviceThread instances
        # Block Logic: Create and start DeviceThread instances for this device.
        # Invariant: Each device will have `Device.nr_t` threads.
        for i in xrange(Device.nr_t):
            self.threads.append(DeviceThread(self, i))
        for i in xrange(Device.nr_t):
            self.threads[i].start()
    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the class-level barrier for synchronizing DeviceThreads across all devices.
        This method should be called once to initialize the barrier.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        Device.barrier = ReusableBarrierSem(Device.nr_t * len(devices))

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific location.
        If a script is provided, it's added to the device's script list and
        `script_received` event is set. If no script is provided (None),
        it signals that a timepoint is done.

        Args:
            script (Script or None): The script object to assign, or None to signal timepoint completion.
            location (str): The location associated with the script or data point.
        """
        # Block Logic: Check if a lock for the given location already exists.
        # Invariant: Each unique location should have one associated lock.
        if location not in [elem[0] for elem in Device.location_locks]:
            Device.location_locks.append((location, Lock()))

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a new script is available
        else:
            self.timepoint_done.set() # Signal that processing for the current timepoint is complete

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            Any or None: The sensor data if available for the location, otherwise None.
        """
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        Args:
            location (str): The location for which to set data.
            data (Any): The new sensor data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining all its worker threads.
        """
        
        for i in xrange(Device.nr_t):
            self.threads[i].join()

class DeviceThread(Thread):
    """
    A worker thread for a Device instance. Each thread is responsible for
    processing a subset of assigned scripts and managing data for its device.
    """

    def __init__(self, device, index):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent Device instance this thread belongs to.
            index (int): The unique index of this thread within its parent device.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.index = index
        self.neighbours = None

    def run(self):
        """
        Executes the main logic of the DeviceThread.
        - Synchronizes with other threads to get neighbour information.
        - Waits for timepoint processing to be signaled.
        - Processes a subset of assigned scripts, acquiring and releasing location-specific locks.
        - Updates sensor data based on script results.
        - Synchronizes with a global barrier after processing.
        """
        
        
        # Block Logic: Main loop for continuous script processing and synchronization across timepoints.
        while True:
            # Block Logic: The first thread (index == 0) is responsible for fetching neighbour information.
            # Invariant: Neighbour information is fetched once per timepoint and shared among threads.
            if self.index == 0:
                self.neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours_event.set() # Signal that neighbours are ready
            else:
                self.device.neighbours_event.wait() # Wait for the first thread to get neighbours
                self.neighbours = self.device.threads[0].neighbours # Retrieve neighbours from the first thread
            if self.neighbours is None:
                break # Exit if no neighbours, indicating shutdown or end of simulation

            # Block Logic: Wait for the main device to signal the start of a new timepoint's processing.
            self.device.timepoint_done.wait()

            # Block Logic: Process assigned scripts in a round-robin fashion among DeviceThreads.
            # Invariant: Each script is processed by exactly one thread.
            for j in range(self.index, len(self.device.scripts), Device.nr_t):
                location = self.device.scripts[j][1]
                script = self.device.scripts[j][0]

                # Block Logic: Acquire a lock for the specific location to prevent race conditions during data access.
                for i in range(len(Device.location_locks)):
                    if location == Device.location_locks[i][0]:
                        Device.location_locks[i][1].acquire() # Acquire lock

                script_data = []
                # Block Logic: Collect data from neighboring devices for the current location.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collect data from the current device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: If there is data, execute the script and update data.
                if script_data != []:
                    # Inline: Execute the assigned script with collected data.
                    result = script.run(script_data)

                    # Block Logic: Propagate the script's result to neighboring devices.
                    for device in self.neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Update the current device's sensor data with the script's result.
                    self.device.set_data(location, result)

                # Block Logic: Release the lock for the current location.
                for i in range(len(Device.location_locks)):
                    if location == Device.location_locks[i][0]:
                        Device.location_locks[i][1].release() # Release lock

            # Block Logic: Synchronize all DeviceThreads using the global barrier.
            # Pre-condition: All scripts for this thread's subset have been processed for the current timepoint.
            Device.barrier.wait()
            # Block Logic: Only the first thread clears the timepoint_done event for the next cycle.
            if self.index == 0:
                self.device.timepoint_done.clear()

            # Block Logic: Only the first thread clears the neighbours_event for the next cycle.
            if self.index == 0:
                self.device.neighbours_event.clear()
            # Block Logic: Synchronize all DeviceThreads again before starting the next timepoint.
            Device.barrier.wait()

