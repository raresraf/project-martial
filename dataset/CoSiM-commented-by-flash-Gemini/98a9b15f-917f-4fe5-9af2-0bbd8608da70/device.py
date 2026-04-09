"""
@file device.py
@brief Implements a simulated distributed device system for sensor data processing
       and communication, utilizing thread synchronization primitives.
       This module defines the core components: ReusableBarrier for thread
       coordination, Device representing a node in the system, DeviceThread
       for supervisor interaction, and ThreadAux for script execution and data handling.

Architectural Intent:
- Simulate a network of interconnected devices (e.g., sensor nodes).
- Support concurrent data processing and inter-device communication.
- Utilize a barrier synchronization mechanism to coordinate execution across devices and their internal threads.
- Manage local sensor data and exchange data with neighboring devices.

Domain: Distributed Systems, Concurrency, Simulation.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier synchronization mechanism.
           Threads wait at the barrier until all `num_threads` have arrived.
           It uses a double-buffering approach with two semaphores to allow for reuse.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier.
        @param num_threads The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        # Functional Utility: Counter for threads arriving at the first phase of the barrier.
        self.count_threads1 = self.num_threads
        # Functional Utility: Counter for threads arriving at the second phase of the barrier.
        self.count_threads2 = self.num_threads

        # Functional Utility: Lock to protect access to the thread counters.
        self.counter_lock = Lock()
        # Functional Utility: Semaphore for synchronizing threads in the first phase.
        self.threads_sem1 = Semaphore(0)
        # Functional Utility: Semaphore for synchronizing threads in the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all threads have reached this point in both phases.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief First phase of the reusable barrier. Threads increment a counter and
               either release all waiting threads if they are the last to arrive,
               or acquire a semaphore to wait.
        """
        # Block Logic: Atomically increments the phase 1 thread counter.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Block Logic: If this is the last thread to arrive, release all waiting threads.
            # Invariant: All threads have reached phase 1.
            if self.count_threads1 == 0:
                # Functional Utility: Release all `num_threads` from waiting on `threads_sem1`.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Functional Utility: Reset the counter for the next use of phase 1.
                self.count_threads1 = self.num_threads

        # Functional Utility: Blocks the thread until released by the last thread in this phase.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Second phase of the reusable barrier. Similar to phase 1, but uses a different
               counter and semaphore to allow the barrier to be reused.
        """
        # Block Logic: Atomically increments the phase 2 thread counter.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Block Logic: If this is the last thread to arrive, release all waiting threads.
            # Invariant: All threads have reached phase 2.
            if self.count_threads2 == 0:
                # Functional Utility: Release all `num_threads` from waiting on `threads_sem2`.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Functional Utility: Reset the counter for the next use of phase 2.
                self.count_threads2 = self.num_threads

        # Functional Utility: Blocks the thread until released by the last thread in this phase.
        self.threads_sem2.acquire()


class Device(object):
    """
    @brief Represents a single device (node) in the simulated distributed system.
           Each device manages its own sensor data, communicates with a supervisor,
           and uses internal and global threads for concurrent operations.
    """
    # Functional Utility: Global barrier for synchronizing all Device instances. Initialized with 1,
    # and re-initialized in setup_devices.
    bar1 = ReusableBarrier(1)
    # Functional Utility: Global event to signal that all devices have been set up.
    event1 = Event()
    # Functional Utility: Global list of locks, potentially for protecting shared data locations across devices.
    locck = []

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id Unique identifier for this device.
        @param sensor_data Dictionary containing local sensor readings.
        @param supervisor Reference to the supervisor object for global coordination.
        """
        # Functional Utility: Event to signal completion of processing for a specific timepoint.
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Functional Utility: List to hold references to other devices in the system.
        self.devices = []

        # Functional Utility: List of Events, likely used for per-timepoint or per-thread signaling within the device.
        self.event = []
        for _ in xrange(11): # Invariant: A fixed number (11) of events are pre-allocated.
            self.event.append(Event())

        # Functional Utility: Number of auxiliary worker threads associated with this device.
        self.nr_threads_device = 8
        # Functional Utility: Counter used for round-robin assignment of scripts to auxiliary threads.
        self.nr_thread_atribuire = 0

        # Functional Utility: Reusable barrier for synchronizing this device's internal auxiliary threads.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device+1) # +1 includes the DeviceThread itself

        # Functional Utility: The main thread for this device, handling supervisor interactions.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Functional Utility: List to hold references to auxiliary worker threads.
        self.threads = []
        # Block Logic: Creates and starts `nr_threads_device` auxiliary worker threads.
        # Invariant: All auxiliary threads are initialized and running in the background.
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the device with references to all other devices in the system.
               Also performs global initialization for shared resources on device 0.
        @param devices A list of all Device instances in the system.
        """
        self.devices = devices
        # Block Logic: Performs global initialization steps only for the device with ID 0.
        # Invariant: Global shared resources (locks, barrier) are initialized once.
        if self.device_id == 0:
            # Functional Utility: Populates the global list of locks, protecting shared data locations.
            for _ in xrange(30): # Invariant: A fixed number (30) of locks are pre-allocated.
                Device.locck.append(Lock())
            # Functional Utility: Re-initializes the global barrier with the actual total number of devices.
            Device.bar1 = ReusableBarrier(len(devices))
            # Functional Utility: Signals that global device setup is complete, unblocking waiting threads.
            Device.event1.set()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to an auxiliary thread to process data at a specific location.
        @param script The script object to be executed.
        @param location The data location (e.g., sensor ID) the script will operate on.
        """
        # Block Logic: If a script is provided, assigns it to an auxiliary thread in a round-robin fashion.
        # Invariant: Scripts are distributed among available auxiliary threads.
        if script is not None:
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            # Functional Utility: Updates the round-robin counter for script assignment.
            self.nr_thread_atribuire = (self.nr_thread_atribuire+1)%
            self.nr_threads_device
        else:
            # Functional Utility: Signals that processing for the current timepoint is done for this device.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location The location (key) for which to retrieve data.
        @return The sensor data or None if the location is not found.
        """
        return self.sensor_data[location] if location in 
        self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.
        @param location The location (key) for which to set data.
        @param data The data to be stored.
        """
        if location in self.sensor_data: # Block Logic: Only updates if the location already exists in sensor_data.
            self.sensor_data[location] = data


    def shutdown(self):
        """
        @brief Shuts down the device by joining its main and auxiliary threads.
        """
        self.thread.join()
        for threadd in self.threads: # Block Logic: Waits for each auxiliary thread to complete its execution.
            threadd.join()


class DeviceThread(Thread):
    """
    @brief The main thread for a Device. Responsible for interacting with the supervisor,
           managing timepoints, and coordinating with the device's auxiliary threads.
    """
    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Functional Utility: Stores a reference to the neighboring devices received from the supervisor.
        self.neighbours = None
        # Functional Utility: Counter for accessing the device's list of events (`self.device.event`).
        self.contor = 0

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
               Handles waiting for global setup, fetching neighbors, managing timepoints,
               and synchronizing with other threads and devices.
        """
        # Functional Utility: Waits until all devices have completed their initial setup.
        Device.event1.wait()

        # Block Logic: Main loop for processing timepoints.
        while True:
            # Functional Utility: Fetches information about neighboring devices from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: If no neighbors are returned (signaling simulation end), break the loop.
            # Invariant: `neighbours` is None when the simulation is to terminate.
            if self.neighbours is None:
                # Functional Utility: Sets the current timepoint event, possibly to unblock `ThreadAux` threads.
                self.device.event[self.contor].set()
                break

            # Functional Utility: Waits until the current timepoint's processing is marked as done by the supervisor.
            self.device.timepoint_done.wait()
            # Functional Utility: Clears the event for the next timepoint.
            self.device.timepoint_done.clear()

            # Functional Utility: Signals completion for the current timepoint to its auxiliary threads.
            self.device.event[self.contor].set()
            # Functional Utility: Increments the event counter for the next timepoint.
            self.contor += 1

            # Functional Utility: Synchronizes with this device's auxiliary threads, ensuring they complete their work.
            self.device.bar_threads_device.wait()

            # Functional Utility: Global synchronization: waits for all devices to complete their timepoint processing.
            Device.bar1.wait()


class ThreadAux(Thread):
    """
    @brief An auxiliary worker thread within a Device. Responsible for executing assigned scripts
           to process data, interacting with sensor data, and communicating with neighboring devices.
    """
    def __init__(self, device):
        """
        @brief Initializes an ThreadAux instance.
        @param device The Device instance this thread belongs to.
        """
        Thread.__init__(self)
        self.device = device
        # Functional Utility: Dictionary to map script objects to their target data locations.
        self.script_loc = {}
        # Functional Utility: Counter for accessing the device's list of events (`self.device.event`).
        self.contor = 0

    def run(self):
        """
        @brief The main execution loop for the ThreadAux.
               Handles waiting for timepoint signals, executing scripts on local and neighbor data,
               and synchronizing with other threads.
        """
        # Block Logic: Main loop for processing assigned scripts based on timepoint events.
        while True:
            # Functional Utility: Waits for the DeviceThread to signal the start of a new timepoint's processing.
            self.device.event[self.contor].wait()
            # Functional Utility: Increments the event counter for the next timepoint.
            self.contor += 1

            # Functional Utility: Retrieves neighbor information from the DeviceThread.
            neigh = self.device.thread.neighbours
            # Block Logic: If no neighbors (signaling simulation end), break the loop.
            # Invariant: `neigh` is None when the simulation is to terminate.
            if neigh is None:
                break

            # Block Logic: Iterates through assigned scripts and executes them.
            # Invariant: Each script processes its designated data location.
            for script in self.script_loc:
                location = self.script_loc[script]
                # Functional Utility: Acquires a global lock to protect data at 'location' during processing.
                # This ensures data consistency when multiple threads/devices might access the same location.
                Device.locck[location].acquire()
                # Functional Utility: List to aggregate data relevant to the current script.
                script_data = []

                # Block Logic: Gathers data for the current location from all neighboring devices.
                # Invariant: `script_data` contains available data from neighbors for the specified `location`.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None: # Block Logic: Only adds data if it exists.
                        script_data.append(data)

                # Functional Utility: Gathers data for the current location from the local device.
                data = self.device.get_data(location)
                if data is not None: # Block Logic: Only adds data if it exists.
                    script_data.append(data)

                # Block Logic: Executes the script if any data was collected.
                # Pre-condition: `script_data` is not empty.
                if script_data != []:
                    # Functional Utility: Executes the assigned script with the collected data.
                    result = script.run(script_data)
                    # Block Logic: Updates the data at 'location' on all neighboring devices with the script's result.
                    for device in neigh:
                        device.set_data(location, result)
                    # Functional Utility: Updates the data at 'location' on the local device with the script's result.
                    self.device.set_data(location, result)

                # Functional Utility: Releases the global lock for 'location', allowing other threads/devices access.
                Device.locck[location].release()

            # Functional Utility: Synchronizes with the device's main thread and other auxiliary threads,
            # indicating completion of script execution for the current timepoint.
            self.device.bar_threads_device.wait()
