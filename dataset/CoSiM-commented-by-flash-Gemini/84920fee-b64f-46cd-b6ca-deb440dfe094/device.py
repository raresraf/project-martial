

"""
This module implements a simulated device within a distributed sensor network,
handling data management, script execution, and synchronization using a reusable barrier.
It supports parallel script execution using a semaphore-controlled thread pool and
ensures data consistency with location-specific locks.

Domain: Distributed Systems, Concurrency, Sensor Networks.
"""

from threading import Semaphore, Event, Lock, Thread

class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for thread synchronization using semaphores.

    This barrier allows a fixed number of threads to wait until all have reached a certain point
    before any are allowed to proceed. It uses a two-phase approach to ensure reusability
    without deadlocks.
    """
    

    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier with a specified number of threads.

        @param num_threads: The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # @brief Counter for the first phase of the barrier.
        self.count_threads1 = self.num_threads
        # @brief Counter for the second phase of the barrier.
        self.count_threads2 = self.num_threads
        # @brief Lock to protect access to the thread counters.
        self.counter_lock = Lock()
        # @brief Semaphore for the first phase of waiting threads.
        self.threads_sem1 = Semaphore(0)
        # @brief Semaphore for the second phase of waiting threads.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks until all participating threads have reached this point.

        This method orchestrates the two phases of the barrier to ensure all threads
        synchronize before proceeding, allowing for reusability.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief First phase of the barrier synchronization.

        Threads decrement a counter and the last thread to reach zero releases all
        waiting threads in this phase.
        Invariant: All threads must pass through this phase before any can proceed to phase 2.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Block Logic: Release all threads waiting in phase 1.
                for _ in xrange(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Second phase of the barrier synchronization.

        Similar to phase 1, threads decrement a counter, and the last thread releases
        all waiting threads for this phase, effectively resetting the barrier for reuse.
        Invariant: All threads must pass through phase 1 and this phase for full synchronization.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Block Logic: Release all threads waiting in phase 2.
                for _ in xrange(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.

    This class manages sensor data, executes scripts, and interacts with a supervisor
    within a network. It uses a shared barrier for synchronization and a hash of locks
    for concurrent data access.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's sensor readings.
                            Keys are locations, values are data.
        @param supervisor: A reference to the supervisor object that manages the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # @brief Event to signal that no new scripts have been received, initiating timepoint processing.
        self.none_script_received = Event()
        # @brief List to store assigned scripts and their locations.
        self.scripts = []
        # @brief Event to signal that processing for a timepoint is done (unused in this version).
        self.timepoint_done = Event()
        # @brief The thread responsible for running the device's main logic.
        self.thread = DeviceThread(self)
        self.thread.start()
        # @brief Flag to indicate the end of a timepoint (unused in this version).
        self.timepoint_end = 0
        # @brief Synchronization barrier for coordinating timepoints across devices.
        self.barrier = None
        # @brief Dictionary of locks, indexed by location, to protect sensor data access.
        self.lock_hash = None

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """
        @brief Sets the shared barrier for this device.

        @param barrier: The ReusableBarrier instance to be used for synchronization.
        """
        self.barrier = barrier

    def set_locks(self, lock_hash):
        """
        @brief Sets the shared lock hash for this device.

        @param lock_hash: A dictionary of locks, where keys are locations and values are Lock objects.
        """
        self.lock_hash = lock_hash

    def setup_devices(self, devices):
        """
        @brief Sets up the synchronization barrier and distributed locks across devices.

        The device with the minimum ID initializes the barrier and lock hash,
        then distributes them to other devices. This ensures a single shared
        barrier and set of locks across the network.

        @param devices: A list of all Device objects in the network.
        """
        # Block Logic: Identify the device with the minimum ID to act as the initializer.
        ids_list = []
        for dev in devices:
            ids_list.append(dev.device_id)


        if self.device_id == min(ids_list):
            # Block Logic: Initialize a new ReusableBarrier and a hash of Locks.
            self.barrier = ReusableBarrier(len(devices))
            self.lock_hash = {}

            # Block Logic: For each sensor data location across all devices, create a unique Lock.
            for dev in devices:
                for location in dev.sensor_data:
                    if location not in self.lock_hash:
                        self.lock_hash[location] = Lock()

            # Block Logic: Distribute the initialized barrier and lock hash to other devices.
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.set_barrier(self.barrier)
                    dev.set_locks(self.lock_hash)


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script is provided, it's added to the list of scripts.
        If no script is provided, it signals that script assignment is complete for the timepoint.

        @param script: The script object to be executed.
        @param location: The data location (index) the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.none_script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The data location (index) to retrieve data from.
        @return The sensor data at the specified location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.

        @param location: The data location (index) to set data for.
        @param data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's main thread, ensuring proper termination.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main thread for a Device, managing its operational lifecycle.

    This thread continuously fetches neighbor information from the supervisor,
    executes assigned scripts using worker threads (`MyThread` instances), and
    synchronizes with other devices via a barrier after each timepoint.
    It manages a semaphore to limit concurrent script execution.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # @brief Semaphore to limit the number of concurrently executing MyThread instances.
        self.semaphore = Semaphore(value=8)

    def run(self):
        """
        @brief The main execution loop for the device thread.

        This loop continuously performs the following actions:
        1. Fetches current neighbor information from the supervisor.
        2. Waits for scripts to be assigned for the current timepoint.
        3. Launches `MyThread` instances to execute assigned scripts concurrently,
           managing concurrency with a semaphore.
        4. Waits for all `MyThread` instances to complete.
        5. Synchronizes with other devices using the barrier.
        Invariant: The device processes data in discrete timepoints, synchronizing with the network
                   after each timepoint.
        """
        while True:
            # Block Logic: Fetch updated neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Wait until all scripts for the current timepoint have been assigned.
            self.device.none_script_received.wait()
            # Inline: Clear the event for the next timepoint.
            self.device.none_script_received.clear()

            thread_list = [] # @brief List to hold MyThread worker instances.

            # Block Logic: Create and start MyThread instances for each assigned script.
            # Concurrency is managed by the semaphore passed to MyThread.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, neighbours, script, location,
                    self.semaphore)
                thread.start()
                thread_list.append(thread)

            # Block Logic: Wait for all MyThread instances to complete their execution.
            for i in xrange(len(thread_list)):
                thread_list[i].join()

            # Block Logic: Synchronize with all other devices in the network via the barrier.
            self.device.barrier.wait()

class MyThread(Thread):
    """
    @brief A worker thread that executes a script on a specific data location.

    This thread collects data from its device and neighbors, runs an assigned script,
    and then disseminates the results. It ensures data consistency using shared locks
    and manages concurrency with a semaphore.
    """
    

    def __init__(self, device, neighbours, script, location, semaphore):
        """
        @brief Initializes a new MyThread instance.

        @param device: The Device object this thread is associated with.
        @param neighbours: A list of neighboring Device objects.
        @param script: The script object to be executed.
        @param location: The data location (index) this thread will operate on.
        @param semaphore: A Semaphore instance to control concurrent access to shared resources.
        """
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.semaphore = semaphore

    def run(self):
        """
        @brief Executes the script for the assigned location.

        This method acquires a permit from the semaphore, then acquires a lock
        for the specific data location, collects data from the device and its neighbors,
        runs the script with the collected data, and then updates the device and
        neighbor data with the results. Finally, it releases the lock and the semaphore permit.
        Ensures thread safety using both semaphores for concurrency control and locks for data integrity.
        """
        # Block Logic: Acquire a permit from the semaphore to limit concurrent thread execution.
        self.semaphore.acquire()

        # Block Logic: Acquire a lock specific to the data location to ensure exclusive access.
        self.device.lock_hash[self.location].acquire()

        script_data = []

        # Block Logic: Collect data from neighboring devices for the current location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Block Logic: Collect data from the current device for the current location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Block Logic: Execute the assigned script with the collected data.
            result = self.script.run(script_data)

            # Block Logic: Disseminate the script's result to neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)

            # Block Logic: Update the current device's data with the script's result.
            self.device.set_data(self.location, result)

        # Block Logic: Release the lock for the data location.
        self.device.lock_hash[self.location].release()

        # Block Logic: Release the semaphore permit.
        self.semaphore.release()
