"""
@file device.py
@brief Implements a distributed device simulation with script-based data processing and multi-threaded execution.
@details This module defines a simulated environment where multiple 'devices' can process sensor data.
The devices operate in parallel, coordinated by a central supervisor and synchronized using barriers.
Each device uses multiple threads to execute scripts on its own data and data from its neighbors.
"""

from threading import Event, Thread, Lock, Semaphore, RLock

class ReusableBarrierSem(object):
    """
    @brief A reusable barrier implementation using semaphores for thread synchronization.
    @details This barrier synchronizes a fixed number of threads over two phases, ensuring that
    all threads reach the barrier before any are allowed to proceed. It is reusable,
    meaning it can be used multiple times to synchronize the same set of threads.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier for a given number of threads.
        @param num_threads The number of threads to synchronize.
        """
        self.num_threads = num_threads
        # Counter for threads arriving at the first phase.
        self.count_threads1 = self.num_threads
        # Counter for threads arriving at the second phase.
        self.count_threads2 = self.num_threads
        # Lock to protect access to the counters.
        self.counter_lock = Lock()
        # Semaphore for the first synchronization phase.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for the second synchronization phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Causes a thread to wait at the barrier until all participating threads have arrived.
        @details The barrier is implemented in two phases to prevent race conditions where
        a thread loops and re-enters the barrier before all other threads have left it from the previous wait.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Handles the first phase of the barrier synchronization.
        @details Threads decrement a counter. The last thread to arrive releases all waiting
        threads by signaling a semaphore `num_threads` times.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            # Pre-condition: Last thread arrives.
            # Invariant: All threads are waiting on threads_sem1.
            if self.count_threads1 == 0:
                # Release all waiting threads for phase 1.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset counter for the next use of the barrier.
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Handles the second phase of the barrier synchronization.
        @details This phase ensures that all threads have exited the first phase before the barrier
        can be used again. The logic is identical to phase1 but uses a separate counter and semaphore.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            # Pre-condition: Last thread arrives.
            # Invariant: All threads are waiting on threads_sem2.
            if self.count_threads2 == 0:
                # Release all waiting threads for phase 2.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset counter for the next use of the barrier.
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()


class Device(object):
    """
    @brief Represents a single device in the distributed network simulation.
    @details Each device holds sensor data, can execute scripts, and communicates with
    neighboring devices. It operates concurrently using multiple threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary representing the device's local sensor data.
        @param supervisor A supervisor object that manages the network and provides neighbor information.
        """
        # Flag to check if the list of neighbors has been fetched for the current timepoint.
        self.got_neighbours = False
        # List of neighboring device objects.
        self.neighbours_list = []

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been received.
        self.script_received = Event()
        # A list to store scripts to be executed.
        self.scripts = []
        # Event to signal that a timepoint (simulation step) is complete.
        self.timepoint_done = Event()

        # Lock to protect access to the neighbors list.
        self.neighbours_lock = Lock()

        self.nr_devices = 0

        # The root device (ID 0) is responsible for managing the global thread barrier.
        self.root_device = None
        self.thread_barrier = None

        self.threads = []
        self.nr_threads = 8

        # A dictionary of locks, one for each data location, to ensure thread-safe access to sensor data.
        self.locks = {}
        for location in sensor_data:
            self.locks[location] = RLock()

        # Create and start device threads.
        for i in range(self.nr_threads):
            self.threads.append(DeviceThread(self, i))
            self.threads[i].start()


    def __str__(self):
        """
        @brief Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs setup by providing the device with a list of all devices in the simulation.
        @param devices A list of all device objects in the network.
        """
        self.nr_devices = len(devices)

        # The root device (ID 0) initializes the global barrier for all threads of all devices.
        if self.device_id == 0:
            self.root_device = self
            self.thread_barrier = ReusableBarrierSem(self.nr_devices * self.nr_threads)

        # Other devices find and store a reference to the root device to access the shared barrier.
        if self.device_id != 0:
            for dev in devices:
                if dev.device_id == 0:
                    self.root_device = dev

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution or signals the end of a timepoint.
        @param script The script to be executed.
        @param location The data location the script will operate on.
        """
        # If a script is provided, add it to the queue and signal the threads.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        # If script is None, it signals the end of the current timepoint's script assignments.
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data from a specific location.
        @param location The key for the desired sensor data.
        @return The sensor data value, or None if the location does not exist.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates sensor data at a specific location.
        @param location The key for the sensor data to update.
        @param data The new value to be stored.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining all of its worker threads.
        """
        for i in range(self.nr_threads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    @brief A worker thread for a Device.
    @details This thread is responsible for fetching neighbors, executing scripts on sensor data,
    and synchronizing with other threads across all devices.
    """

    def __init__(self, device, thread_id):
        """
        @brief Initializes the device thread.
        @param device The parent device object.
        @param thread_id A unique ID for the thread within its parent device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        @brief The main execution loop for the device thread.
        """
        while True:
            # Block Logic: Lazily fetch neighbor list once per timepoint.
            # A lock ensures that the neighbor list is fetched by only one thread per device.
            self.device.neighbours_lock.acquire()

            # Pre-condition: The neighbors for this timepoint have not been fetched.
            # Invariant: After this block, the 'neighbors' variable holds the list of neighbors for the current timepoint.
            if not self.device.got_neighbours:
                self.device.got_neighbours = True
                neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours_list = neighbours
            else:
                neighbours = self.device.neighbours_list

            self.device.neighbours_lock.release()

            # A None value for neighbors is the signal to shut down.
            if neighbours is None:
                break

            # Wait until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Distribute script execution among the device's threads.
            # Each thread processes a subset of the assigned scripts based on its thread_id.
            for i in range(len(self.device.scripts)):
                # Distribute work based on thread ID.
                if i % self.device.nr_threads == self.thread_id:
                    (script, location) = self.device.scripts[i]

                    script_data = []
                    
                    # Block Logic: Acquire locks and gather data from all neighbor devices.
                    # This prevents data from being modified while it is being read.
                    for device in neighbours:
                        # Acquire lock for the specific data location on the neighbor.
                        if location in device.locks:
                            device.locks[location].acquire()

                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                    # Acquire lock and gather data from the local device.
                    if location in self.device.locks:
                        self.device.locks[location].acquire()
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                    
                    # Pre-condition: At least one data point was gathered.
                    # Invariant: The script processes the aggregated data and produces a single result.
                    if script_data != []:
                        # Execute the script with the collected data.
                        result = script.run(script_data)

                        # Block Logic: Broadcast the result by updating the data on all neighbors.
                        for device in neighbours:
                            device.set_data(location, result)

                        # Update the local device's data.
                        self.device.set_data(location, result)
                    
                    # Block Logic: Release all acquired locks.
                    # This must be done to prevent deadlocks in subsequent operations.
                    if location in self.device.locks:
                        self.device.locks[location].release()
                    for device in neighbours:
                        if location in device.locks:
                            device.locks[location].release()

            # Synchronize with all other threads from all devices to signal the end of the computation phase for this timepoint.
            self.device.root_device.thread_barrier.wait()

            # Reset events and flags for the next timepoint.
            self.device.timepoint_done.clear()
            self.device.got_neighbours = False

            # Synchronize again to ensure all threads have completed their reset before the next timepoint begins.
            self.device.root_device.thread_barrier.wait()
