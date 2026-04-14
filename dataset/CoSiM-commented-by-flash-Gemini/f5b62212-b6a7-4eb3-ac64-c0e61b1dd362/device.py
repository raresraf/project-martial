"""
@file device.py
@brief Implements a simulated distributed system with devices, synchronization barriers, and worker threads.

This module defines the core components for simulating a network of devices that can
receive and execute scripts, exchange data with neighbors, and synchronize their
operations using a reusable barrier mechanism. It's designed to model distributed
computation or data processing scenarios.
"""

from threading import Thread, Lock, Semaphore, Event


class ReusableBarrier(object):
    """
    @class ReusableBarrier
    @brief A synchronization barrier that can be used multiple times by a fixed number of threads.

    Algorithm: This barrier uses two alternating phases, each managed by a counter
    and a semaphore. Threads decrement a counter, and when the counter reaches zero,
    all waiting threads are released. This two-phase approach allows the barrier
    to be reset and reused without deadlocking.
    Time Complexity:
        - `wait()`: O(num_threads) for releasing semaphores in the release phase,
          and O(1) for acquiring a semaphore in the waiting phase.
    Space Complexity: O(1) beyond thread-local storage for counters and semaphores.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier with a specified number of participating threads.
        @param num_threads The total number of threads that must reach the barrier before it releases.
        """
        self.num_threads = num_threads
        # Invariant: These counters track how many threads have arrived at each phase of the barrier.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # Invariant: Protects access to the shared counters.
        self.count_lock = Lock()
        # Invariant: Semaphores used to block and release threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have called this method.
        The barrier is reusable, meaning it can be used again after all threads have passed.
        """
        # Block Logic: Executes the first phase of the barrier synchronization.
        self.phase(self.count_threads1, self.threads_sem1)
        # Block Logic: Executes the second phase of the barrier synchronization.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Internal method to manage a single phase of the barrier synchronization.
        @param count_threads A list containing the current count of threads for this phase.
        @param threads_sem The semaphore associated with this phase.
        """
        # Block Logic: Decrements the thread counter and checks if all threads have arrived.
        with self.count_lock:
            count_threads[0] -= 1
            # Pre-condition: `count_threads[0]` is 0, meaning all threads have reached this phase.
            if count_threads[0] == 0:
                # Block Logic: Releases all waiting threads and resets the counter for reuse.
                for dummy_i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        # Block Logic: Blocks the current thread until the semaphore is released by the last arriving thread.
        threads_sem.acquire()


class Device(object):
    """
    @class Device
    @brief Represents a single device in a simulated distributed system.

    A device can hold sensor data, receive and execute scripts, and interact with a supervisor
    to get information about its neighbors. Each device runs its operations in a dedicated thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary holding the device's sensor data.
        @param supervisor The supervisor object responsible for managing device interactions.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Invariant: Event to signal when a new script has been received.
        self.script_received = Event()
        # Invariant: List to store (script, location) tuples for execution.
        self.scripts = []
        # Invariant: Event to signal when a timepoint's processing is done.
        self.timepoint_done = Event()
        # Invariant: The dedicated thread for this device's operations.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Invariant: Reference to the ReusableBarrier, set up by device 0.
        self.rbarrier = None

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    @staticmethod
    def send_barrier(devices, barrier):
        """
        @brief Static method to assign a reusable barrier to a list of devices.
        @param devices A list of Device objects.
        @param barrier The ReusableBarrier instance to assign.
        """
        # Block Logic: Iterates through devices and assigns the barrier if not already assigned.
        for dev in devices:
            if dev.rbarrier is None and dev is not None:
                dev.rbarrier = barrier

    def setup_devices(self, devices):
        """
        @brief Sets up the reusable barrier for all devices.
        This method should typically be called by a designated device (e.g., device_id == 0).
        @param devices A list of all Device objects in the simulation.
        """
        # Block Logic: Only device 0 creates and distributes the barrier.
        if self.device_id == 0:
            mybarrier = ReusableBarrier(len(devices))
            self.rbarrier = mybarrier
            self.send_barrier(devices, mybarrier)

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.
        If a script is provided, it's added to the queue and `script_received` is set.
        If `script` is None, it signals that the timepoint is done.
        @param script The script object to execute, or None to signal timepoint completion.
        @param location The data location relevant to the script.
        """
        # Block Logic: Handles script assignment or timepoint completion signaling.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location The key for the sensor data.
        @return The data associated with the location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.
        @param location The key for the sensor data.
        @param data The new data to set.
        """
        # Pre-condition: The location must already exist in `sensor_data` to be updated.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its associated thread.
        """
        self.thread.join()


class MyWorker(Thread):
    """
    @class MyWorker
    @brief A worker thread that executes a given script on collected data.

    Each worker is responsible for gathering relevant data from its device and its
    neighbors, running a specified script, and then updating the data on its
    device and neighbors based on the script's output.
    """

    def __init__(self, device, location, neighbours, script):
        """
        @brief Initializes a MyWorker thread.
        @param device The Device object this worker belongs to.
        @param location The specific data location this worker is processing.
        @param neighbours A list of neighboring Device objects.
        @param script The script object to be executed.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script
        # Invariant: Stores data collected for script execution.
        self.script_data = []

    def run(self):
        """
        @brief The main execution logic for the MyWorker thread.

        Block Logic: Gathers data from neighboring devices and its own device,
        executes the assigned script, and then updates the relevant data on
        neighbors and itself.
        """
        # Block Logic: Collects data from neighboring devices for the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            # Invariant: Only append if data is valid (not None).
            if data is not None:
                self.script_data.append(data)

        # Block Logic: Collects data from its own device for the specified location.
        data = self.device.get_data(self.location)
        # Invariant: Only append if data is valid (not None).
        if data is not None:
            self.script_data.append(data)

        # Pre-condition: `script_data` is not empty, indicating there is data to process.
        if self.script_data != []:
            # Block Logic: Executes the script with the collected data.
            result = self.script.run(self.script_data)

            # Block Logic: Updates the data on neighboring devices with the script's result.
            for device in self.neighbours:
                device.set_data(self.location, result)

            # Block Logic: Updates the data on its own device with the script's result.
            self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief The dedicated operational thread for a Device object.

    This thread continuously performs the device's main loop:
    1. Fetches information about neighboring devices from the supervisor.
    2. Waits for a timepoint to be declared complete (i.e., scripts assigned).
    3. Dispatches `MyWorker` threads to execute assigned scripts.
    4. Synchronizes with other DeviceThreads using a `ReusableBarrier` to ensure
       all devices complete their timepoint processing before proceeding.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device object this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Invariant: The loop continues as long as the supervisor provides neighbor information.
        """
        while True:
            # Block Logic: Retrieves current neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If `neighbours` is None, it signifies a shutdown or termination condition.
            if neighbours is None:
                break

            # Block Logic: Waits until the `timepoint_done` event is set, indicating scripts are ready.
            self.device.timepoint_done.wait()

            thrds = []
            # Block Logic: Creates and stores `MyWorker` threads for each assigned script.
            for (script, location) in self.device.scripts:
                thrd = MyWorker(self.device, location, neighbours, script)
                thrds.append(thrd)

            # Block Logic: Starts all worker threads and waits for their completion.
            for thrd in thrds:
                thrd.start()
            for thrd in thrds:
                thrd.join()

            # Block Logic: Resets the `timepoint_done` event for the next cycle and synchronizes
            # with other devices using the reusable barrier.
            self.device.timepoint_done.clear()
            self.device.rbarrier.wait()
