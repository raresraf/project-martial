"""
@8da38bb4-aa56-44c7-8f51-52c0baf19076/device.py
@brief This script implements device behavior for a distributed system simulation,
featuring thread-safe data access, synchronization via a reusable barrier, and
parallel script execution across multiple worker threads within each device.
It focuses on managing sensor data, executing scripts, and coordinating with
neighboring devices and a supervisor.
Domain: Concurrency, Distributed Systems, Simulation, Thread Synchronization, Parallel Processing.
"""

from threading import Event, Lock, Thread, RLock, Semaphore

class ReusableBarrier():
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.
    It allows a fixed number of threads to wait for each other before proceeding
    together, and can be reused multiple times.
    Algorithm: Double-phase semaphore-based barrier.
    Time Complexity: O(N) for each `wait` call where N is the number of threads, due to semaphore releases.
    Space Complexity: O(1) for internal state.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # Inline: Counter for the first phase of the barrier. Using a list to allow modification within methods.
        self.count_threads1 = [self.num_threads]
        # Inline: Counter for the second phase of the barrier.
        self.count_threads2 = [self.num_threads]
        # Inline: Lock to protect access to the thread counters.
        self.count_lock = Lock()
        # Inline: Semaphore for synchronizing threads in the first phase.
        self.threads_sem1 = Semaphore(0)
        # Inline: Semaphore for synchronizing threads in the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all participating threads
        have reached this point.
        """
        # Block Logic: Execute the first phase of the barrier synchronization.
        self.phase(self.count_threads1, self.threads_sem1)
        # Block Logic: Execute the second phase of the barrier synchronization.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements a single phase of the barrier synchronization.
        @param count_threads: The counter list for the current phase.
        @param threads_sem: The semaphore for the current phase.
        """
        # Block Logic: Acquire lock to safely decrement the thread counter.
        with self.count_lock:
            count_threads[0] -= 1
            # Pre-condition: If this is the last thread to reach the barrier in this phase.
            if count_threads[0] == 0:
                # Block Logic: Release the semaphore 'num_threads' times to unblock all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Inline: Reset the counter for the next use of this phase of the barrier.
                count_threads[0] = self.num_threads
        # Post-condition: Acquire the semaphore, waiting if not yet released by the last thread.
        threads_sem.acquire()


class Device(object):
    """
    @brief Represents a single device in the simulated distributed system.
    Each device manages its own sensor data, executes scripts, and coordinates
    with a supervisor and neighboring devices. This version uses RLock for
    certain operations and a ReusableBarrier for synchronization across devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data for various locations.
        @param supervisor: A reference to the supervisor object for coordination.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Inline: Event to signal that a new script has been assigned to the device.
        self.script_received = Event()
        # Inline: List to store assigned scripts and their locations.
        self.scripts = []
        # Inline: Event to signal that all scripts for the current timepoint have been processed.
        self.timepoint_done = Event()
        # Inline: The main thread for the device's operational logic.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Inline: Reentrant lock for protecting shared data within the device (e.g., sensor_data).
        self.lock = RLock()
        # Inline: Reentrant lock for protecting script assignment operations.
        self.script_lock = RLock()
        # Inline: Reentrant lock for protecting run operations (though its specific use here might be redundant with DeviceThread).
        self.run_lock = RLock()
        # Inline: Reference to the shared ReusableBarrier, initialized by device 0.
        self.barrier = None

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        @return: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the device with a list of other devices in the system.
        Device 0 initializes a shared barrier and distributes it to all other devices.
        @param devices: A list of all Device objects in the simulation.
        """
        # Block Logic: Only device with device_id 0 initializes the shared barrier.
        if self.device_id is 0:
            # Inline: Create a new barrier instance for all devices.
            self.barrier = ReusableBarrier(len(devices))
            # Block Logic: Distribute the initialized barrier to all devices.
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location for this device.
        @param script: The script object to be executed.
        @param location: The data location the script operates on.
        """
        # Block Logic: Acquire script_lock to ensure thread-safe assignment of scripts.
        with self.script_lock:
            if script is not None:
                self.scripts.append((script, location))
                self.script_received.set()
            else:
                self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location in a thread-safe manner.
        @param location: The location identifier for which to retrieve data.
        @return: The sensor data for the location, or None if the location is not present.
        """
        # Block Logic: Acquire lock to ensure thread-safe access to sensor data.
        with self.lock:
            result = self.sensor_data[location] if location in self.sensor_data else None
        return result

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location in a thread-safe manner.
        @param location: The location identifier for which to set data.
        @param data: The new data value to set.
        """
        # Block Logic: Acquire lock to ensure thread-safe modification of sensor data.
        with self.lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its main thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main thread responsible for a Device's operational loop.
    It continuously waits for timepoint signals, fetches neighbors, creates
    worker threads for concurrent script execution, and synchronizes via a barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device: The Device object that this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main loop for the device thread. It continuously fetches neighbors,
        waits for timepoint signals, creates and manages worker threads for script
        processing, and synchronizes with other devices using a barrier.
        """
        # Block Logic: Continuous operational loop for the device thread.
        while True:
            # Block Logic: Fetch the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If supervisor returns None, it indicates simulation termination.
            if neighbours is None:
                break

            # Inline: Add the current device itself to the list of neighbors for data collection purposes.
            neighbours.append(self.device)

            # Block Logic: Wait until the current timepoint's scripts are ready to be processed.
            self.device.timepoint_done.wait()

            # @var num_threads: The number of concurrent worker threads to spawn for script execution.
            num_threads = 8
            # Block Logic: Create a list of worker threads, each configured to process a subset of scripts.
            threads = [Thread(target=self.concurrent_work,
                              args=(neighbours, i, num_threads)) for i in range(num_threads)]

            # Block Logic: Start all worker threads concurrently.
            for thread in threads:
                thread.start()

            # Block Logic: Wait for all worker threads to complete their execution.
            for thread in threads:
                thread.join()

            # Block Logic: Synchronize with other devices via the shared barrier, ensuring all devices
            # complete their processing for the current timepoint before proceeding.
            self.device.barrier.wait()
            # Inline: Clear the timepoint_done event for the next cycle.
            self.device.timepoint_done.clear()

    def concurrent_work(self, neighbours, thread_id, num_threads):
        """
        @brief This method is executed by worker threads to process a subset of assigned scripts.
        It collects data from neighbors and itself, runs the script, and updates data.
        @param neighbours: A list of neighboring devices, including the current device.
        @param thread_id: The ID of the current worker thread, used to determine its assigned scripts.
        @param num_threads: The total number of worker threads.
        """
        # Block Logic: Iterate through the scripts assigned to this specific worker thread.
        for (script, location) in self.keep_assigned(self.device.scripts, thread_id, num_threads):
            script_data = []
            
            # Block Logic: Collect data from all neighboring devices (including self) for the current location.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Pre-condition: Only execute the script if there is data to process.
            if script_data != []:
                # Inline: Execute the script with the collected data.
                result = script.run(script_data)

                # Block Logic: Update data on all neighboring devices with the script's result.
                # Note: This logic for updating neighbors with `max(result, device.get_data(location))`
                # implies a specific data aggregation strategy (e.g., maximum value propagation).
                for device in neighbours:
                    res = max(result, device.get_data(location))
                    device.set_data(location, res)

    def keep_assigned(self, scripts, thread_id, num_threads):
        """
        @brief Filters the list of scripts to return only those assigned to a specific worker thread.
        @param scripts: The full list of scripts assigned to the device.
        @param thread_id: The ID of the current worker thread.
        @param num_threads: The total number of worker threads.
        @return: A list of scripts assigned to this worker thread.
        """
        assigned_scripts = []
        # Block Logic: Distribute scripts among worker threads using a modulo operation.
        for i, script in enumerate(scripts):
            if i % num_threads is thread_id: # Inline: Script is assigned if its index modulo num_threads matches thread_id.
                assigned_scripts.append(script)

        return assigned_scripts
