"""
@8b4daef6-14f0-4cb5-9247-4d8f0dc38a55/device.py
@brief This script implements a distributed system simulation with device behavior,
thread management, and synchronization using a reusable barrier. It models devices
collecting sensor data, processing it with scripts, and coordinating data updates
among neighbors under a supervisor.
Domain: Concurrency, Distributed Systems, Simulation, Thread Synchronization.
"""

from threading import Event, Thread, Lock, Semaphore

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
            # Pre-condition: If this is the last thread to reach the barrier.
            if count_threads[0] == 0:
                # Block Logic: Release the semaphore 'num_threads' times to unblock all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Inline: Reset the counter for the next use of the barrier.
                count_threads[0] = self.num_threads
        # Post-condition: Acquire the semaphore, waiting if not yet released by the last thread.
        threads_sem.acquire()

class Device(object):
    """
    @brief Represents a single device in the simulated distributed system.
    Each device manages its own sensor data, executes scripts, and coordinates
    with a supervisor and neighboring devices.
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
        # Inline: List to store references to neighboring devices.
        self.devices = []
        # Inline: Event to signal that all scripts for the current timepoint have been processed.
        self.timepoint_done = Event()
        # Inline: The main thread for the device's operational logic.
        self.thread = DeviceThread(self)
        # Inline: Reference to the reusable barrier for synchronizing with other devices.
        self.barrier = None
        # Inline: List to hold threads spawned for script execution within a timepoint.
        self.list_thread = []
        # Inline: Start the main device thread.
        self.thread.start()
        # Inline: Array to store locks for different locations, preventing concurrent writes. Max 100 locations.
        self.location_lock = [None] * 100

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the device with a list of other devices in the system.
        Initializes a shared barrier if it hasn't been set up yet.
        @param devices: A list of all Device objects in the simulation.
        """
        # Block Logic: Initialize a shared ReusableBarrier if not already set.
        if self.barrier is None:
            # Inline: Create a new barrier instance for all devices.
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            # Block Logic: Assign the newly created barrier to all devices.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        # Block Logic: Populate the list of neighboring devices.
        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location for this device.
        Manages location-specific locks and signals script reception.
        @param script: The script object to be executed.
        @param location: The data location the script operates on.
        """
        ok = 0 # Inline: Flag to indicate if a lock for the location was found among neighbors.

        # Block Logic: If a script is provided, assign it and manage locks.
        if script is not None:
            self.scripts.append((script, location))
            # Block Logic: If no lock exists for this location, try to reuse one from a neighbor.
            if self.location_lock[location] is None:
                # Block Logic: Iterate through neighbors to find an existing lock for this location.
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        ok = 1
                        break
                # Block Logic: If no existing lock was found, create a new one for this location.
                if ok == 0:
                    self.location_lock[location] = Lock()
            # Inline: Signal that a script has been received.
            self.script_received.set()
        # Block Logic: If no script is provided, signal that the timepoint is done (end of a simulation step).
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The location identifier for which to retrieve data.
        @return: The sensor data for the location, or None if the location is not present.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.
        @param location: The location identifier for which to set data.
        @param data: The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its main thread.
        """
        self.thread.join()

class NewThread(Thread):
    """
    @brief A worker thread responsible for executing a single script on sensor data
    for a specific location, considering neighboring device data.
    """
    def __init__(self, device, location, script, neighbours):
        """
        @brief Initializes a NewThread instance.
        @param device: The Device object owning this thread.
        @param location: The data location this script operates on.
        @param script: The script object to execute.
        @param neighbours: A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief Executes the assigned script, collects data from neighbors and itself,
        processes it, and updates relevant device data.
        """
        script_data = []
        # Block Logic: Acquire the lock for the current location to prevent race conditions during data access/modification.
        self.device.location_lock[self.location].acquire()
        
        # Block Logic: Collect data from neighboring devices for the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
        # Block Logic: Collect data from the current device for the specified location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Pre-condition: Only execute the script if there is data to process.
        if script_data != []:
            # Inline: Execute the script with the collected data.
            result = self.script.run(script_data)
            
            # Block Logic: Update data on neighboring devices with the script's result.
            for device in self.neighbours:
                device.set_data(self.location, result)
                
            # Block Logic: Update data on the current device with the script's result.
            self.device.set_data(self.location, result)
        # Post-condition: Release the lock for the current location.
        self.device.location_lock[self.location].release()

class DeviceThread(Thread):
    """
    @brief The main thread responsible for a Device's operational loop.
    It continuously waits for scripts, executes them, and synchronizes with other devices.
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
        waits for timepoint signals, executes scripts in parallel, and synchronizes
        with other devices via a barrier.
        """
        # Block Logic: Continuous operational loop for the device.
        while True:
            # Block Logic: Fetch the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If supervisor returns None, it indicates simulation termination.
            if neighbours is None:
                break

            # Block Logic: Wait until the current timepoint's scripts are ready to be processed.
            self.device.timepoint_done.wait()

            # Block Logic: Create a new worker thread for each assigned script.
            # Invariant: Each script is prepared for concurrent execution.
            for (script, location) in self.device.scripts:
                thread = NewThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            # Block Logic: Start all script execution threads concurrently.
            for thread_elem in self.device.list_thread:
                thread_elem.start()
            # Block Logic: Wait for all script execution threads to complete.
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            # Inline: Clear the list of worker threads for the next timepoint.
            self.device.list_thread = []

            # Block Logic: Reset the timepoint_done event for the next cycle.
            self.device.timepoint_done.clear()
            # Block Logic: Wait at the barrier for all devices to complete their timepoint processing.
            self.device.barrier.wait()
