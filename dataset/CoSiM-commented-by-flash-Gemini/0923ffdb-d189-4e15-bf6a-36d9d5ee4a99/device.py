"""
This script defines the core components for a simulated distributed device network, including
a reusable barrier for synchronization, a device representation, and threads for device operation
and script execution.

It focuses on managing sensor data across devices, synchronizing their operations, and executing
scripts that process local and neighboring device data.
"""

from threading import Thread, Event, Semaphore, Lock

class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.
    This barrier allows a fixed number of threads to wait for each other to reach a common point
    before any can proceed, and can then be reset and reused.
    """

    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier with a specified number of threads to synchronize.

        Args:
            num_threads (int): The total number of threads that must reach the barrier to release it.
        """
        self.num_threads = num_threads
        # Tracks the count of threads for the first phase of the barrier.
        self.count_threads1 = [self.num_threads]
        # Tracks the count of threads for the second phase of the barrier.
        self.count_threads2 = [self.num_threads]
        # Mutex to protect access to the thread count.
        self.count_lock = Lock()
        # Semaphore for the first phase, initialized to 0 to block threads until all arrive.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for the second phase, initialized to 0 to block threads until all arrive.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all `num_threads` have arrived.
        This method orchestrates a two-phase synchronization to allow for barrier reuse.
        """
        # First phase of synchronization.
        self.phase(self.count_threads1, self.threads_sem1)
        # Second phase of synchronization, ensuring proper reset and reuse.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Manages a single phase of the barrier synchronization.

        Args:
            count_threads (list): A mutable list (to allow modification within the lock) storing
                                  the current count of threads remaining to reach this phase.
            threads_sem (Semaphore): The semaphore associated with this phase to block/release threads.
        """
        # Critical section: Decrement thread count and check for barrier release condition.
        with self.count_lock:
            # Decrement the number of threads waiting for this phase.
            count_threads[0] -= 1
            # Check if all threads have arrived at this phase.
            # Pre-condition: `count_threads[0]` becomes 0 when the last thread enters the critical section.
            if count_threads[0] == 0:
                # If all threads have arrived, release them by incrementing the semaphore `num_threads` times.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the thread count for the next use of this phase.
                count_threads[0] = self.num_threads
        # Block the current thread until the semaphore is released, indicating all threads have passed this phase.
        threads_sem.acquire()


class Device(object):
    """
    @brief Represents a single device in the distributed network, managing its sensor data,
    assigned scripts, and interaction with a supervisor and other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to their current sensor data.
            supervisor (Supervisor): A reference to the central supervisor managing the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to the device.
        self.script_received = Event()
        # List of (script, location) tuples assigned to this device for execution.
        self.scripts = []
        # List to store references to other devices in the network.
        self.devices = []
        # Event to signal completion of a timepoint's script execution.
        self.timepoint_done = Event()
        # Dedicated thread for the device's operational logic.
        self.thread = DeviceThread(self)
        # Barrier for synchronizing all devices at specific timepoints.
        self.barrier = None
        # Starts the device's operational thread upon initialization.
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures shared resources (barrier) across all devices in the network.
        This method ensures all devices share the same synchronization barrier.

        Args:
            devices (list): A list of all Device instances in the network.
        """
        # Block Logic: Initializes a shared barrier if not already set.
        # Pre-condition: This device's barrier is currently None.
        if self.barrier is None:
            # Creates a new ReusableBarrier for all devices.
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            # Assigns the newly created barrier to all devices that don't have one yet.
            # Invariant: All devices in the provided list will end up with the same barrier instance.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        # Stores references to all devices in the network.
        # Invariant: 'self.devices' will contain all valid device references from the input list.
        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific location.
        If no script is provided (None), it signals that the timepoint is done for this device.

        Args:
            script (Script): The script object to be executed.
            location (int): The location ID associated with the script.
        """
        # Block Logic: Manages script assignment or signals timepoint completion.
        if script is not None:
            # If a script is provided, add it to the device's list of scripts and set the script_received event.
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # If no script is provided, it indicates that this device has no more scripts
            # to run for the current timepoint, so set the timepoint_done event.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location ID for which to retrieve data.

        Returns:
            Any: The sensor data for the specified location, or None if the location is not found.
        """
        # Block Logic: Safely retrieves sensor data if available.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location.

        Args:
            location (int): The location ID for which to set data.
            data (Any): The new sensor data to set.
        """
        # Block Logic: Updates sensor data if the location exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Waits for the device's operational thread to complete, ensuring graceful shutdown.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main operational thread for a Device, responsible for continuously
    executing assigned scripts, synchronizing with other devices, and managing its lifecycle.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance that this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop of the device thread. It continuously fetches neighbors,
        waits for timepoint completion signals, executes scripts, and then
        synchronizes with other devices via a barrier.
        """
        # Invariant: The device thread continuously runs until a shutdown signal is received from the supervisor.
        while True:
            # Block Logic: Retrieves neighboring devices from the supervisor.
            # Pre-condition: The supervisor is active and can provide neighbor information.
            neighbours = self.device.supervisor.get_neighbours()
            # If no neighbors are returned (e.g., supervisor signals shutdown), break the loop.
            if neighbours is None:
                break

            # Waits until the `timepoint_done` event is set, indicating all scripts for the current timepoint are assigned.
            # This implicitly means that all scripts for this timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Iterates through each assigned script and executes it.
            # Invariant: Each script assigned to the device for the current timepoint is processed.
            for (script, location) in self.device.scripts:
                script_data = []
                # Block Logic: Gathers sensor data from neighboring devices for the current script's location.
                for device in neighbours:
                    data = device.get_data(location)
                    # Invariant: Only valid sensor data (not None) from neighbors is added to `script_data`.
                    if data is not None:
                        script_data.append(data)

                # Gathers local sensor data for the current script's location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script and propagates results if data is available.
                # Pre-condition: `script_data` contains at least one data point.
                if script_data != []:
                    # Executes the script with the collected data.
                    result = script.run(script_data)
                    # Updates the sensor data of neighboring devices with the script's result.
                    # Invariant: Each neighbor's sensor data for the current `location` is updated with `result`.
                    for device in neighbours:
                        device.set_data(location, result)
                    # Updates the local device's sensor data with the script's result.
                    self.device.set_data(location, result)
            
            # Clears the `timepoint_done` event, resetting it for the next timepoint.
            # This indicates that all scripts for the current timepoint have been executed.
            self.device.timepoint_done.clear()
            # Synchronizes with all other devices using the reusable barrier.
            # All devices must reach this point before any can proceed to the next timepoint.
            self.device.barrier.wait()