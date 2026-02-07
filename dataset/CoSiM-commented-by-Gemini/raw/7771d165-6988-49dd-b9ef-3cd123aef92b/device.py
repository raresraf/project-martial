


"""
@file device.py
@brief Implements simulated device functionality for a distributed system, including synchronization and thread pooling.
This module defines a `Device` class to represent individual simulated entities, manages concurrent script execution
via `DeviceThread` and a `ThreadPool`, and uses a `ReusableBarrier` for inter-device synchronization.
It facilitates simulating data processing and communication in sensor networks or similar distributed environments.
"""


from threading import Event, Thread, Lock , Condition
from queue import Worker, ThreadPool
from reusable_barrier_semaphore import ReusableBarrier

class Device(object):
    """
    @brief Represents a simulated device in a distributed system, handling sensor data,
    script execution, and synchronization with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data for various locations.
        @param supervisor: A reference to the supervisor entity managing the devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal that a script has been received for processing
        self.wait_neighbours = Event()  # Event to signal that neighbor information is available
        self.scripts = []  # List to store assigned scripts and their locations
        self.neighbours = []  # List of neighboring devices
        self.allDevices = []  # List of all devices in the system
        self.locks = []  # List of Locks for protecting sensor data access by location
        self.pool = ThreadPool(8)  # ThreadPool for managing concurrent script execution
        self.lock = Lock()  # Generic lock for device-level synchronization
        self.thread = DeviceThread(self)  # Main device thread for managing timepoints and neighbors
        self.thread.start()

    def __str__(self):
        """
        @brief Provides a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the device's shared resources and synchronization mechanisms.
        This includes initializing the list of all devices, creating a reusable barrier
        for inter-device synchronization, and setting up per-location locks for data protection.
        @param devices: A list of all Device instances in the system.
        Pre-condition: This method is expected to be called by all devices during initial setup.
        Post-condition: `self.allDevices`, `self.barrier`, and `self.locks` are initialized.
        """
        self.allDevices = devices
        self.barrier = ReusableBarrier(len(devices))

        # Functional Utility: Initializes a list of locks for protecting sensor data access per location.
        for i in range(0, 50):
            self.locks.append(Lock())

        pass

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location on the device.
        If a script is provided, it's added to the device's script list and submitted to the thread pool.
        If `script` is None, it signals that all scripts for the current timepoint have been assigned.
        @param script: The script object to assign, or `None` to signal timepoint script assignment completion.
        @param location: The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.pool.add_task(self.executeScript, script, location)  # Submit script execution to the thread pool
        else:
            self.script_received.set()  # Signal that all scripts for this timepoint have been received

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The location for which to retrieve sensor data.
        @return The sensor data if `location` exists, otherwise `None`.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location if it exists.
        @param location: The location for which to set sensor data.
        @param data: The new sensor data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the main device thread.
        Pre-condition: Assumes `DeviceThread` has a mechanism to gracefully exit its run loop.
        """
        self.thread.join()

    def executeScript(self, script, location):
        """
        @brief Executes a script for a specific location, collecting data from neighbors and updating it.
        This method is designed to be run by a worker from the `ThreadPool`.
        @param script: The script to be executed.
        @param location: The location for which the script is being executed.
        """
        self.wait_neighbours.wait()  # Wait until neighbor information is updated and available

        script_data = []

        # Block Logic: Collect data from neighboring devices, ensuring thread-safe access to their sensor data.
        if not self.neighbours is None:
            for device in self.neighbours:
                device.locks[location].acquire()  # Acquire lock for neighbor's location data
                data = device.get_data(location)
                device.locks[location].release()  # Release lock

                if data is not None:
                    script_data.append(data)

        # Block Logic: Collect data from the current device's sensor data, ensuring thread-safe access.
        self.locks[location].acquire()  # Acquire lock for current device's location data
        data = self.get_data(location)
        self.locks[location].release()  # Release lock

        if data is not None:
            script_data.append(data)

        # Block Logic: If data was collected, execute the script and update data on neighbors and the current device.
        if script_data != []:
            result = script.run(script_data)  # Execute the script

            if not self.neighbours is None:
                for device in self.neighbours:
                    device.locks[location].acquire()  # Acquire lock for neighbor's location data
                    device.set_data(location, result)  # Update neighbor's data
                    device.locks[location].release()  # Release lock

            self.locks[location].acquire()  # Acquire lock for current device's location data
            self.set_data(location, result)  # Update current device's data
            self.locks[location].release()  # Release lock


class DeviceThread(Thread):
    """
    @brief A worker thread responsible for managing timepoints, updating neighbor information,
    and coordinating script execution for a `Device` instance.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread.
        @param device: The `Device` instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        This loop continuously manages timepoint synchronization, updates the device's
        neighbor information, submits scripts for execution to the thread pool, and
        wats for completion signals.
        """
        while True:
            # Block Logic: Clear events to prepare for the new timepoint cycle.
            self.device.script_received.clear()
            self.device.wait_neighbours.clear()

            # Functional Utility: Fetches the latest neighbor information from the supervisor.
            self.device.neighbours = [] # Clear old neighbors
            self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.wait_neighbours.set() # Signal that neighbors are updated

            # Invariant: If neighbors list is None, it indicates a shutdown signal.
            if self.device.neighbours is None:
                # Block Logic: Gracefully shuts down the thread pool before exiting.
                self.device.pool.wait_completion()
                self.device.pool.terminateWorkers()
                self.device.pool.threadJoin()
                return

            # Block Logic: Submit all assigned scripts for the current timepoint to the thread pool for execution.
            for (script, location) in self.device.scripts:
                self.device.pool.add_task(self.device.executeScript, script, location)

            # Block Logic: Wait for scripts for the current timepoint to be assigned and then for their completion.
            self.device.script_received.wait() # Wait for `assign_script` to signal all scripts are assigned
            self.device.pool.wait_completion() # Wait for all scripts in the pool to finish

            # Block Logic: Synchronize all devices in the system at the end of the timepoint.
            for dev in self.device.allDevices:
                dev.barrier.wait() # All devices wait at the barrier


