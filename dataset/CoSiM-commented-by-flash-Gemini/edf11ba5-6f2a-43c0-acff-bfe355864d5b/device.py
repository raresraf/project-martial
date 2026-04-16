

"""
@edf11ba5-6f2a-43c0-acff-bfe355864d5b/device.py
@brief Implements a simulated sensor device for a distributed sensor network, including its operational thread and worker threads for script execution.

This module defines the core components for a simulated device in a distributed sensor network.
The `Device` class represents an individual sensor device, managing its own sensor data,
a queue of scripts to execute, and synchronization primitives. The `DeviceThread` class
provides a dedicated execution context for each `Device`, continuously processing scripts
assigned to it. The `Worker` class encapsulates the execution of a single script,
handling data aggregation from neighboring devices and the local device,
and ensuring data consistency through location-specific locks.

Domain: Distributed Systems, Concurrency, Simulation, Sensor Networks.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    @brief Represents a simulated sensor device in a distributed sensor network.

    This class manages the device's unique identifier, sensor data,
    and a reference to its supervisor. It handles the receipt and
    storage of scripts for execution, along with synchronization
    mechanisms such as an `Event` for script reception, a list
    to store incoming scripts, and a list of `Lock` objects for
    managing concurrent access to sensor data locations. Each device
    also has a dedicated thread (`DeviceThread`) for operation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        :param device_id: A unique identifier for the device.
        :param sensor_data: A dictionary representing the sensor data
                            collected by this device, where keys are
                            sensor locations and values are data points.
        :param supervisor: A reference to the supervisor object that
                           manages the network of devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Functional Utility: Signals when a new script has been assigned to the device.
        self.script_received = Event()

        # Functional Utility: Stores a list of (script, location) tuples to be executed.
        self.scripts = []

        # Functional Utility: Stores Lock objects, each protecting a specific sensor data location.
        self.lock_locations = []

        # Functional Utility: A reusable barrier for synchronizing all devices before and after script execution cycles.
        self.barrier = ReusableBarrierSem(0)

        # Functional Utility: The dedicated thread for this device's operations.
        self.thread = DeviceThread(self)

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        :return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources and starts threads for all devices.

        This method is typically called by a supervisor or a central entity
        to initialize all devices in the network. It calculates the total
        number of unique sensor locations across all devices, creates a
        `Lock` for each unique location, and then distributes these shared
        locks and a common `ReusableBarrierSem` to all devices. Finally,
        it starts the dedicated thread for each device.

        :param devices: A list of all Device instances in the network.
        """
        # Block Logic: Initializes a shared barrier for all devices, setting its initial capacity
        # to the total number of devices to ensure all devices synchronize.
        barrier = ReusableBarrierSem(len(devices))

        # Block Logic: Determines the maximum location ID to correctly size the shared lock array.
        if self.device_id == 0:  # Ensures this block runs only once, typically by the first device.
            nr_locations = 0

            # Precondition: `devices` is a list of Device objects.
            # Invariant: `nr_locations` tracks the highest location ID encountered.
            for i in range(len(devices)):
                for location in devices[i].sensor_data.keys():
                    if location > nr_locations:
                        nr_locations = location
            # Inline: Increments to account for zero-based indexing and provide a count.
            nr_locations += 1

            # Block Logic: Creates a unique lock for each sensor location.
            # Invariant: `self.lock_locations` will contain `nr_locations` Lock objects.
            for i in range(nr_locations):
                lock_location = Lock()
                self.lock_locations.append(lock_location)

            # Block Logic: Distributes the shared barrier and location locks to all devices.
            for i in range(len(devices)):
                # Functional Utility: Assigns the common synchronization barrier to each device.
                devices[i].barrier = barrier

                # Functional Utility: Provides each device with access to the global set of location locks.
                for j in range(nr_locations):
                    devices[i].lock_locations.append(self.lock_locations[j])

                # Functional Utility: Starts the dedicated thread for each device.
                devices[i].thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.

        If a script is provided, it is added to the device's script queue.
        If no script is provided (i.e., `script` is None), it signals
        that script assignment is complete for the current cycle, allowing
        the device's thread to proceed.

        :param script: The script object to be executed, or None to signal completion.
        :param location: The sensor location associated with the script.
        """
        if script is not None:
            # Functional Utility: Appends the new script and its target location to the processing queue.
            self.scripts.append((script, location))
        else:
            # Functional Utility: Sets the event to notify the DeviceThread that all scripts for the current cycle have been assigned.
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.

        :param location: The sensor location to retrieve data from.
        :return: The sensor data at the given location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.

        :param location: The sensor location to update.
        :param data: The new data value to set at the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its dedicated thread.

        Ensures that the `DeviceThread` completes its execution before the program exits.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief A dedicated thread for a Device to manage its operational lifecycle.

    Each `Device` instance has a `DeviceThread` that continuously
    monitors for assigned scripts, creates `Worker` threads to
    execute these scripts, and synchronizes with other `DeviceThread`s
    using a shared barrier. This thread ensures that script processing
    is handled asynchronously and in parallel with other devices.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        :param device: The Device instance this thread is associated with.
        """
        # Functional Utility: Calls the base Thread class constructor, setting a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        This method continuously:
        1. Retrieves neighbors from the supervisor. If no neighbors, it breaks the loop.
        2. Waits for scripts to be assigned by the supervisor.
        3. Creates and starts `Worker` threads for each assigned script.
        4. Waits for all `Worker` threads to complete.
        5. Clears the list of processed scripts.
        6. Synchronizes with other `DeviceThread`s using a shared barrier
           before starting the next cycle of script processing.
        """
        workers = []

        # Block Logic: Main loop for continuous script processing.
        # Precondition: The device is active and connected to a supervisor.
        # Invariant: The device continuously checks for new scripts, processes them, and synchronizes.
        while True:
            # Functional Utility: Obtains the list of neighboring devices from the supervisor for data exchange.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Block Logic: Termination condition if no neighbors are found, indicating the simulation might be ending.
                break

            # Functional Utility: Pauses execution until new scripts are assigned by the supervisor.
            self.device.script_received.wait()
            # Functional Utility: Resets the event for the next script assignment cycle.
            self.device.script_received.clear()

            # Block Logic: Creates a Worker thread for each assigned script.
            # Invariant: `workers` list contains Worker instances for all current scripts.
            for (script, location) in self.device.scripts:
                workers.append(Worker(self.device, script,
                                        location, neighbours))

            # Block Logic: Starts all worker threads in parallel.
            for i in range(len(workers)):
                workers[i].start()

            # Block Logic: Waits for all worker threads to complete their execution.
            # Precondition: All worker threads have been started.
            # Invariant: All scripts for the current cycle are processed before proceeding.
            for i in range(len(workers)):
                workers[i].join()

            # Functional Utility: Resets the list of worker threads for the next cycle.
            workers = []

            # Functional Utility: Resets the scripts list after they have been processed.
            self.device.scripts = []

            # Functional Utility: Synchronizes all DeviceThreads, ensuring all devices complete a processing cycle
            # before moving to the next. This acts as a global checkpoint.
            self.device.barrier.wait()


class Worker(Thread):
    """
    @brief A worker thread responsible for executing a single script for a device at a specific location.

    This class extends `threading.Thread` to enable concurrent execution
    of scripts. Each `Worker` thread acquires a lock for its designated
    sensor location, aggregates data from the local device and its neighbors,
    executes the assigned script with this data, updates the sensor data
    (locally and on neighbors), and finally releases the lock.
    """

    def __init__(self, device, script, location, neighbours):
        """
        @brief Initializes a new Worker instance.

        :param device: The Device instance for which this worker is processing a script.
        :param script: The script object to be executed.
        :param location: The sensor location pertinent to this script execution.
        :param neighbours: A list of neighboring Device instances for data exchange.
        """
        # Functional Utility: Calls the base Thread class constructor, setting a descriptive name.
        Thread.__init__(self, name="Worker Thread for Device %d at location %d" % (device.device_id, location))
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def solve_script(self, script, location, neighbours):
        """
        @brief Executes the assigned script, handling data aggregation and updates with concurrency control.

        This method performs the core logic of the worker:
        1. Acquires a lock for the specific `location` to prevent race conditions.
        2. Gathers relevant sensor data from all neighboring devices and the local device
           for the given `location`.
        3. Executes the `script` using the aggregated data.
        4. Updates the sensor data on all neighboring devices and the local device
           with the result of the script execution.
        5. Releases the `location` lock.

        :param script: The script object to run.
        :param location: The sensor location being processed.
        :param neighbours: The list of neighboring Device instances.
        """
        # Functional Utility: Acquires a lock for the specific sensor location to ensure exclusive access
        # during data aggregation and update, preventing race conditions.
        self.device.lock_locations[location].acquire()

        script_data = []

        # Block Logic: Aggregates sensor data from all neighboring devices for the current location.
        # Precondition: `neighbours` is a list of Device objects.
        # Invariant: `script_data` will contain valid sensor data from neighbors if available.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Functional Utility: Aggregates sensor data from the local device for the current location.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data: # Block Logic: Proceeds with script execution only if there is data to process.
            # Functional Utility: Executes the script with the aggregated sensor data.
            result = script.run(script_data)

            # Block Logic: Updates the sensor data on all neighboring devices with the script's result.
            # Invariant: Neighboring devices' data at `location` will reflect the script's outcome.
            for device in neighbours:
                device.set_data(location, result)

            # Functional Utility: Updates the local device's sensor data with the script's result.
            self.device.set_data(location, result)

        # Functional Utility: Releases the lock for the sensor location, allowing other threads to access it.
        self.device.lock_locations[location].release()

    def run(self):
        """
        @brief The main execution method for the Worker thread.

        Simply calls `solve_script` to execute the assigned script.
        """
        self.solve_script(self.script, self.location, self.neighbours)
