

"""
@d8f9d7cd-06ed-44b5-8475-266ec196ab61/device.py
@brief Defines core components for simulating a distributed sensor network or device system.
This module provides classes for devices, their operational threads, and a reusable
barrier synchronization mechanism, enabling simulation of concurrent operations
and data exchange across multiple simulated entities. This version introduces
a `DeviceCore` for parallel script execution and a per-data-location locking mechanism.

Domain: Distributed Systems, Concurrency, Simulation.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    @brief Represents a single device in the simulated distributed system.
    Each device manages its own sensor data, processes scripts, and interacts
    with a supervisor to coordinate with other devices. It encapsulates
    device-specific state and behavior, including fine-grained data locking.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data readings
                            keyed by location.
        @param supervisor: A reference to the supervisor object that
                            manages inter-device communication and coordination.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to this device.
        self.script_received = Event()
        # List to store assigned scripts, each paired with its execution location.
        self.scripts = []
        # Event to signal completion of a timepoint's processing for this device.
        self.timepoint_done = Event()
        # Event to signal the start of the device's operational thread.
        self.start_event = Event()
        # The dedicated thread responsible for this device's operations.
        self.thread = DeviceThread(self)
        # Start the operational thread for this device.
        self.thread.start()

        # Dictionary to hold Locks for each sensor data entry, preventing
        # race conditions during data access and ensuring atomicity.
        self.data_lock = {}

        # Block Logic: Initializes a specific lock for each piece of sensor data.
        # Pre-condition: `sensor_data` is a dictionary where keys represent data locations.
        for data in sensor_data:
            self.data_lock[data] = Lock()

        # Barrier for synchronizing all devices at the end of a timepoint.
        self.barrier = None

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        @return A string in the format "Device {device_id}".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures shared resources (barrier) across devices and signals readiness.
        This method ensures that all devices share the same synchronization barrier.
        The first device to call this method initializes the barrier, which is then
        shared among all other devices.
        @param devices: A list of all Device objects in the simulation.
        """
        # Block Logic: Initializes the shared barrier if it hasn't been already.
        # Invariant: After this block, `self.barrier` will reference a `ReusableBarrierSem` instance.
        if self.barrier == None:
            self.barrier = ReusableBarrierSem(len(devices))
            # Propagate the initialized barrier to all other devices.
            for dev in devices:
                dev.barrier = self.barrier

        # Signal that the device is set up and ready to start its operations.
        self.start_event.set()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.
        If a script is provided, it's added to the device's queue. If no script
        is provided (None), it indicates that the current timepoint's assignments
        are complete. In either case, `script_received` is signaled.
        @param script: The script object to be executed, or None to signal completion.
        @param location: The data location pertinent to the script's execution.
        """
        # Block Logic: Appends a new script if provided, otherwise signals timepoint completion.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()

        # Signal that a script assignment has been received, waking up the device's thread.
        self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location, ensuring thread-safe access.
        This method acquires a specific lock for the data location before returning
        the data, preventing concurrent modifications.
        @param location: The location for which to retrieve data.
        @return The sensor data at the given location, or None if the location
                does not exist in the device's sensor_data.
        """
        # Block Logic: Acquires a lock for the specific data location to ensure
        # exclusive access during data retrieval.
        if location in self.sensor_data:
            self.data_lock[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data for a specified location and releases the lock.
        This method assumes a lock for the `location` has already been acquired
        (e.g., by `get_data`) and releases it after updating the data.
        @param location: The location whose data needs to be updated.
        @param data: The new data value to set for the location.
        """
        # Block Logic: Updates sensor data if the location exists, then releases the associated lock.
        # Pre-condition: The lock for `location` must have been acquired prior to calling this method.
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_lock[location].release()

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread.
        Ensures proper termination by waiting for the device's thread to complete its execution.
        """
        self.thread.join()


class DeviceCore(Thread):
    """
    @brief A dedicated thread for executing a single script within a device.
    This class handles the execution of a script at a specific location,
    including collecting data from neighbors, running the script, and
    propagating the results back to the device and its neighbors.
    """
    def __init__(self, device, location, script, neighbours):
        """
        @brief Initializes a new DeviceCore instance.
        @param device: The parent Device object.
        @param location: The data location pertinent to the script.
        @param script: The script object to execute.
        @param neighbours: A list of neighboring Device objects.
        """
        # Functional Utility: Initializes the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device core %d" % device.device_id)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for the DeviceCore thread.
        Collects data, runs the script, and propagates the results,
        respecting data locks.
        """
        script_data = []
        
        # Block Logic: Collects sensor data from neighboring devices for the specified location.
        # Invariant: Each neighbor's data is retrieved via `get_data`, which handles its own locking.
        for device in self.neighbours:
            # Inline: Ensures data is not collected from itself if the device is in the neighbors list.
            if self.device.device_id != device.device_id:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
        
        # Block Logic: Collects sensor data from the current device itself for the specified location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script only if there is any collected data.
        if script_data != []:
            # Functional Utility: Executes the assigned script with the aggregated data.
            result = self.script.run(script_data)

            # Block Logic: Propagates the script's result to neighboring devices.
            # Pre-condition: Each neighbor's `set_data` method is called, assuming
            # the corresponding lock for `self.location` was acquired during their `get_data` call.
            for device in self.neighbours:
                if self.device.device_id != device.device_id:
                    device.set_data(self.location, result)
            
            # Block Logic: Updates the current device's sensor data with the script's result.
            # Pre-condition: The lock for `self.location` was acquired during `self.device.get_data` call.
            self.device.set_data(self.location, result)

class DeviceThread(Thread):
    """
    @brief Manages the execution lifecycle of a single Device.
    This thread is responsible for coordinating with the supervisor,
    managing a pool of `DeviceCore` threads for parallel script execution,
    and synchronizing with other device threads using a barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread for a given device.
        @param device: The Device object that this thread will manage.
        """
        # Functional Utility: Initializes the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the device thread.
        This loop waits for device setup, synchronizes with other devices via
        a barrier, manages the execution of scripts using `DeviceCore` threads,
        and signals completion for each timepoint.
        """
        # Block Logic: Waits for the device to be fully set up before starting operations.
        self.device.start_event.wait()

        # Block Logic: Main simulation loop, continuously processing timepoints.
        while True:
            # Pre-condition: All devices must reach this barrier before proceeding
            # to the next timepoint's operations.
            self.device.barrier.wait()

            # Block Logic: Retrieves current neighbors from the supervisor.
            # Invariant: `neighbours` is either a list of Device objects or None (signaling termination).
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits until the supervisor has finished assigning all scripts
            # for the current timepoint and `timepoint_done` is signaled.
            while not self.device.timepoint_done.is_set():
                # Waits for new scripts to be received to avoid busy-waiting.
                self.device.script_received.wait()

            # Functional Utility: Manages a fixed-size pool of `DeviceCore` threads
            # to simulate parallel script execution within the device.
            used_cores = 0
            free_core = list(range(8)) # Represents available processing cores (max 8).
            threads = {} # Dictionary to keep track of active DeviceCore threads.

            # Block Logic: Iterates through assigned scripts and dispatches them
            # to available DeviceCore threads for parallel execution.
            # Invariant: Scripts are processed using a limited number of "cores".
            for (script, location) in self.device.scripts:
                # Pre-condition: Checks if there is an available processing core.
                if used_cores < 8:
                    dev_core = DeviceCore(self.device, location, script, neighbours)
                    dev_core.start()
                    threads[free_core.pop()] = dev_core
                    used_cores = used_cores + 1
                else:
                    # Block Logic: If all cores are in use, waits for a core to become free
                    # by joining a completed thread.
                    for thread in threads:
                        if not threads[thread].isAlive():
                            threads[thread].join()
                            free_core.append(thread)
                            used_cores = used_cores - 1

            # Block Logic: Ensures all active DeviceCore threads complete their execution.
            for thread in threads:
                threads[thread].join()

            # Reset events for the next timepoint.
            self.device.timepoint_done.clear()
            if self.device.script_received.is_set():
                self.device.script_received.clear()
