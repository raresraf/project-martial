"""
@file device.py
@brief This module defines the components for a distributed device simulation framework.

It includes classes for:
- Device: Represents a single simulated device, managing its sensor data, scripts, and synchronization.
- DeviceCore: A thread that executes a single script on a device, gathering data from neighbors and updating results.
- DeviceThread: The main thread for each Device, managing its lifecycle, script assignment, and coordination with a supervisor and other devices.
- ReusableBarrierSem: A reusable barrier synchronization primitive (imported from 'barrier' module) used for coordinating multiple threads.

The framework simulates devices that can execute scripts, exchange sensor data with neighbors, and synchronize their operations.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    @brief Represents a single simulated device in the distributed system.

    Manages the device's unique identifier, sensor data, script execution state,
    and interaction with the supervisor and other devices.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings, keyed by data identifier.
        supervisor (object): A reference to the central supervisor managing all devices.
        script_received (Event): An event flag set when a new script is assigned.
        scripts (list): A list to store assigned scripts and their locations.
        timepoint_done (Event): An event flag set when a timepoint's script execution is complete.
        start_event (Event): An event flag used to signal the DeviceThread to start its main loop.
        thread (DeviceThread): The dedicated thread managing this device's operations.
        data_lock (dict): A dictionary of locks, one for each sensor data, to ensure thread-safe data access.
        barrier (ReusableBarrierSem): A synchronization barrier for coordinating with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id (int): The unique identifier for this device.
        @param sensor_data (dict): Initial sensor data for the device.
        @param supervisor (object): The supervisor object managing this device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()

        self.scripts = []
        self.timepoint_done = Event()
        self.start_event = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        self.data_lock = {}

        # Initialize a lock for each sensor data for fine-grained access control.
        for data in sensor_data:
            self.data_lock[data] = Lock()

        self.barrier = None

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return (str): A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device dependencies, specifically the synchronization barrier.
        Ensures all devices share the same barrier instance.
        @param devices (list): A list of all Device instances in the simulation.
        """
        # If the barrier is not yet initialized for this device, create one and assign it to all devices.
        if self.barrier == None:
            self.barrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                dev.barrier = self.barrier

        # Signal the DeviceThread to start its main processing loop.
        self.start_event.set()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific sensor location.
        If script is None, it signals that all scripts for the current timepoint have been assigned.
        @param script (object or None): The script object to execute, or None to signal completion.
        @param location (str): The sensor data location relevant to the script.
        """
        # If a valid script is provided, add it to the list of scripts to be executed.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If script is None, it means no more scripts for this timepoint, so set the timepoint_done event.
            self.timepoint_done.set()

        # Signal that a new script (or completion signal) has been received.
        self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data from a specified location, ensuring thread safety.
        @param location (str): The key corresponding to the desired sensor data.
        @return (any): The sensor data at the given location, or None if the location does not exist.
        """
        # Acquire the specific lock for this data location to ensure exclusive access during read.
        if location in self.sensor_data:
            self.data_lock[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location, ensuring thread safety.
        @param location (str): The key corresponding to the sensor data to be updated.
        @param data (any): The new data value to set.
        """
        # Update the data and release the lock for this location.
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_lock[location].release()

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device, joining its dedicated thread.
        """
        self.thread.join()


class DeviceCore(Thread):
    """
    @brief A dedicated thread for executing a single script on a device.

    This thread is responsible for gathering necessary data from its own device
    and its neighbors, executing a script with that data, and then updating
    the relevant sensor data on both the local device and its neighbors.
    """
    
    def __init__(self, device, location, script, neighbours):
        """
        @brief Initializes a new DeviceCore thread.
        @param device (Device): The local Device instance.
        @param location (str): The sensor data location relevant to the script.
        @param script (object): The script object to execute.
        @param neighbours (list): A list of neighboring Device instances for data exchange.
        """
        Thread.__init__(self, name="Device core %d" % device.device_id)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for the DeviceCore thread.
        It collects data, runs the script, and distributes the results.
        """
        script_data = []
        # Block Logic: Gathers data from neighboring devices.
        # It explicitly avoids requesting data from itself when iterating through neighbors.
        for device in self.neighbours:
            if self.device.device_id != device.device_id:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
        
        # Gathers data from the local device itself.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script if any relevant data was collected.
        if script_data != []:
            # Execute the script with the collected data.
            result = self.script.run(script_data)

            # Block Logic: Updates sensor data on neighboring devices with the script's result.
            for device in self.neighbours:
                if self.device.device_id != device.device_id:
                    device.set_data(self.location, result)
            
            # Updates sensor data on the local device with the script's result.
            self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    @brief The main thread managing the lifecycle and continuous operation of a Device.

    This thread coordinates the device's activities, including synchronizing with
    other devices, fetching neighbors information from the supervisor, and
    dispatching scripts for execution.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread.
        @param device (Device): The Device instance that this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        It waits for a start signal, then continuously synchronizes with other devices,
        retrieves neighbor information, and dispatches assigned scripts using DeviceCore threads.
        """
        # Wait for the device to be set up by the supervisor (e.g., barrier initialized).
        self.device.start_event.wait()

        while True:
            # Wait at the barrier to synchronize with all other devices before starting a new timepoint.
            self.device.barrier.wait()

            # Block Logic: Fetches information about neighboring devices from the supervisor.
            # If the supervisor returns None, it indicates the simulation should terminate.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits until all scripts for the current timepoint have been assigned.
            # This ensures that all scripts are known before starting execution.
            while not self.device.timepoint_done.is_set():
                self.device.script_received.wait()

            # Initialize variables for managing a pool of DeviceCore threads.
            used_cores = 0
            free_core = list(range(8)) # Assuming a maximum of 8 cores/executors per device.
            threads = {} # Dictionary to hold active DeviceCore threads.

            # The commented out section below seems to be part of a more complex
            # thread management strategy, potentially involving dynamic thread allocation
            # or a thread pool. The current implementation proceeds to iterate through
            # assigned scripts and launch DeviceCore threads directly.

            # Block Logic: Iterates through assigned scripts and launches DeviceCore threads for execution.
            # This part appears incomplete or uses a simplified thread management approach.
            for (script, location) in self.device.scripts:
                # This conditional logic for 'used_cores < 8' and reusing 'threads'
                # indicates an attempt to manage a fixed pool of threads (8 cores).
                # However, the `else` branch which reclaims finished threads is commented out.
                # As currently implemented, it will only launch up to 8 threads and then stop processing scripts
                # if more than 8 scripts are assigned before the next timepoint.
                if used_cores < 8:
                    dev_core = DeviceCore(self.device, location, script, neighbours)
                    dev_core.start()
                    threads[free_core.pop()] = dev_core
                    used_cores = used_cores + 1

                # Original code had an 'else' block here, suggesting more elaborate thread pooling logic:
                # else:
                #     for thread in threads:
                #         if not threads[thread].isAlive():
                #             threads[thread].join()
                #             free_core.append(thread)
                #             used_cores = used_cores - 1

            # Wait for all currently active DeviceCore threads to complete their execution.
            for thread in threads:
                threads[thread].join()

            # Reset event flags for the next timepoint.
            self.device.timepoint_done.clear()
            if self.device.script_received.is_set():
                self.device.script_received.clear()
