


"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `Device`: Represents a single device, managing its sensor data, communication with a supervisor,
  and orchestrating multi-threaded script execution.
- `DeviceThread`: The main thread for a `Device`, which utilizes a `ThreadPoolExecutor` to
  run scripts concurrently, manages neighbor information, and handles timepoint synchronization.

The system features concurrent execution of scripts within each device using `ThreadPoolExecutor`,
and inter-device synchronization via a custom `ReusableBarrierCond` imported from the `barrier` module.
It also uses `threading.Event` for signaling and `threading.Lock` for protecting location-specific data.
"""

from threading import Event, Thread, Lock
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from barrier import ReusableBarrierCond

class Device(object):
    """
    Represents a single device within a simulated distributed environment.
    Each device manages its own sensor data, communicates with a supervisor,
    and orchestrates multi-threaded script execution. It utilizes per-location
    locks for data consistency and participates in global synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for the device.
            supervisor (object): A reference to a supervisor object for inter-device communication.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.script_received = Event() # Event to signal when new scripts are assigned.
        self.scripts = [] # List to hold assigned scripts (tuples of (script, location)).
        self.timepoint_done = Event() # Event to signal that the current timepoint's processing is complete.
        self.barrier = None # Placeholder for the global ReusableBarrierCond, set in setup_devices.
        self.locks = {} # Dictionary of `Lock` objects, one for each data location in sensor_data.
        # Inline: Initialize a unique `Lock` for each location where the device holds sensor data.
        for location in sensor_data:
            self.locks[location] = Lock()

        self.thread = DeviceThread(self) # The main orchestrating thread for this device.
        self.thread.start() # Start the DeviceThread.

    def __str__(self):
        """
        Returns a string representation of the device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the global barrier for all devices in the simulation.
        This method is designed to be called only by the device with `device_id == 0`.
        It creates a single `ReusableBarrierCond` for inter-device synchronization
        and distributes it to all other devices.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        if self.device_id == 0:
            # Inline: Creates a global ReusableBarrierCond for synchronization among all DeviceThreads.
            self.barrier = ReusableBarrierCond(len(devices))
            # Inline: Distributes the created global barrier to all other devices.
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals timepoint completion.
        If a script is provided, it's appended to the device's internal script list
        and `script_received` event is set. If `script` is None, `timepoint_done`
        event is set.

        Args:
            script (object): The script object to be executed, or None to signal timepoint completion.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
            self.script_received.set() # Signal that a new script has been received.
        else:
            self.timepoint_done.set() # If script is None, signal that script assignments for the timepoint are done.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's `sensor_data` dictionary.

        Args:
            location (int): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def acquire_lock(self, location):
        """
        Acquires the lock associated with a specific data location on this device.
        This ensures exclusive access to the `sensor_data` at that location.

        Args:
            location (int): The location identifier for which to acquire the lock.
        """
        if location in self.sensor_data: # Only acquire if the location is managed by this device.
            self.locks[location].acquire()

    def set_data(self, location, data):
        """
        Sets sensor data for a given location in this device's `sensor_data` dictionary.
        The data is updated only if the location exists in the `sensor_data`.

        Args:
            location (int): The location identifier for which to set data.
            data (any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def release_lock(self, location):
        """
        Releases the lock associated with a specific data location on this device.

        Args:
            location (int): The location identifier for which to release the lock.
        """
        if location in self.sensor_data: # Only release if the location is managed by this device.
            self.locks[location].release()

    def shutdown(self):
        """
        Initiates the shutdown process for the device by waiting for its main `DeviceThread` to complete.
        This implicitly triggers the shutdown of the associated `ThreadPoolExecutor`.
        """
        self.thread.join() # Wait for the DeviceThread to finish its execution.


class DeviceThread(Thread):
    """
    The main orchestrating thread for a `Device`.
    This thread manages a `ThreadPoolExecutor` to run scripts concurrently,
    fetches neighbor information from the supervisor, and handles timepoint
    synchronization for the device.
    """

    def __init__(self, device):
        """
        Initializes a `DeviceThread` instance.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=8) # Initialize a thread pool with 8 worker threads.

    def run_script(self, script, location, neighbours):
        """
        Executes a given script at a specific location, collecting necessary data
        from the current device and its neighbors, and then updates their sensor data
        with the script's result. This method is designed to be run by a worker
        thread from the `ThreadPoolExecutor`.

        Args:
            script (object): The script object to be executed.
            location (int): The data location identifier to which the script applies.
            neighbours (list): A list of Device objects representing neighboring devices.
        """
        script_data = [] # List to collect input data for the script.

        # Block Logic: Collect data from all neighboring devices (excluding self) at the specified location.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                # Inline: Acquire the neighbor's location-specific lock before accessing its data.
                device.acquire_lock(location)

                data = device.get_data(location) # Get data from the neighbor.
                if data is not None:
                    script_data.append(data)
        
        # Block Logic: Collect data from this device's own sensor data.
        self.device.acquire_lock(location) # Acquire this device's location-specific lock.

        data = self.device.get_data(location) # Get data from this device.
        if data is not None:
            script_data.append(data)

        # Block Logic: If input data is available, execute the script and update device data.
        if script_data != []:
            # Inline: Execute the script's `run` method with the collected data.
            result = script.run(script_data)

            # Block Logic: Update sensor data for all involved devices (neighbors and self) with the result.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result) # Update neighbor's data.

                    device.release_lock(location) # Release neighbor's location-specific lock.

            self.device.set_data(location, result) # Update this device's own data.

            self.device.release_lock(location) # Release this device's location-specific lock.

    def run(self):
        """
        The main execution loop for the `DeviceThread`.
        It continuously fetches neighbor data, manages script execution via the
        `ThreadPoolExecutor`, and synchronizes across timepoints and devices.
        """
        while True:
            # Block Logic: Fetch neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Inline: If `neighbours` is None, it signals termination for the device.
            if neighbours is None:
                break # Exit the main loop, initiating the shutdown sequence.

            futures = [] # List to hold Future objects returned by `executor.submit`.

            # Block Logic: Inner loop to manage script submission and completion for a timepoint.
            while True:
                # Block Logic: Wait for either new scripts to be received or the timepoint to be done.
                if self.device.script_received.isSet() or self.device.timepoint_done.wait():
                    # Inline: If `script_received` is set, new scripts are available for processing.
                    if self.device.script_received.isSet():
                        self.device.script_received.clear() # Clear the event.

                        # Block Logic: Submit all currently assigned scripts to the ThreadPoolExecutor.
                        for (script, location) in self.device.scripts:
                            futures.append(self.executor.submit(self.run_script, script,
                                                                location, neighbours))

                    # Inline: If `timepoint_done` is set, all scripts for this timepoint have been submitted.
                    else:
                        # Block Logic: Wait for all submitted scripts (futures) to complete.
                        wait(futures, timeout=None, return_when=ALL_COMPLETED)
                        self.device.timepoint_done.clear() # Clear the timepoint done event for the next cycle.
                        self.device.script_received.set() # Set script_received to break out of the inner while True.
                        break # Exit the inner loop, all scripts for timepoint processed.

            # Block Logic: Synchronize with other devices at the global barrier.
            # This ensures all devices have completed their script processing for the timepoint.
            self.device.barrier.wait()

        # Block Logic: When the main loop breaks (device shutdown), gracefully shutdown the executor.
        self.executor.shutdown()
