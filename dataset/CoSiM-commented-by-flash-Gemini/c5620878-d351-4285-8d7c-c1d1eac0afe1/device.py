

"""
This module defines a multi-threaded device simulation framework for collaborative
sensor data processing. It employs a device-centric architecture with dedicated threads
for managing device logic and executing scripts,
utilizing barriers and fine-grained locking for synchronization and data consistency.
"""

from threading import Event, Thread, Lock
import barrier # Assumed to contain ReusableBarrierSem

class Device(object):
    """
    Represents a simulated device within a distributed processing network.
    Manages its sensor data, assigned scripts, and coordinates with other devices
    and a central supervisor through threading and synchronization primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding sensor data, keyed by location.
            supervisor (Supervisor): A reference to the central supervisor managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are assigned.
        self.scripts = [] # List to store assigned scripts, each with its target location.
        self.timepoint_done = Event() # Event to signal when all scripts for a timepoint are processed.
        self.thread = DeviceThread(self) # Dedicated thread for this device's operational logic.
        self.thread.start()
        self.locks = None # Shared dictionary of locks, keyed by sensor data location.
        self.barrier = None # Shared ReusableBarrier for synchronizing devices.

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the shared synchronization primitives and data locks across all devices.
        This method is typically called once by a designated master device.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        # Block Logic: Initializes shared barrier and location-specific locks on the master device,
        # then propagates these references to all other devices.
        # Invariant: All devices share the same barrier and set of location locks for data consistency.
        devices[0].barrier = barrier.ReusableBarrierSem(len(devices))
        devices[0].locks = {}
        list_index = list(range(len(devices)))
        # Block Logic: Iterates through devices (excluding the master) to set their shared barrier and locks.
        # Invariant: Each device correctly references the shared barrier and locks.
        for i in list_index[1:len(devices)]:
            devices[i].barrier = devices[0].barrier
            devices[i].locks = devices[0].locks

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific sensor data location.
        If script is None, it signals the completion of script assignments for the current timepoint.

        Args:
            script (Script): The script object to be executed.
            location (int): The identifier for the sensor data location the script targets.
        """
        # Block Logic: Appends the script and its target location to the device's script list.
        # If a script is provided, it sets an event to indicate that scripts are available.
        # If script is None, it signals that all scripts for the current timepoint have been assigned.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set() # Signals that all scripts for the current timepoint have been assigned.

    def get_data(self, location):
        """
        Retrieves sensor data from a specified location on this device.

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or None if the location does not exist.
        """
        # Functional Utility: Safely retrieves sensor data if the location exists.
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data at a specified location on this device.

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new data value to set.
        """
        # Functional Utility: Safely updates sensor data if the location exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the graceful shutdown of the device's operational thread.
        """
        # Functional Utility: Waits for the device's thread to complete its execution before proceeding.
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the operational lifecycle of a single simulated device.
    It orchestrates script execution, synchronizes with other devices using a barrier,
    and manages the simulation timepoints.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread.

        Args:
            device (Device): The Device instance this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def thread_script(self, neighbours, script, location):
        """
        Executes a single script, handling data collection from neighbors and local device,
        and propagating results. This function acts as the target for individual script threads.

        Args:
            neighbours (list): A list of neighboring Device instances.
            script (Script): The script object to be executed.
            location (int): The identifier for the sensor data location.
        """
        # Block Logic: Acquires a location-specific lock to ensure exclusive access for script execution.
        # Invariant: Data at 'location' is processed atomically.
        if location not in self.device.locks:
            self.device.locks[location] = Lock()

        self.device.locks[location].acquire()

        script_data = []

        # Block Logic: Collects relevant sensor data from neighboring devices.
        # Invariant: All available data for the specified location from neighbors is gathered.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Block Logic: Collects sensor data from the current device.
        # Invariant: The current device's relevant sensor data is gathered.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script if sensor data is available and propagates results.
        # Invariant: If a script runs, its output is used to update relevant sensor data.
        if script_data != []:
            # Functional Utility: Executes the assigned script with the collected data.
            result = script.run(script_data)

            # Block Logic: Propagates the script's result to the relevant sensor locations on neighboring devices.
            # Invariant: Neighboring devices' sensor data is updated consistently with the script's output.
            for device in neighbours:
                device.set_data(location, result)

            # Functional Utility: Updates the current device's sensor data with the script's result.
            self.device.set_data(location, result)

        # Functional Utility: Releases the lock on the sensor data location.
        self.device.locks[location].release()

    def run(self):
        """
        The main execution loop for the device thread.
        It continuously retrieves neighbors, waits for scripts, dispatches them to ScriptThreads,
        and synchronizes with other devices at timepoint boundaries.
        """
        # Block Logic: Main operational loop for the device thread, running until the simulation ends.
        # Invariant: The device continuously processes timepoints, executes scripts, and synchronizes.
        while True:
            # Functional Utility: Retrieves information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Checks if there are no more neighbors, signaling the end of the simulation.
            # If so, it terminates the thread.
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            threads_script = []
            # Block Logic: Iterates through the device's assigned scripts, creating a new thread for each.
            # Invariant: Each script is prepared for concurrent execution in its own thread.
            for (script, location) in self.device.scripts:
                # Functional Utility: Creates a new thread to execute the script concurrently.
                thread = Thread(target=self.thread_script,
                    args=(neighbours, script, location))
                thread.start()
                threads_script.append(thread)

            # Block Logic: Waits for all script execution threads to complete for the current timepoint.
            # Invariant: All scripts assigned for the timepoint are fully processed before proceeding.
            for j in xrange(len(threads_script)):
                threads_script[j].join()

            # Functional Utility: Waits at the shared barrier to synchronize with all other devices
            # after all scripts for the current timepoint have been executed.
            self.device.barrier.wait()
            self.device.timepoint_done.clear() # Redundant clear, already cleared before wait
