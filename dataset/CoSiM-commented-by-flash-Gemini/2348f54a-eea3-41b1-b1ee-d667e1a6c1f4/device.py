

"""
This module implements a multi-threaded simulation framework for devices
that process sensor data collaboratively. It features a device-centric architecture
with dedicated threads for managing device logic and executing scripts,
utilizing barriers and fine-grained locking for synchronization and data consistency.
"""

from threading import Event, Thread, Lock
import barrier # Assumed to contain ReusableBarrierCond

class Device(object):
    """
    Represents a simulated device within a collaborative processing network.
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
        self.scripts_received = Event() # Event to signal when new scripts are assigned.
        self.scripts = [] # List to store assigned scripts, each with its target location.
        self.thread = DeviceThread(self) # Dedicated thread for this device's operational logic.
        self.data_lock = Lock() # Lock to protect access to this device's sensor_data.
        self.list_locks = {} # Dictionary to store location-specific locks for shared data access.
        self.barrier = None # Reference to the shared ReusableBarrier for synchronization.
        self.devices = None # List of all devices in the simulation (set only for master device).

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
        self.devices = devices

        # Block Logic: Initializes shared barrier and location-specific locks if this is the master device.
        # Otherwise, it retrieves references to these shared objects from the master device.
        # Invariant: All devices correctly share the same barrier and set of location locks.
        if self.device_id == self.devices[0].device_id:
            # Initializes a reusable barrier for synchronizing all devices.
            self.barrier = barrier.ReusableBarrierCond(len(self.devices))
            # Block Logic: Initializes a lock for each unique sensor data location across all devices.
            # These locks prevent race conditions during concurrent updates to shared sensor data.
            for dev in self.devices:
                for location in dev.sensor_data:
                    self.list_locks[location] = Lock()
        else:
            # Retrieves the shared barrier from the master device.
            self.barrier = devices[0].get_barrier()
            # Retrieves the shared list of location-specific locks from the master device.
            self.list_locks = devices[0].get_list_locks()
        
        # Functional Utility: Starts the dedicated thread for this device's operational logic.
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific sensor data location.
        If script is None, it signals the completion of script assignments for the current timepoint.

        Args:
            script (Script): The script object to be executed.
            location (int): The identifier for the sensor data location the script targets.
        """
        # Block Logic: Appends the script and its target location to the device's script queue.
        # If no script is provided, it signals that script assignment is complete for the current timepoint.
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_received.set() # Signals that all scripts for the current timepoint have been received.


    def get_barrier(self):
        """
        Returns the shared barrier instance associated with this device.

        Returns:
            ReusableBarrierCond: The shared barrier object.
        """
        return self.barrier

    def get_list_locks(self):
        """
        Returns the dictionary of shared location-specific locks.

        Returns:
            dict: A dictionary where keys are locations and values are Lock objects.
        """
        return self.list_locks

    def get_data(self, location):
        """
        Retrieves sensor data from a specified location on this device, protected by a data lock.

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or None if the location does not exist.
        """
        # Block Logic: Acquires the device's data lock to safely read sensor data.
        # Invariant: Sensor data is read consistently without race conditions.
        with self.data_lock:
            if location in self.sensor_data:
                data = self.sensor_data[location]
            else:
                data = None
        return data

    def set_data(self, location, data):
        """
        Sets sensor data at a specified location on this device, protected by a data lock.

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new data value to set.
        """
        # Block Logic: Acquires the device's data lock to safely write sensor data.
        # Invariant: Sensor data is written consistently without race conditions.
        with self.data_lock:
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

            # Functional Utility: Waits for scripts to be assigned for the current timepoint, then clears the event.
            self.device.scripts_received.wait()
            self.device.scripts_received.clear()

            threads = []
            # Block Logic: Iterates through assigned scripts, creating and managing ScriptThread execution.
            # Scripts are processed in batches of 8 to limit concurrent execution.
            # Invariant: All scripts for the current timepoint are eventually dispatched and executed.
            for (script, location) in self.device.scripts:
                threads.append(
                    ScriptThread(self.device, script, location, neighbours))
                # Block Logic: If 8 ScriptThreads are ready, start them and wait for their completion
                # before clearing the list and continuing with the next batch.
                if len(threads) == 8:
                    for thr in threads:
                        thr.start()
                    for thr in threads:
                        thr.join()
                    threads = []
            
            # Block Logic: After processing all scripts in batches, handles any remaining ScriptThreads.
            # Invariant: All remaining scripts are started and their completion is awaited.
            for thr in threads:
                thr.start()
            for thr in threads:
                thr.join()

            # Functional Utility: Waits at the shared barrier to synchronize with all other devices
            # after all scripts for the current timepoint have been executed.
            self.device.barrier.wait()



class ScriptThread(Thread):
    """
    A dedicated thread for executing a single script at a specific sensor data location.
    It handles data retrieval from neighbors and the local device, script execution,
    and propagation of results, ensuring thread-safe data access.
    """

    def __init__(self, device, script, location, neighbours):
        """
        Initializes a new ScriptThread.

        Args:
            device (Device): The local Device instance.
            script (Script): The script object to be executed.
            location (int): The identifier for the sensor data location.
            neighbours (list): A list of neighboring Device instances.
        """
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        The main execution logic for the script thread.
        It acquires a location-specific lock, collects data, executes the script,
        updates data on the local and neighboring devices, and then releases the lock.
        """
        # Functional Utility: Acquires a lock for the specific sensor data location to ensure exclusive access.
        # This prevents race conditions when reading and writing to shared data.
        self.device.list_locks[self.location].acquire()

        script_data = []

        # Block Logic: Collects relevant sensor data from neighboring devices.
        # Invariant: All available data for the specified location from neighbors is gathered.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Collects sensor data from the current device.
        # Invariant: The current device's relevant sensor data is gathered.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script if sensor data is available and propagates results.
        # Invariant: If a script runs, its output is used to update relevant sensor data.
        if script_data != []:
            # Functional Utility: Executes the assigned script with the collected data.
            result = self.script.run(script_data)

            # Block Logic: Propagates the script's result to the relevant sensor locations on neighboring devices.
            # Invariant: Neighboring devices' sensor data is updated consistently with the script's output.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Functional Utility: Updates the current device's sensor data with the script's result.
            self.device.set_data(self.location, result)

        # Functional Utility: Releases the lock on the sensor data location, allowing other threads to access it.
        self.device.list_locks[self.location].release()
