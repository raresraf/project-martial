"""
Models a network of interconnected devices that process sensor data in a synchronized, time-stepped simulation.
Each device runs in its own thread, communicates with its neighbors, and executes assigned scripts
to transform data. Synchronization across all devices is maintained using a reusable barrier.
"""

from threading import Event, Thread, Lock
from multiprocessing.dummy import Pool as ThreadPool
from reusablebarrier import ReusableBarrierCond

class Device(object):
    """
    Represents a single device within the simulated network. It manages its own state,
    data, and execution flow, while coordinating with other devices via a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor readings,
                                keyed by location.
            supervisor (Supervisor): An external object that manages the network topology
                                     (i.e., provides neighbor information).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Events to control the flow of the device's execution thread.
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        
        # A dictionary of locks to ensure thread-safe access to sensor data at different locations.
        self.data_locks = {}
        for location in sensor_data:
            self.data_locks[location] = Lock()
            
        # The main execution thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared barrier for synchronization among all devices.
        This method is intended to be called once by a designated root device (ID 0).

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Pre-condition: This method should only be executed by the root device.
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            # Distribute the shared barrier to all other devices.
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device for a specific location.
        Also used to signal the end of a timepoint by passing None.

        Args:
            script (Script): The script object to be executed. If None, it signals
                             that all scripts for the current timepoint have been assigned.
            location (str): The location associated with the script's execution.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of the current simulation timepoint for this device.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Thread-safely retrieves sensor data from a specified location.

        Args:
            location (str): The location from which to retrieve data.

        Returns:
            The sensor data if the location exists, otherwise None.
        """
        if location in self.sensor_data:
            self.data_locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Thread-safely updates the sensor data at a specified location.

        Args:
            location (str): The location to update.
            data: The new data to be stored.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_locks[location].release()

    def shutdown(self):
        """Waits for the device's main execution thread to complete."""
        self.thread.join()

class Helper(object):
    """
    A worker class that manages the execution of scripts for a device using a thread pool.
    It gathers data from neighboring devices, runs the scripts, and distributes the results.
    """
    def __init__(self, device):
        """
        Initializes the Helper.

        Args:
            device (Device): The parent device this helper serves.
        """
        self.device = device
        self.pool = ThreadPool(8)
        self.neighbours = None
        self.scripts = None

    def set_neighbours_and_scripts(self, neighbours, scripts):
        """
        Sets the context for a script execution run.

        Args:
            neighbours (list): A list of neighboring Device objects.
            scripts (list): A list of (script, location) tuples to be executed.
        """
        self.neighbours = neighbours
        self.scripts = scripts

    def script_run(self, (script, location)):
        """
        Executes a single script. It gathers data from the local device and its
        neighbors, runs the script on the aggregated data, and then updates the data
        on all participating devices with the result.

        Args:
            (script, location): A tuple containing the script to run and the target location.
        """
        script_data = []
        # Invariant: Collect data from all neighbors that have the specified location.
        for device in self.neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        
        # Also collect data from the local device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
            
        # Pre-condition: Only run the script if there is data to process.
        if script_data:
            result = script.run(script_data)
            # Post-condition: Distribute the result back to all participating neighbors.
            for device in self.neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)
            # Update the local device's data as well.
            self.device.set_data(location, result)

    def run(self):
        """Executes all assigned scripts asynchronously in the thread pool."""
        self.pool.map_async(self.script_run, self.scripts)

    def close_pool(self):
        """Closes the thread pool and waits for all script executions to complete."""
        self.pool.close()
        self.pool.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device. It orchestrates the device's lifecycle
    through the simulation's timepoints, handling script execution and synchronization.
    """

    def __init__(self, device):
        """
        Initializes the device thread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.helper = None

    def run(self):
        """
        The main control loop for the device. It continuously processes timepoints
        until the simulation ends.
        """
        # Outer loop representing the entire simulation.
        while True:
            # At the beginning of each major step, get the current network topology.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbors signals the end of the simulation.
                break
            self.helper = Helper(self.device)
            
            # Inner loop representing a single timepoint in the simulation.
            while True:
                # Pre-condition: Wait for scripts to be assigned or for the end of the timepoint.
                if (self.device.script_received.is_set() or
                self.device.timepoint_done.is_set()):
                    
                    # Logic: If scripts have been received, execute them.
                    if self.device.script_received.is_set():
                        self.device.script_received.clear()
                        self.helper.set_neighbours_and_scripts(neighbours,
							self.device.scripts)
                        self.helper.run()
                    else:
                        # Logic: If the timepoint is marked as done, exit the inner loop
                        # to proceed to the synchronization barrier.
                        self.device.timepoint_done.clear()
                        self.device.script_received.set()
                        break
            
            # Clean up the helper's resources after the timepoint's script executions are finished.
            self.helper.close_pool()
            # Invariant: All devices must wait at the barrier, ensuring that no device
            # starts the next timepoint until all devices have finished the current one.
            self.device.barrier.wait()