"""
@file device.py
@brief Implements a device model for a distributed sensor network simulation.

This file defines the `Device` and `DeviceThread` classes, which together model
the behavior of a single node in a sensor network. Devices can execute scripts
on their data and exchange information with their neighbors. The simulation
progresses in synchronized timepoints.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a single device in the sensor network simulation.

    Each device holds its own sensor data, can be assigned scripts to execute,
    and communicates with neighboring devices under the coordination of a
    supervisor. It uses a separate thread for its main execution loop.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's sensor readings,
                                mapping locations to data values.
            supervisor (Supervisor): The central supervisor managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

        # Events for synchronizing the device's main loop.
        self.timepoint_done = Event()
        self.script_received = Event()

        # Synchronization primitives, initialized by the root device (device_id 0).
        self.barrier = None
        self.location_lock = None
        self.lock = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the synchronization primitives for all devices in the simulation.

        This method is called on the root device (device_id 0) to initialize
        and distribute shared locks and barriers to all other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Pre-condition: This method should only be executed by the root device.
        if self.device_id == 0:
            self.lock = Lock()
            self.barrier = ReusableBarrierSem(len(devices))
            self.location_lock = {}
            # Invariant: After this loop, all devices will share the same barrier and locks.
            for device in devices:
                device.location_lock = self.location_lock
                for location in device.sensor_data:
                    self.location_lock[location] = Lock()
                    if device.device_id != 0:
                        device.barrier = self.barrier
                        device.lock = self.lock


    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific location.

        If a script is provided, it is added to the device's script queue.
        If the script is None, it signals that the current timepoint is complete.

        Args:
            script (Script): The script object to be executed.
            location (str): The location context for the script execution.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signifies the end of script assignments for the current timepoint.
            self.timepoint_done.set()
            

    def get_data(self, location):
        """
        Retrieves sensor data for a given location, with thread safety.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            The sensor data at the specified location, or None if the location
            is not in this device's sensor data map.
        """
        with self.lock:
            res = self.sensor_data[location] if location in self.sensor_data else None
        return res

    def set_data(self, location, data):
        """
        Updates sensor data for a given location, with thread safety.

        Args:
            location (str): The location at which to update data.
            data: The new data value to be set.
        """
        with self.lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()

    def run_script(self, script, location, neighbours):
        """
        Executes a script on data gathered from this device and its neighbors.

        The method acquires a lock for the specified location, gathers data,
        runs the script, and then updates the data on all involved devices
        with the script's result.

        Args:
            script (Script): The script to execute.
            location (str): The location context for data gathering and updates.
            neighbours (list): A list of neighboring Device objects.
        """
        # Pre-condition: A lock for the location must be acquired to ensure data consistency.
        self.location_lock[location].acquire()
        script_data = []
        
        # Block Logic: Gathers data from all neighboring devices for the given location.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
            
        # Also gather data from the current device.
        data = self.get_data(location)
        if data is not None:
            script_data.append(data)

        # Invariant: script_data now contains all available sensor readings for 'location'
        # from this device and its neighbors.
        if script_data != []:
            
            result = script.run(script_data)
            
            # Block Logic: Propagates the script's result back to all involved devices.
            for device in neighbours:
                device.set_data(location, result)
            self.set_data(location, result)

        
        self.location_lock[location].release()
        

class DeviceThread(Thread):
    """
    The main execution thread for a Device.

    This thread manages the device's lifecycle, coordinating with the supervisor
    to process scripts at each timepoint of the simulation.
    """
    
    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        
    def run(self):
        """
        The main loop of the device thread.

        The thread waits for a signal to start a timepoint, executes all assigned
        scripts in parallel, and then waits at a barrier for all other devices
        to complete the timepoint before proceeding to the next.
        """
        while True:    
            
            # Block Logic: Fetches the current set of neighbors from the supervisor.
            # If the supervisor returns None, the simulation is over.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Waits until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            
            thread_list = []

            
            # Block Logic: Executes all assigned scripts for the current timepoint.
            # A pool of 8 threads is used to run scripts in parallel.
            for (script, location) in self.device.scripts:
                
                # Invariant: The number of active script-running threads is capped at 8.
                if len(thread_list) < 8:
                    
                    t = Thread(target=self.device.run_script, args=(script, location, neighbours))
                    t.start()
                    thread_list.append(t)
                else:
                    
                    # If the thread pool is full, wait for the oldest thread to complete.
                    out_thread = thread_list.pop(0)
                    out_thread.join()

            
            # Block Logic: Waits for all remaining script threads to complete.
            for thread in thread_list:
                thread.join()
                
            # All devices in the simulation synchronize at this barrier before the next timepoint.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()