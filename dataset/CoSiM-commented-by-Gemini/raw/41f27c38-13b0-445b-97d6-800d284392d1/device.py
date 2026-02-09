"""
@file device.py
@brief Implements a device model for a distributed simulation using a thread pool.

This file defines a `Device` class that processes sensor data by executing scripts.
It uses a `ThreadPoolExecutor` for concurrent script execution and employs
location-specific semaphores for fine-grained locking of sensor data.
Synchronization across devices is achieved with a reusable barrier.
"""

from threading import Event, Thread, Semaphore
from concurrent.futures import wait, ALL_COMPLETED, ThreadPoolExecutor
from barrier import ReusableBarrier

class Device(object):
    """
    Represents a single device in a simulated network.

    Each device manages its own sensor data and uses a thread pool to execute
    assigned scripts. It coordinates with other devices through a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): A dictionary mapping locations to sensor values.
            supervisor: The central supervisor managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = ReusableBarrier(1) # Initially a barrier for one.
        self.thread = DeviceThread(self)
        # A dictionary of semaphores, one for each sensor data location, for fine-grained locking.
        self.location_sems = {location : Semaphore(1) for location in sensor_data}
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for all devices in the simulation.

        This method should be called on the root device (device_id == 0) to
        create and distribute the main synchronization barrier.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: The root device creates a barrier for all participating devices.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            # Invariant: After this loop, all devices will share the same barrier instance.
            for dev in devices:
                dev.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be run at a specific location.

        Args:
            script: The script object to execute.
            location: The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location, acquiring a location-specific lock.

        Args:
            location: The location identifier to fetch data from.

        Returns:
            The data at the location, or None if not available. The caller is
            responsible for ensuring `set_data` is eventually called to release the lock.
        """
        if location in self.sensor_data:
            self.location_sems[location].acquire()
            data = self.sensor_data[location]
        else:
            data = None
        return data

    def set_data(self, location, data):
        """
        Updates sensor data for a location and releases its specific lock.

        Args:
            location: The location identifier to update.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_sems[location].release()

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The execution thread for a Device, using a ThreadPoolExecutor.

    This thread manages the device's lifecycle, submitting script execution tasks
    to a thread pool and synchronizing timepoints with other devices.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPoolExecutor(8)
        self.neighbours = []

    def gather_info(self, location):
        """
        Gathers sensor data for a location from this device and its neighbors.
        
        This function is intended to be called by a worker thread from the pool.

        Args:
            location (str): The location to gather data for.

        Returns:
            A list of data from all devices that have data for the location.
        """
        script_data = []
        for device in self.neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        return script_data

    def spread_info(self, result, location):
        """
        Distributes the result of a script execution back to the neighbors and self.

        This function releases the locks acquired by `gather_info`.

        Args:
            result: The result from a script execution.
            location (str): The location context for the data update.
        """
        for device in self.neighbours:
            if device.device_id != self.device.device_id:
                device.set_data(location, result)
        self.device.set_data(location, result)

    def update(self, script, location):
        """
        The main task submitted to the thread pool for executing a single script.
        
        It gathers data, runs the script, and spreads the result.

        Args:
            script: The script object to run.
            location (str): The location context.
        """
        script_data = self.gather_info(location)
        result = None
        if script_data != []:
            result = script.run(script_data)
            self.spread_info(result, location)

    def run(self):
        """
        The main simulation loop for the device thread.
        """
        while True:
            # Block Logic: Fetches the list of neighbors for the current timepoint.
            self.neighbours = self.device.supervisor.get_neighbours()

            # A return value of None from the supervisor signals simulation shutdown.
            if self.neighbours is None:
                break
            futures = []
            
            # Block Logic: This inner loop handles script processing for a single timepoint.
            while True:
                
                # Pre-condition: The loop waits until either a script is received or the
                # timepoint is marked as done by the supervisor.
                if self.device.script_received.is_set() or self.device.timepoint_done.wait():

                    # If a script has been received, process it.
                    if self.device.script_received.is_set():
                        
                        self.device.script_received.clear()
                        for (script, location) in self.device.scripts:
                            future = self.thread_pool.submit(self.update, script, location)
                            futures.append(future)
                    else:
                        # If no new scripts, but timepoint is done, exit the inner loop.
                        self.device.timepoint_done.clear()
                        self.device.script_received.set()
                        break
            
            # Block Logic: After all scripts for the timepoint have been submitted,
            # wait for them all to complete.
            wait(futures, timeout=None, return_when=ALL_COMPLETED)
            
            # Synchronize with all other devices before starting the next timepoint.
            self.device.barrier.wait()
            
        self.thread_pool.shutdown()