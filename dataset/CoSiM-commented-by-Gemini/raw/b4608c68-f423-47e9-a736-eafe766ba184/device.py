# -*- coding: utf-8 -*-
"""
This module implements a simulation of a distributed sensor network.

It defines a `Device` class representing a sensor node that operates concurrently.
Unlike a thread-pool-based approach, each device here creates its own threads
for parallel script execution in each time step.

Classes:
    Device: Represents a single device in the network.
    DeviceThread: The main control thread for a Device, which spawns worker threads.
"""

from threading import Event, Thread, Lock
import Barrier


class Device(object):
    """
    Represents a single device in the distributed sensor network.

    Each device manages its own sensor data and executes assigned scripts.
    It uses a thread-based model where the main device thread spawns new threads
    for each batch of scripts.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor data.
        supervisor (Supervisor): An object that oversees the device's operation.
        script_received (Event): An event to signal the arrival of new scripts.
        scripts (list): A list of scripts to be executed.
        setup_done (Event): An event to signal completion of device setup.
        devices (list): A list of other devices in the network.
        barrier (Barrier.Barrier): A shared barrier for synchronizing devices.
        locks (dict): A shared dictionary of locks for sensor data locations.
        timepoint_done (Event): An event to signal the completion of a timepoint.
        thread (DeviceThread): The main execution thread for this device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor (Supervisor): The supervisor for this device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()


        self.scripts = []
        self.setup_done = Event()
        self.devices = []
        self.barrier = None
        self.locks = None
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device's connections and shared resources.

        Device with ID 0 acts as the coordinator to initialize the shared barrier
        and locks dictionary.

        Args:
            devices (list): A list of all devices in the network.
        """
        for device in devices:
            if self.device_id != device.device_id:
                self.devices.append(device)

        # Device 0 is the coordinator for setting up shared resources.
        if self.device_id == 0:
            self.barrier = Barrier.Barrier(len(devices))
            self.locks = {}
            for device in devices:
                device.barrier = self.barrier
                device.locks = self.locks
                
        self.setup_done.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        A lock for the specified location is created if it doesn't exist.
        A `None` script is a sentinel to indicate that all scripts for the
        current timepoint have been assigned.

        Args:
            script (Script): The script to execute.
            location (str): The sensor data location the script will operate on.
        """
        if script is not None:
            # Create a lock for the location if one doesn't already exist.
            if not (self.locks).has_key(location):
                self.locks[location] = Lock()
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location.

        Note: This method is not thread-safe by itself. Locking is handled
        by the calling thread (`DeviceThread.run_scripts`).

        Args:
            location (str): The location of the sensor data to retrieve.

        Returns:
            The sensor data at the given location, or None if not found.
        """
        res = None
        if location in self.sensor_data:
            res = self.sensor_data[location]
        return res

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location.

        Note: This method is not thread-safe by itself. Locking is handled
        by the calling thread (`DeviceThread.run_scripts`).

        Args:
            location (str): The location of the sensor data to update.
            data: The new data to be stored.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device.

    This thread orchestrates the device's operation, including spawning
    threads for script execution and synchronizing with other devices.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    @staticmethod
    def split(script_list, number):
        """
        Splits a list into a specified number of sublists.

        Args:
            script_list (list): The list to be split.
            number (int): The number of sublists to create.

        Returns:
            A list of sublists.
        """
        res = [[] for i in range(number)]
        i = 0
        while i < len(script_list):
            part = script_list[i]
            res[i%number].append(part)
            i = i + 1
        return res

    def run_scripts(self, scripts, neighbours):
        """
        Executes a list of scripts.

        This method is intended to be run in a separate thread. It iterates
        through the provided scripts, acquires locks, gathers data, executes
        the script, and updates data.

        Args:
            scripts (list): A list of (script, location) tuples to execute.
            neighbours (list): A list of neighboring devices.
        """
        for (script, location) in scripts:
            # Acquire a lock for the location to ensure data consistency.
            with self.device.locks[location]:
                script_data = []
                # Gather data from neighbors.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                # Gather data from the current device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    result = script.run(script_data)
                    # Broadcast the result to neighbors.
                    for device in neighbours:
                        device.set_data(location, result)
                    # Update the current device's data.
                    self.device.set_data(location, result)


    def run(self):
        """
        The main loop of the device thread.
        """
        # Wait for all devices to be set up.
        self.device.setup_done.wait()
        for device in self.device.devices:
            device.setup_done.wait()
            
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for all scripts for the timepoint to be assigned.
            self.device.script_received.wait()
            
            if self.device.scripts:
                # Split scripts among a fixed number of threads.
                scripts_list = self.split(self.device.scripts, 8)
                
                thread_list = []
                # Create and start a thread for each sublist of scripts.
                for scripts in scripts_list:
                    new_thread = Thread(target=self.run_scripts,
                                                     args=(scripts, neighbours))
                    thread_list.append(new_thread)
                    
                for thread in thread_list:
                    thread.start()
                    
                # Wait for all spawned threads to complete.
                for thread in thread_list:
                    thread.join()
                    
            self.device.script_received.clear()
            
            # Synchronize with all other devices.
            self.device.barrier.wait()
