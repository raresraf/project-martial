"""
@file device.py
@brief Defines a device model for a synchronized, distributed simulation.

This file implements a `Device` and its `DeviceThread` for a simulated network.
Synchronization between devices is managed using a shared, class-level barrier
and lock, initialized by a designated root device.
"""

import barrier
from threading import Event, Thread, Lock


class Device(object):
    """
    Represents a device in the simulation, managing sensor data and script execution.

    All devices share a single barrier and a lock for synchronization, which are
    managed as class-level variables.
    """
    
    # Class-level variables to hold shared synchronization objects.
    barrier = None
    lock = None

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of the device's sensor readings.
            supervisor: The central supervisor object managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization primitives and starts the device's thread.

        If this is the root device (device_id == 0), it creates the shared
        barrier and lock for all devices. It then starts the main execution
        thread for the current device.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        # Block Logic: The root device (ID 0) is responsible for initializing
        # the shared barrier and lock that all device instances will use.
        if(self.device_id == 0):
             Device.barrier = barrier.ReusableBarrierCond(len(devices))
             Device.lock = Lock()
        
        # The execution thread is created and started for each device.
        self.thread = DeviceThread(self, Device.barrier, Device.lock)
        self.thread.start()        
        
    def assign_script(self, script, location):
        """
        Assigns a script to the device for a specific location.

        Args:
            script: The script to be executed.
            location: The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts for the current
            # timepoint have been received.
            self.script_received.set()        
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Note: Access to sensor_data is not internally locked in this method;
        callers are responsible for acquiring the shared lock.

        Args:
            location: The location identifier.
        
        Returns:
            The data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location.

        Note: Access to sensor_data is not internally locked in this method;
        callers are responsible for acquiring the shared lock.

        Args:
            location: The location identifier.
            data: The new data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()




class DeviceThread(Thread):
    """
    The main execution thread for a Device.

    This thread synchronizes with other devices at the beginning of each
    timepoint, processes assigned scripts, and then waits for a signal
    to end the timepoint.
    """

    def __init__(self, device, barrier, lock):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device object.
            barrier: The shared ReusableBarrier for synchronization.
            lock (Lock): The shared lock for protecting access to sensor data.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.lock = lock

    def run(self):
        """The main simulation loop for the device thread."""
        while True:
            # Block Logic: All threads wait at the barrier, ensuring they start
            # each timepoint in a synchronized manner.
            self.barrier.wait()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbours signals the end of the simulation.
                break

            # Waits for the signal that all scripts for this timepoint are assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            
            # Block Logic: Executes all assigned scripts for the current timepoint.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Block Logic: Gathers data from neighbors. A lock is acquired and
                # released for each neighbor, which can be inefficient.
                for device in neighbours:
                    self.lock.acquire()
                    data = device.get_data(location)
                    self.lock.release()
                    if data is not None:
                        script_data.append(data)
                
                # Also gather data from the local device.
                self.lock.acquire()
                data = self.device.get_data(location)
                self.lock.release()
                if data is not None:
                    script_data.append(data)
                
                # Invariant: `script_data` now holds all data for the specified
                # location from the local device and its neighbors.
                if script_data != []:
                    
                    result = script.run(script_data)

                    # Block Logic: Propagates the script result back to all devices.
                    # As with data gathering, the lock is acquired for each device update.
                    for device in neighbours:
                        self.lock.acquire()
                        device.set_data(location, result)
                        self.lock.release()
                    
                    self.lock.acquire()
                    self.device.set_data(location, result)
                    self.lock.release()

            
            # Waits for a signal indicating the end of the timepoint before looping.
            self.device.timepoint_done.wait()