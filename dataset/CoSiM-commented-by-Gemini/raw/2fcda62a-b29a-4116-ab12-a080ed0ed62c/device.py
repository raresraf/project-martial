"""
@file device.py
@brief Defines a multi-threaded device simulation framework.

This module provides classes to simulate a network of devices that process
sensor data in parallel. It relies on a shared barrier for synchronization
across devices and uses multiple threads within each device to process scripts
concurrently.
"""

from threading import Event, Thread, Lock
import Barrier


class Device(object):
    """
    Represents a single device in the simulation network.

    Manages the device's state, its assigned scripts, and its worker thread.
    It plays a role in a coordinated setup process where device 0 acts as
    the master.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device.

        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary containing the initial sensor data.
        @param supervisor The supervisor object managing the device network.
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
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the setup of shared synchronization objects for all devices.

        Functional Utility: Device 0 is responsible for creating a single barrier
        and a shared dictionary of locks, which are then distributed to all other
        devices in the simulation. This ensures all devices synchronize on the
        same objects.
        """
        for device in devices:
            if self.device_id != device.device_id:
                self.devices.append(device)

        # Block Logic: Device 0 acts as the master for initialization.
        if self.device_id == 0:
            # The barrier synchronizes all devices at the end of a timepoint.
            self.barrier = Barrier.Barrier(len(devices))
            # The locks dictionary provides per-location thread safety.
            self.locks = {}
            # Distribute shared objects to all other devices.
            for device in devices:
                device.barrier = self.barrier
                device.locks = self.locks
                
        # Signal that this device has completed its setup phase.
        self.setup_done.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device and ensures a lock exists for its location.

        @param script The script to be executed.
        @param location The data location the script targets.
        """

        if script is not None:
            # Block Logic: Lazily initialize locks for new data locations.
            if not (self.locks).has_key(location):
                self.locks[location] = Lock()
                
            self.scripts.append((script, location))

        else:
            # A None script signals that all scripts for the timepoint are assigned.
            self.script_received.set()
            

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        res = None
        if location in self.sensor_data:
            res = self.sensor_data[location]

        return res

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's worker thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The worker thread for a Device, which orchestrates script execution and
    synchronization for each simulation timepoint.
    """

    def __init__(self, device):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    @staticmethod
    def split(script_list, number):
        """
        Partitions a list into a specified number of sublists.

        @param script_list The list to partition.
        @param number The desired number of partitions.
        @return A list of sublists.
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
        Executes a batch of scripts, handling data aggregation and synchronization.

        @param scripts A list of (script, location) tuples to execute.
        @param neighbours A list of neighboring devices.
        """
        for (script, location) in scripts:
            # Use a 'with' statement to ensure the lock is acquired and released correctly.
            with self.device.locks[location]:
                # This block is a critical section for the given 'location'.
                script_data = []
                
                # Aggregate data from all neighbors.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Aggregate data from the current device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Execute the script on the aggregated data.
                    result = script.run(script_data)

                    # Broadcast the result to all participating devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)


    def run(self):
        """The main simulation loop for the device."""

        # Block Logic: Ensure all devices have completed their setup before starting.
        self.device.setup_done.wait()
        for device in self.device.devices:
            device.setup_done.wait()
            

        while True:
            # Get the current network topology for this timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbors signals the end of the simulation.
                break

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()
            
            # Block Logic: Parallelize the execution of assigned scripts.
            if self.device.scripts:
                # Partition the scripts into 8 batches for concurrent execution.
                scripts_list = self.split(self.device.scripts, 8)
                
                thread_list = []
                # Create a thread for each batch of scripts.
                for scripts in scripts_list:
                    new_thread = Thread(target=self.run_scripts,
                                                     args=(scripts, neighbours))
                    thread_list.append(new_thread)
                    
                # Start and wait for all script-processing threads to complete.
                for thread in thread_list:
                    thread.start()
                for thread in thread_list:
                    thread.join()

            # Reset the event for the next timepoint.
            self.device.script_received.clear()
            
            # Block Logic: Synchronize with all other devices.
            # No device can proceed to the next timepoint until all have reached this barrier.
            self.device.barrier.wait()
