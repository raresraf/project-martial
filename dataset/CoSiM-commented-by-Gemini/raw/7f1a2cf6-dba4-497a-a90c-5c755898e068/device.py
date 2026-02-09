"""
@file device.py
@brief Defines a device model using a manual thread pool for script execution.

This file implements a simulation device that parallelizes script execution
by creating a pool of "minion" threads. The main `DeviceThread` distributes
tasks to these minions in a round-robin fashion. The actual work is performed
by a standalone `run_task` function, which correctly uses shared, location-
specific locks to ensure data consistency.
"""

from threading import Event, Thread, Lock
from multiprocessing import cpu_count
from barrier import *


class Device(object):
    """
    Represents a device in the simulation.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        # The timepoint_done attribute is used to hold the shared barrier instance.
        self.timepoint_done = None
        self.thread = DeviceThread(self)
        self.neighbours = []
        self.locks = None
        # Determine the number of worker threads based on CPU count, with a minimum of 8.
        self.max_minions = max(8, cpu_count())

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared barrier and location-specific locks.

        Executed by the root device (ID 0). It scans all devices to find every
        unique location and creates a corresponding lock, then shares these
        resources with all other devices.
        """
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            locks = {}
            
            # Block Logic: Discover all unique locations and create a lock for each.
            for dev in devices:
                for pair in dev.sensor_data:
                    if not pair in locks:
                        locks[pair] = Lock()
            
            # Invariant: Distribute the shared barrier and lock map to all devices.
            for dev in devices:
                dev.timepoint_done = barrier
                dev.locks = locks
                dev.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be executed."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that script assignment is complete.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data. This read is not synchronized by this method."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data. This write is not synchronized by this method."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

def run_task(device, tasks):
    """
    The target function for worker threads. Processes a batch of script tasks.

    Args:
        device (Device): The parent device instance.
        tasks (list): A list of (script, location) tuples to be executed.
    """
    for task in tasks:
        (script, location) = task
        script_data = []

        # Pre-condition: Acquire the specific lock for the script's location.
        device.locks[location].acquire()

        # Block Logic: Gather data from neighbors and the local device.
        for dev in device.neighbours:
            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = device.get_data(location)
        if data is not None:
            script_data.append(data)
        
        # Invariant: Data is gathered and ready for execution.
        if script_data != []:
            result = script.run(script_data)

            # Propagate the result.
            for dev in device.neighbours:
                dev.set_data(location, result)
            
            device.set_data(location, result)
        
        # Release the location-specific lock.
        device.locks[location].release()

class DeviceThread(Thread):
    """
    The main control thread for a device, managing a pool of minion threads.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        minions = []
        while True:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                break
            
            # Wait for supervisor to signal that scripts are assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Block Logic: Distribute the assigned scripts to the worker threads
            # in a round-robin fashion.
            tasks = {}
            for i in xrange(self.device.max_minions):
                tasks[i] = []

            for i in xrange(len(self.device.scripts)):
                tasks[i % self.device.max_minions].append(self.device.scripts[i])

            # Block Logic: Create and start a thread for each batch of tasks.
            for i in xrange(self.device.max_minions):
                if len(tasks[i]) > 0:
                    minions.append(Thread(target=run_task, args=(self.device, tasks[i])))

            for minion in minions:
                minion.start()
            
            # Wait for all worker threads to complete.
            for minion in minions:
                minion.join()

            # Clear the list of worker threads for the next timepoint.
            while len(minions) > 0:
                minions.remove(minions[0])
            
            # Wait at the barrier for all other devices to finish the timepoint.
            self.device.timepoint_done.wait()