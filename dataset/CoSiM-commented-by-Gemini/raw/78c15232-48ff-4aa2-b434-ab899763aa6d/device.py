"""
@file device.py
@brief Defines a device model for a simulation using pre-allocated location locks.

This file implements a `Device` class that uses a list of pre-allocated, shared
locks to synchronize access to data based on location. The root device (ID 0)
is responsible for creating and distributing these locks and a shared,
condition-based `ReusableBarrier`. Script execution is parallelized by dynamically
creating threads that run a method of the main `DeviceThread`.
"""

from threading import *

class ReusableBarrier():
    """
    A reusable barrier implemented using a Condition variable.

    @note This implementation may be subject to race conditions. If a notified
          thread re-enters `wait()` before the last thread (which triggered the
          notification) has released the condition lock and exited the `wait()`,
          it could lead to deadlocks or unpredictable behavior.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        self.cond.acquire()
        self.count_threads -= 1;
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait();
        self.cond.release();

class Device(object):
    """
    Represents a single device in the simulation.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barrier = None
        self.locationLock = None
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared barrier and a list of location locks.

        The root device (ID 0) pre-allocates 10,000 locks, assuming location IDs
        will be used as indices into this list.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            self.locationLock = []

            # Pre-allocate a large number of locks for location-based synchronization.
            for i in range(0, 10000):
                loc = Lock()
                self.locationLock.append(loc)
            
            # Invariant: Distribute the shared barrier and lock list to all devices.
            for i in devices:
                i.barrier = self.barrier
                i.locationLock = self.locationLock

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of script assignment for the timepoint.
            self.timepoint_done.set()

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

class DeviceThread(Thread):
    """
    The main control thread for a device, which spawns worker threads for scripts.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_script(self, script, location, neighbours):
        """
        The target function for worker threads, executing a single script.
        
        This method uses a `with` statement to ensure that the location-specific
        lock is acquired before data access and released afterward.
        """
        script_data = []
        # Pre-condition: Acquire the lock for the specific location.
        with self.device.locationLock[location]:
            # Block Logic: Gather data from neighbors and self.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Invariant: Data is gathered, and the script is ready for execution.
            if script_data != []:
                result = script.run(script_data)
                
                # Propagate the result back to all devices.
                for device in neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)

    def run(self):
        """The main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for the supervisor to signal that scripts have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Create a list of arguments for each script to be executed.
            queue = []
            for (script, location) in self.device.scripts:
                queue.append((script, location, neighbours))

            # Block Logic: Create a thread for each script task.
            subThList = []
            while len(queue) > 0:
                subThList.append(Thread(target = self.run_script, args = queue.pop()))

            # Start all threads, then wait for them to complete.
            for t in subThList:
                t.start()
            for t in subThList:
                t.join()
            
            # Wait at the barrier for all other devices to finish the timepoint.
            self.device.barrier.wait()

            self.device.script_received.clear()
            self.device.timepoint_done.clear()