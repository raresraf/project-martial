"""
@file device.py
@brief Defines a device model for a distributed simulation with a condition-based barrier.

This file implements a `Device` class that uses a dedicated `MyThread` worker for
each script execution. Synchronization is managed by a custom `ReusableBarrier`
class that is implemented using a `threading.Condition` variable.
"""

from threading import Event, Thread, Condition, Lock


class ReusableBarrier():
    """
    A reusable barrier implemented using a Condition variable.

    This barrier blocks threads calling `wait()` until a specified number of
    threads have arrived.

    Note: This implementation may be subject to race conditions. If a notified
    thread re-enters `wait()` before the last thread (which triggered the
    notification) has released the condition lock and exited the `wait()`,
    it could lead to deadlocks or unpredictable behavior.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
        
    def wait(self):
        """
        Blocks the calling thread until all threads have reached the barrier.
        """
        self.cond.acquire()
        
        self.count_threads -= 1
        # Pre-condition: The last thread to arrive at the barrier enters this block.
        if self.count_threads == 0:
            # Notify all waiting threads to wake up.
            self.cond.notify_all()
            # Reset the counter for the barrier's next use.
            self.count_threads = self.num_threads
        else:
            # Wait to be notified by the last thread.
            self.cond.wait()
            
        self.cond.release()

class Device(object):
    """
    Represents a single device in the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): Dictionary of sensor readings.
            supervisor: The central simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.lock = Lock()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for all devices.

        The root device (ID 0) creates the barrier, and all other devices
        obtain a reference to it.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
        else:
            self.barrier = devices[0].barrier
        
    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script: The script object to run.
            location: The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of script assignment.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class MyThread(Thread):
    """
    A worker thread for executing a single script.
    """
    def __init__(self, neighbours, device, location, script):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.device = device
        self.location = location
        self.script = script
        
    def run(self):
        """
        Gathers data, runs the script, and propagates the result.
        
        Locking is performed on a per-device basis during the data update phase.
        """
        script_data = []
        # Block Logic: Gather data from all neighbors.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Gather data from the local device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Invariant: Data is gathered and ready for script execution.
        if script_data != []:
            
            result = self.script.run(script_data)

            # Block Logic: Propagate the result. Each device's set_data call
            # is protected by its own lock.
            for device in self.neighbours:
                device.lock.acquire()
                device.set_data(self.location, result)
                device.lock.release()
            
            self.device.lock.acquire()
            self.device.set_data(self.location, result)
            self.device.lock.release()

class DeviceThread(Thread):
    """
    The main control thread for a Device.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop.
        """
        while True:
            # Get neighbors for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for supervisor to signal end of script assignment.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            threads = []
            
            # Block Logic: Spawn a worker thread (`MyThread`) for each assigned script.
            for (script, location) in self.device.scripts:
                t = MyThread(neighbours, self.device, location, script)
                t.start()
                threads.append(t)

            # Wait for all script threads to complete.
            for i in range(len(threads)):
                threads[i].join()
            
            # Synchronize with all other devices before the next timepoint.
            self.device.barrier.wait()