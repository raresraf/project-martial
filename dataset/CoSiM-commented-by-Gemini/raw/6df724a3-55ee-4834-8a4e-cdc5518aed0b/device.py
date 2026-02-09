"""
@file device.py
@brief Defines a device model for a simulation using per-device write locks.

This file implements a simulation device that uses a dedicated `MiniT` worker
thread for each script execution. Data consistency during the write-back phase
is handled by acquiring a lock on each individual device before updating its data.
A custom, potentially racy, condition-based `ReusableBarrier` is used for
timepoint synchronization.
"""

from threading import Event, Thread, Lock, Condition

class ReusableBarrier(object):
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
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    Represents a single device in the simulation, with its own lock for data writes.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.bariera = ReusableBarrier(0)
        # Each device has its own lock for protecting writes to its data.
        # "lacat_date" is Romanian for "data lock".
        self.lacat_date = Lock()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared barrier to all devices.
        """
        if self.device_id == 0:
            barria = ReusableBarrier(len(devices))
            for device in devices:
                device.bariera = barria

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of script assignment for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. This read operation is not synchronized."""
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        """Updates sensor data. This write operation is not synchronized by this method."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, which spawns `MiniT` worker threads.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_list = list()

    def run(self):
        """The main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to signal that scripts have been assigned.
            self.device.timepoint_done.wait()

            self.thread_list = list()
            # Block Logic: Create a worker thread for each script.
            for (script, location) in self.device.scripts:
                minithrd = MiniT(neighbours, self.device, location, script)
                self.thread_list.append(minithrd)
            
            # Start all worker threads, then wait for them to complete.
            for i in range(len(self.thread_list)):
                self.thread_list[i].start()

            for i in range(len(self.thread_list)):
                self.thread_list[i].join()

            self.device.timepoint_done.clear()
            # Wait at the barrier for all other devices to finish the timepoint.
            self.device.bariera.wait()

class MiniT(Thread):
    """
    A worker thread that executes a single script and handles write-locking.
    """
    def __init__(self, neighbours, device, location, script):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script, acquiring a lock on each device before writing to it.
        """
        script_data = []
        # Block Logic: Gather data from neighbors. This read is unsynchronized.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)
        
        # Invariant: Data is gathered and ready for script execution.
        if script_data != []:
            result = self.script.run(script_data)

            # Block Logic: Write back the result. Acquires a lock on each
            # individual device before setting its data, ensuring write safety.
            for device in self.neighbours:
                device.lacat_date.acquire()
                device.set_data(self.location, result)
                device.lacat_date.release()

            self.device.lacat_date.acquire()
            self.device.set_data(self.location, result)
            self.device.lacat_date.release()