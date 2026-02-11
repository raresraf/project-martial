"""
This module provides a device simulation for a concurrent system.

It features a simple, non-reentrant reusable barrier and a coarse-grained
locking strategy where a single global lock is used to serialize all script
executions across all devices.
"""

from threading import Event, Thread, Condition, Lock

class ReusableBarrier(object):
    """
    A simple, non-reentrant barrier implementation using a Condition variable.

    This barrier is not a proper two-phase reusable barrier. Its attempt at
    reusability by resetting the counter can lead to race conditions if threads
    enter the `wait` cycle at different speeds.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
    def wait(self):
        """Blocks until all threads have reached the barrier."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread notifies all others and resets the counter.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a single device in the simulated network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barr = None
        self.lock = None
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects.

        A leader device (the first in the list) creates a shared barrier and a
        single global lock, which are then distributed to all devices.
        """
        
        if devices[0].barr is None and devices[0].device_id == self.device_id:
            bariera = ReusableBarrier(len(devices))
            lock = Lock()
            for i in devices:
                i.barr = bariera
            for j in devices:
                j.lock = lock
    def assign_script(self, script, location):
        """Assigns a script to the device for execution."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Sets sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for the device's lifecycle."""

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device.

        It processes assigned scripts serially for each timepoint, using a
        coarse-grained global lock.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break



            # Wait for the signal to start processing scripts for this timepoint.
            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts:
                # Acquire the single global lock, serializing all script executions
                # across the entire system.
                self.device.lock.acquire()
                script_data = []
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)




                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                self.device.lock.release()

            
            self.device.timepoint_done.clear()
            # Wait at the global barrier for all devices to complete the timepoint.
            self.device.barr.wait()
