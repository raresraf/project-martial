"""
This module implements a device simulation for a concurrent system.

Its architecture is notable for dynamically creating a new thread for every
script execution in each timepoint, which is an inefficient, heavyweight
approach to concurrency. It uses a simple, non-reentrant barrier and a large,
pre-allocated list of locks.
"""

from threading import *
import Queue

class ReusableBarrier():
    """
    A simple, non-reentrant barrier implementation using a Condition variable.

    This barrier is not a proper two-phase reusable barrier and may be prone to
    race conditions if threads enter the `wait` cycle at different speeds.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()


    def wait(self):
        """Blocks until all threads have reached the barrier."""
        self.cond.acquire()
        self.count_threads -= 1;
        if self.count_threads == 0:
            # Last thread notifies all others and resets the counter.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait();
        self.cond.release();

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
        self.barrier = None
        self.locationLock = None
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization objects for all devices.

        The leader device (ID 0) creates a shared barrier and pre-allocates a
        large, fixed-size list of 10,000 locks, which are then distributed
        to all other devices.
        """

        if self.device_id == 0:

            
            self.barrier = ReusableBarrier(len(devices))

            
            self.locationLock = []
            # Pre-allocate a large number of locks for different locations.
            for i in range(0, 10000):
                loc = Lock()
                self.locationLock.append(loc)

            
            for i in devices:
                i.barrier = self.barrier
                i.locationLock = self.locationLock

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

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

    def run_script(self, script, location, neighbours):
        """
        The target function for script execution threads.

        This method contains the logic for a single script run, including
        acquiring the correct lock, gathering data, and updating results.
        """
        script_data = []
        # Acquire the specific lock for this location from the shared list.
        with self.device.locationLock[location]:

            
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


    def run(self):
        """
        The main execution loop for the device.

        In each timepoint, this loop spawns a new thread for every assigned
        script, which is a highly inefficient threading model.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the signal to start processing for the timepoint.
            self.device.timepoint_done.wait()

            
            # Create a list of tasks for the current timepoint.
            queue = []
            for (script, location) in self.device.scripts:
                queue.append((script, location, neighbours))

            subThList = []
            # Inefficiently create a new thread for every single task.
            while len(queue) > 0:
                subThList.append(Thread(target = self.run_script, args = queue.pop()))



            for t in subThList:
                t.start()

            for t in subThList:
                t.join()

            
            # Wait at the global barrier for all other devices.
            self.device.barrier.wait()

            
            self.device.script_received.clear()
            self.device.timepoint_done.clear()

