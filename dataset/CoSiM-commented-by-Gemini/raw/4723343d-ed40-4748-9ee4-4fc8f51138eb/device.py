"""
This module contains a version of the device simulation framework that uses a
dynamic, per-timepoint thread pool for script execution.

Key architectural features:
- A custom `ReusableBarrier` is implemented for synchronization.
- Each `Device` has a single main `DeviceThread`.
- The `DeviceThread` acts as a dispatcher, dynamically creating a new set of
  `DeviceThread_Worker` threads for each batch of scripts received in a timepoint.
- Locking is complex and appears to have significant performance and correctness
  issues (e.g., risk of deadlock, inefficient lock acquisition patterns).
"""

from threading import Event, Thread
from threading import Condition, RLock    

class ReusableBarrier():
    """A custom implementation of a reusable barrier.

    This barrier allows a set number of threads to wait for each other to reach
    a certain point of execution before any of them are allowed to continue.
    It resets automatically after all threads have passed.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()
 
    def wait(self):
        """Causes a thread to block until all `num_threads` have called wait."""
        self.cond.acquire()                      
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread has arrived; notify all waiting threads and reset.
            self.cond.notify_all()              
            self.count_threads = self.num_threads    
        else:
            self.cond.wait()                    
        self.cond.release()


class DeviceThread_Worker(Thread):
    """A short-lived worker thread created to process a subset of a device's scripts."""
    def __init__(self, device, neighbours, tid, scripts):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.device = device
        self.scripts = scripts
        self.tid = tid 

    def run(self):
        """Processes assigned scripts.

        Note: The locking strategy in this method is highly convoluted and
        appears prone to deadlocks and performance issues. It acquires and
        releases locks repeatedly in a nested fashion.
        """
        for (script, location) in self.scripts:
            script_data = []
            index = location

            # Gather data from neighbors with very inefficient locking.
            for device in self.neighbours:
                # Acquiring both a location lock and a global device lock per neighbor.
                self.device.locks[index].acquire()
                self.device.lock.acquire()
                data = device.get_data(location)
                self.device.lock.release()
                self.device.locks[index].release()
                if data is not None:
                    script_data.append(data)

            # Gather data from self, again with redundant locking.
            self.device.locks[index].acquire()
            self.device.lock.acquire()
            data = self.device.get_data(location)
            self.device.lock.release()
            self.device.locks[index].release()
            if data is not None:
                script_data.append(data)

            if script_data:
                result = script.run(script_data)

                # Update data on all devices, with custom logic.
                with self.device.locks[index]:
                    for dev in self.neighbours:
                        # Only updates if the new result is greater.
                        if result > dev.get_data(location):
                            dev.set_data(location, result)
                    self.device.set_data(location, result)


class Device(object):
    """Represents a device that uses a dynamic thread pool for work."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.lock = RLock()  # A general-purpose reentrant lock.
        self.barrier = None
        self.devices = []
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.locations = []
        self.locks = []  # A pre-allocated pool of locks for locations.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared barriers and a large, pre-allocated pool of locks."""
        self.devices = devices
        if self.device_id == 0:
            # Create and distribute a fixed pool of 1000 locks to all devices.
            for num in range(1000):
                lock = RLock()
                for i in range (len(devices)):
                    devices[i].locks.append(lock)
            
            barrier = ReusableBarrier(len(devices)) 
            for i in range(len(devices)):
                if devices[i].barrier is None:
                    devices[i].barrier = barrier

    def assign_script(self, script, location):
        """Assigns a script to be processed."""
        if script is not None:
            self.scripts.append((script, location)) 
            # This signal seems to indicate work is ready, not that a timepoint is done.
            self.timepoint_done.set()
        else:
            # A None script signals all work for the timepoint has been assigned.
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """Gets sensor data. Not thread-safe on its own."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data. Uses a reentrant lock for safety."""
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """The main dispatcher thread for a device.

    It waits for scripts for a timepoint, dynamically creates worker threads to
    process them, and then synchronizes at a barrier.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def divide_in_threads(self, neighbours):
        """Divides the current script workload among temporary worker threads."""
        threads = []
        num_scripts = len(self.device.scripts)
        num_threads = min(8, num_scripts) # Use at most 8 threads.
        if num_threads == 0:
            return

        scripts_per_thread = num_scripts // num_threads
        
        # Create and start worker threads, partitioning the scripts list.
        for i in range(num_threads):
            start_index = i * scripts_per_thread
            end_index = len(self.device.scripts) if i == num_threads - 1 else start_index + scripts_per_thread
            
            t = DeviceThread_Worker(self.device, neighbours, i, self.device.scripts[start_index:end_index])
            threads.append(t)
            t.start()

        # Wait for all worker threads for this timepoint to complete.
        for t in threads:
            t.join()

    def run(self):
        """Main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Terminate signal.

            # Wait for the signal that all scripts for the timepoint have arrived.
            self.device.script_received.wait()

            # Process the batch of scripts using a dynamic pool of workers.
            self.divide_in_threads(neighbours)

            self.device.script_received.clear()

            # Synchronize with all other devices to mark the end of the timepoint.
            self.device.barrier.wait()
