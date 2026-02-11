"""
This module provides a Python 2 implementation for a simulated device in a
concurrent, distributed network. It defines the device's behavior, its
synchronization mechanisms, and the threading model for executing tasks.

A notable feature of this implementation is a custom `ReusableBarrier` built
using a `threading.Condition` and a work-stealing approach for task distribution
among worker threads.
"""

from threading import Event, Thread, Lock, Condition


class Device(object):
    """
    Represents a single device node in the simulation.

    Each device has an ID, sensor data, and executes scripts in coordination
    with other devices. Synchronization is managed through a shared barrier and
    a shared set of locks for different data locations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary holding the device's sensor data.
            supervisor (object): The supervisor managing the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects to all devices.

        This method creates a single shared barrier and a single shared dictionary
        of locks (`lock_set`) that are distributed to all devices in the network.
        This centralized approach ensures all devices use the same sync primitives.
        """

        lock_set = {}
        barrier = ReusableBarrier(len(devices))
        idx = len(devices) - 1

        # Iterate backwards to set up each device.
        while idx >= 0:
            current_device = devices[idx]
            current_device.barrier = barrier
            for current_location in current_device.sensor_data:
                lock_set[current_location] = Lock()
            current_device.lock_set = lock_set
            idx = idx - 1

    def assign_script(self, script, location):
        """Assigns a script to the device for execution."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data
        else:
            pass

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()

class ReusableBarrier(object):
    """
    A custom implementation of a reusable barrier using a Condition variable.

    This barrier allows a set number of threads to wait for each other to reach
    a certain point of execution before proceeding.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # All threads have arrived, notify all waiting threads.
            self.cond.notify_all()
            # Reset the barrier for the next use.
            self.count_threads = self.num_threads
        else:
            # Wait for the other threads to arrive.
            self.cond.wait()
        self.cond.release()

    def print_barrier(self):
        print self.num_threads, self.count_threads

class DeviceThread(Thread):
    """The main execution thread that manages the device's lifecycle."""

    def __init__(self, device):
        """
        Initializes the main device thread.
        
        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device

    def run(self):
        """
        The main loop, coordinating script execution in discrete timepoints.
        
        It uses a fixed-size pool of worker threads (`MakeUpdate`) to execute
        assigned scripts in parallel.
        """
        nr_threads = 8
        while True:
            self.device.timepoint_done.clear()
            neigh = self.device.supervisor.get_neighbours()
            # First barrier: synchronize all devices before starting the timepoint.
            self.device.barrier.wait()
            if neigh is None:
                break
            # Wait for a signal to start script execution for this timepoint.
            self.device.timepoint_done.wait()
            execute_script = []
            threads = []
            for script in self.device.scripts:
                execute_script.append(script)
            
            # Create a pool of worker threads to process the script list.
            for i in xrange(nr_threads):
                threads.append(MakeUpdate(self.device, neigh, execute_script))
                threads[i].start()

            for t in threads:
                t.join()
            # Second barrier: synchronize all devices after the timepoint is complete.
            self.device.barrier.wait()

class MakeUpdate(Thread):
    """
    A worker thread that executes scripts from a shared list.
    
    This class implements a work-stealing mechanism, where each worker thread
    atomically pops a script from a shared list and executes it.
    """

    def __init__(self, device, neighbours, execute_script):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.execute_script = execute_script

    def run(self):
        """Pops a script from the shared list and executes it."""
        if len(self.execute_script) != 0:
            collected = []
            # Atomically pop a script from the shared list (task queue).
            (script, location) = self.execute_script.pop()
            # Acquire the lock for the specific location to ensure exclusive access.
            self.device.lock_set[location].acquire()
            for neigh_c in self.neighbours:
                collected.append(neigh_c.get_data(location))
            collected.append(self.device.get_data(location))

            if collected != []:
                result = script.run(collected)
                # Update data on the current device and all neighbors.
                self.device.set_data(location, result)
                for neigh_c in self.neighbours:
                    neigh_c.set_data(location, result)
            self.device.lock_set[location].release()
