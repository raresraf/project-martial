"""
This module implements a distributed device simulation using Python's threading
capabilities. It models a network of devices that can execute scripts and
share data with their neighbors.

The simulation is coordinated through a combination of synchronization primitives:
- A reusable barrier to synchronize devices at the end of each time step.
- Locks to protect shared data structures like script locations.
- Semaphores to limit the number of concurrently running script calculators.
- Events to signal state changes, such as the receipt of new scripts.
"""

from threading import Event, Thread, Condition, Lock, Semaphore
from random import shuffle


class Device(object):
    """
    Represents a single device in the simulated network.

    Each device manages its own sensor data and a list of scripts to be executed.
    It communicates with a supervisor to get information about its neighbors and
    uses various synchronization mechanisms to coordinate with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary of the device's local sensor data.
            supervisor (Supervisor): The central supervisor for the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.setup_ev = Event()
        self.barrier = None
        self.lock = None
        self.get_lock = None
        self.scripts = []
        self.script_locations = {}
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared synchronization objects for all devices.

        This method should be called on the master device (ID 0) to create
        and distribute the barrier and locks to all other devices.
        """
        self.devices = devices

        if (self.device_id == 0):
            self.barrier = ReusableBarrier(len(devices))
            self.lock = Lock()
            self.get_lock = Lock()

            for i in xrange(1, len(devices)):
                devices[i].barrier = self.barrier
                devices[i].lock = self.lock
                devices[i].get_lock = self.get_lock
                devices[i].script_locations = self.script_locations
                devices[i].setup_ev.set()

            self.setup_ev.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A None script signals the end of a
        timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates the sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, managing its execution loop.
    """

    def __init__(self, device):
        """
        Initializes the device thread and a semaphore to limit concurrent script
        executions.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.no_cores = 8
        self.semaphore = Semaphore(self.no_cores + 1)

    def run(self):
        """
        The main simulation loop for the device.
        
        Waits for setup, then enters a loop of fetching neighbors, processing
        scripts, and synchronizing with other devices.
        """
        self.device.setup_ev.wait()

        while True:
            # Safely get the list of neighbors.
            self.device.get_lock.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.get_lock.release()

            if neighbours is None:
                break

            # Wait for the signal that all scripts for the timepoint are received.
            self.device.script_received.wait()
            self.device.script_received.clear()

            calcs = []

            # Launch a ScriptCalculator thread for each script.
            for i in xrange(0, len(self.device.scripts)):
                with self.device.lock:
                    if not self.device.script_locations.has_key(self.device.scripts[i][1]):
                        # Create a lock for each new script location.
                        self.device.script_locations[self.device.scripts[i][1]] = Lock()
                calcs.append(ScriptCalculator(i, self.device, neighbours, self.device.scripts[i], self.semaphore))
                self.semaphore.acquire()
                calcs[i].start()
                self.semaphore.release()

            # Wait for all script calculators to finish.
            for i in xrange(0, len(self.device.scripts)):
                calcs[i].join()

            # Synchronize with all other devices at the barrier.
            self.device.barrier.wait()


class ReusableBarrier():
    """
    A simple reusable barrier implementation for thread synchronization.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
        self.lock = Lock()

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have arrived.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread to arrive notifies all waiting threads.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class ScriptCalculator(Thread):
    """
    A worker thread for executing a single script.
    """

    def __init__(self, name, device, neighbours, script, semaphore):
        Thread.__init__(self)
        self.name = name
        self.device = device
        self.neighbours = neighbours
        self.scripts = script
        self.semaphore = semaphore

    def run(self):
        """
        The main execution logic for the script calculator.

        It acquires a semaphore slot, gathers data, executes the script,
        and updates the data on all relevant devices.
        """
        self.semaphore.acquire()

        script = self.scripts[0]
        location = self.scripts[1]

        # Acquire a lock for the specific script location to prevent race conditions.
        self.device.script_locations[location].acquire()
        script_data = []

        # Gather data from neighboring devices.
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        # Gather data from the local device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Run the script with the collected data.
            result = script.run(script_data)

            # Propagate the result to all neighbors and the local device.
            for device in self.neighbours:
                device.set_data(location, result)
            self.device.set_data(location, result)
            
        self.device.script_locations[location].release()
        self.semaphore.release()