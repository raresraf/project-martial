"""
Models a distributed system of devices that execute scripts on sensor data.

This file defines a simulation framework for a network of devices that can
execute computational scripts. The devices operate in parallel, synchronize at
each timepoint, and use locks to ensure data consistency when accessing sensor
data at shared locations. The simulation appears to be designed for a scenario
where devices and their neighbors collaboratively process data.
"""

from threading import Event, Thread, Lock
from reusable_barrier_semaphore import ReusableBarrier


class Device(object):
    """Represents a single device in the distributed system.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary mapping locations to sensor readings.
        supervisor (Supervisor): A central object for managing device interactions.
        script_received (Event): An event to signal the arrival of new scripts.
        scripts (list): A list of (script, location) tuples to be executed.
        location_locks (list): A list of locks, one for each data location.
        next_timepoint_barrier (ReusableBarrier): A barrier for synchronizing all devices.
        thread (DeviceThread): The worker thread for this device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data


        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.location_locks = []
        self.next_timepoint_barrier = ReusableBarrier(0)
        self.thread = DeviceThread(self)

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    @staticmethod
    def count_locations(devices, devices_nr):
        """Calculates the total number of unique sensor locations across all devices.

        Args:
            devices (list): The list of all Device objects.
            devices_nr (int): The number of devices.

        Returns:
            int: The highest location index + 1.
        """
        locations_number = 0

        for i in range(devices_nr):
            for location in devices[i].sensor_data.keys():
                if location > locations_number:
                    locations_number = location

        locations_number = locations_number + 1

        return locations_number

    def setup_devices(self, devices):
        """Initializes and starts the threads for all devices.

        This method is intended to be called on one device (e.g., device 0)
        to set up the entire system. It creates shared locks for all locations
        and a shared barrier for synchronization, then starts each device's thread.

        Args:
            devices (list): The list of all Device objects in the system.
        """
        devices_nr = len(devices)
        
        next_timepoint_barrier = ReusableBarrier(devices_nr)

        # The device with ID 0 is responsible for global setup.
        if self.device_id == 0:
            locations_number = self.count_locations(devices, devices_nr)

            # Create a shared lock for each location.
            for i in range(locations_number):
                lock = Lock()
                self.location_locks.append(lock)

            # Distribute the shared locks and barrier to all devices.
            for i in range(devices_nr):


                for j in range(locations_number):
                    devices[i].location_locks.append(self.location_locks[j])

                devices[i].next_timepoint_barrier = next_timepoint_barrier
                devices[i].thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be executed by this device.

        Args:
            script (Script): The script object to execute.
            location (int): The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts for the current timepoint have been assigned.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location.

        Args:
            location (int): The location to query.

        Returns:
            The sensor data if the location exists for this device, otherwise None.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data for a given location.

        Args:
            location (int): The location to update.
            data: The new sensor data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main execution thread for a device.

    This thread orchestrates the device's lifecycle, which consists of waiting
    for scripts, executing them, and synchronizing with other devices at each
    timepoint.
    """

    def __init__(self, device):
        """Initializes the device thread."""


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main loop for the device thread."""
        scriptsolvers = []

        while True:
            # Get the device's neighbors for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                # A None value for neighbours signals the end of the simulation.
                break

            # Wait until all scripts for the current timepoint have been assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Create a ScriptSolver thread for each assigned script.
            for (script, location) in self.device.scripts:
                scriptsolvers.append(
                    ScriptSolver(self.device, script, neighbours, location))

            workers_nr = len(scriptsolvers)

            for index in range(workers_nr):
                scriptsolvers[index].start()

            for index in range(workers_nr):
                scriptsolvers[index].join()

            
            scriptsolvers = []

            # Wait at the barrier for all other devices to finish the current timepoint.
            self.device.next_timepoint_barrier.wait()


class ScriptSolver(Thread):
    """A thread responsible for executing a single script on collected data."""
    def __init__(self, device, script, neighbours, location):
        """Initializes the ScriptSolver."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location

    def collect_data(self, neighbours, location):
        """Gathers data from the device and its neighbors at a specific location.

        Args:
            neighbours (list): A list of neighboring Device objects.
            location (int): The location from which to collect data.

        Returns:
            list: A list of data points from the device and its neighbors.
        """
        data_script = []
        own_data = self.device.get_data(location)

        
        if own_data is not None:
            data_script.append(own_data)

        
        for device in neighbours:

            data = device.get_data(location)

            if data is not None:
                data_script.append(device.get_data(location))

        return data_script

    def update_data(self, neighbours, location, run_result):
        """Updates the data on the device and its neighbors with the script's result.

        Args:
            neighbours (list): A list of neighboring Device objects.
            location (int): The location to update.
            run_result: The result from the script execution.
        """
     	
        self.device.set_data(location, run_result)

        
        for device in neighbours:
            device.set_data(location, run_result)

    def solve(self, script, neighbours, location):
        """Executes the script.

        This method acquires a lock for the target location, collects data,
        runs the script, updates the data, and releases the lock.

        Args:
            script (Script): The script to execute.
            neighbours (list): List of neighboring devices.
            location (int): The location to process.
        """
        
        # Acquire a lock to ensure exclusive access to the location's data.
        self.device.location_locks[location].acquire()

        data_script = self.collect_data(neighbours, location)

        
        if data_script != []:
            # Run the script on the collected data.
            run_result = script.run(data_script)



            self.update_data(neighbours, location, run_result)

        
        self.device.location_locks[location].release()

    def run(self):
        """The entry point for the ScriptSolver thread."""
        self.solve(self.script, self.neighbours, self.location)

# The following code seems to be an bundled dependency.
# It is a common pattern in competitive programming to include all code in one file.
from threading import *

class ReusableBarrier():
    """A reusable barrier implementation using semaphores.
    
    This barrier synchronizes a fixed number of threads at a rendezvous point.
    It is reusable, meaning it can be used multiple times. It employs a two-phase
    protocol to prevent threads from one barrier instance from proceeding into
    the next instance before all threads have left the first one.
    """
    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        # Counter for threads arriving at the first phase.
        self.count_threads1 = [self.num_threads]
        # Counter for threads arriving at the second phase.
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        # Semaphore for the first phase.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for the second phase.
        self.threads_sem2 = Semaphore(0)
    
    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
    
    def phase(self, count_threads, threads_sem):
        """Implements one phase of the barrier.
        
        Args:
            count_threads (list): A list containing the current thread count for this phase.
            threads_sem (Semaphore): The semaphore to block/release threads for this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread to arrive opens the gate for all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # All other threads wait here until the gate is opened.
        threads_sem.acquire()
                                                 
