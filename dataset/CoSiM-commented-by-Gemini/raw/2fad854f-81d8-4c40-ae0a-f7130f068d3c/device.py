"""
@file device.py
@brief Defines a framework for a multi-threaded device simulation.

This module provides classes for simulating a network of devices that process sensor
data in a synchronized, parallel manner. It includes a reusable barrier for
synchronization and a device model that executes assigned scripts at discrete
timepoints.
"""

from threading import Semaphore, Lock, Event, Thread

class ReusableBarrier():
    """
    A reusable barrier implementation for synchronizing a fixed number of threads.

    This barrier uses a two-phase protocol to allow multiple uses. Threads wait
    at the barrier by calling the wait() method.
    """
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        @param num_threads The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        # Counters for each phase of the barrier. Using a list to pass by reference.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # Lock to protect access to the counters.
        self.count_lock = Lock()
        # Semaphores to block and release threads for each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to wait at the barrier. Consists of two distinct phases
        to ensure reusability.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier synchronization.

        @param count_threads The counter for the current phase.
        @param threads_sem The semaphore for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # If this is the last thread to arrive, release all waiting threads.
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # All threads wait here until the last one arrives and releases them.
        threads_sem.acquire()

class Device(object):
    """
    Represents a single device (node) in the simulation network.

    Each device manages its own sensor data, executes assigned scripts, and
    communicates with its neighbors under the coordination of a supervisor.
    """
    
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary holding the device's sensor data.
        @param supervisor An object that manages the network topology and neighborhood.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Event to signal the start of a new simulation timepoint.
        self.timepoint_done = Event()


        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        
        # A dictionary of locks, shared among all devices, for location-based data access.
        self.hash_ld = {}

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    
    def exists(self,key):
        """Checks if a key exists in the shared lock dictionary."""
        nr = 0;
        for k in self.hash_ld.keys():
            if(k == key):
                nr += 1


        return nr
    def setup_devices(self, devices):
        """
        Sets up the synchronization objects for a group of devices.

        Functional Utility: Device 0 acts as a coordinator, creating and distributing
        a shared barrier and a shared dictionary of locks to all other devices. This
        centralizes the management of synchronization primitives.
        """
        
        nrd = len(devices)
        if(self.device_id == 0):
            # Device 0 creates the single ReusableBarrier for all devices.
            self.barrier = ReusableBarrier(nrd)
            
            # Distribute the barrier and the shared lock dictionary to other devices.
            for device in devices:
                if(device.device_id != 0):
                    device.barrier = self.barrier
                    device.hash_ld = self.hash_ld

        # Populate the shared lock dictionary with a lock for each unique sensor data location.
        for device in devices:
            for k in device.sensor_data.keys():
                if(self.exists(k) == 0):
                    self.hash_ld[k] = Lock()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device.

        @param script The script object to execute.
        @param location The data location the script operates on.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of the script assignment phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's worker thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The worker thread for a Device, executing its simulation logic.
    """

    def __init__(self, device):
        """Initializes the device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        self.t = []

    
    def method(self,script, script_data):
        """Wrapper method to run a script."""
        result = script.run(script_data)
        return result

    
    def method_scripts(self,neighbours,script,location):
        """
        Core data processing logic for a single script.

        Functional Utility: This method ensures synchronized access to sensor data for a
        specific 'location'. It acquires a lock, gathers data from this device and its
        neighbors, executes the script, and then broadcasts the result back to all
        involved devices.
        """
         # Acquire a lock for the specific data location to prevent race conditions.
         self.device.hash_ld[location].acquire()
         script_data = []
         # Gather data from all neighboring devices.
         for device in neighbours:
             data = device.get_data(location)
             if data is not None:
                 script_data.append(data)
         # Gather data from the current device.
         data = self.device.get_data(location)
         if data is not None:
             script_data.append(data)

         # Run the script and update data on all involved devices with the result.
         if script_data != []:
             result = self.method(script,script_data)
             for device in neighbours:
                 device.set_data(location, result)
             self.device.set_data(location, result)
         # Release the lock for the data location.
         self.device.hash_ld[location].release()

    def run(self):
        """The main simulation loop for the device thread."""
        while True:
            # Retrieve the current set of neighbors for this timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals simulation end by returning None.
                break

            # Wait for the supervisor to signal the start of the next timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Process all assigned scripts for this timepoint, potentially in parallel.
            self.t = []
            number = len(self.device.scripts)
            number_of_threads = min(8, number) # Limits concurrency
            nr = 1
            for (script, location) in self.device.scripts:
                # Create threads to execute script logic.
                self.t.append(Thread(target = self.method_scripts, args = (neighbours,script,location)))
                # Batch execution of script threads.
                if(nr == number_of_threads):
                    for i in range(0,nr):
                        self.t[i].start()
                    for i in range(0,nr):
                        self.t[i].join()
                    self.t = []
                    nr = 0
                nr += 1

            # Wait at the barrier to synchronize with all other devices before the next timepoint.
            self.device.barrier.wait()
