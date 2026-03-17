"""
This module provides a simulation framework for a network of devices.

It appears to be a combination of multiple files, defining classes for device
simulation (`Device`, `DeviceThread`), a reusable barrier (`ReusableBarrier`),
and a custom thread pool (misleadingly named `MyQueue`). The simulation
involves devices executing scripts on data from themselves and their neighbors,
and synchronizing at each time step using the barrier.
"""

from Queue import Queue 
from threading import Thread

class MyQueue():
    """A thread pool for executing device scripts.

    Despite its name, this class does not implement a queue data structure.
    Instead, it manages a pool of worker threads that consume tasks from an
    internal `Queue.Queue` object.
    """
    
    def __init__(self, num_threads):
        """Initializes the thread pool and starts the worker threads.

        Args:
            num_threads (int): The number of worker threads to create.
        """
        self.queue = Queue(num_threads)
        self.threads = []
        self.device = None  # The parent device is assigned later.

        # Block Logic: Creates and starts the specified number of worker threads.
        for _ in xrange(num_threads):
            thread = Thread(target=self.run)
            self.threads.append(thread)
        
        for thread in self.threads:
            thread.start()
    
    def run(self):
        """The main execution loop for a worker thread.

        A worker continuously fetches a task, which consists of a script and its
        context (neighbors, location). It gathers data, executes the script, and
        propagates the results back to the relevant devices.
        """
        while True:
            # A task is a tuple of (neighbors, script, location).
            neighbours, script, location = self.queue.get()

            # A (None, None, None) tuple is the signal to terminate the worker.
            if neighbours is None and script is None:
                self.queue.task_done()
                return
        
            script_data = []
            
            # Block Logic: Aggregates data from neighboring devices.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            # Block Logic: Appends data from the local device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Functional Utility: Executes the script with the collected data.
                result = script.run(script_data)
                
                # Block Logic: Distributes the result to all neighboring devices.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                # Block Logic: Updates the result on the local device.
                self.device.set_data(location, result)
            
            self.queue.task_done()
    
    def finish(self):
        """Shuts down the thread pool gracefully.

        Waits for all tasks to be completed and then sends a termination
        signal to each worker thread.
        """
        self.queue.join()

        for _ in xrange(len(self.threads)):
           self.queue.put((None, None, None))

        for thread in self.threads:
            thread.join()


# The following classes appear to be concatenated from other modules.

from threading import Thread, Event, Lock, Semaphore
from MyQueue import MyQueue

class ReusableBarrier():
    """A reusable barrier for synchronizing a fixed number of threads.

    This implementation uses a two-phase protocol with two semaphores to ensure
    that threads from one "wave" do not overlap with threads from the next.
    """
    
    def __init__(self, num_threads):
        """
        Args:
            num_threads (int): The number of threads that will synchronize on this barrier.
        """
        self.num_threads = num_threads
        # Counters are stored in a list to be mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
 
    def wait(self):
        """Causes a thread to wait at the barrier. Consists of two phases."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the counter for the current phase.
            threads_sem (Semaphore): The semaphore for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                # The last thread to arrive releases all other waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()

class Device(object):
    """Represents a single node in the distributed device simulation."""
    
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of the device's local sensor data.
            supervisor (object): The central supervisor managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.location_locks = {location: Lock() for location in self.sensor_data}
        self.scripts_available = False
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the shared barrier for all devices in the simulation.

        Args:
            devices (list): A list of all Device objects.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to the device for later execution.

        Args:
            script (object): The script object to be run.
            location (any): The data location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            # A None script signals the end of a timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Acquires a lock and returns data from a specific location.

        Note: This method acquires a lock but does not release it, creating
        a potential for deadlocks if not handled carefully by the caller.

        Args:
            location (any): The key for the desired data.

        Returns:
            The data at the given location, or None.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]     
        else:
            return None

    def set_data(self, location, data):
        """Sets data at a specific location and releases the corresponding lock.

        Note: This method assumes a lock for the location has already been acquired.

        Args:
            location (any): The key for the data.
            data (any): The new value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
        else:
            return None

    def shutdown(self):
        """Waits for the device's main thread to complete."""
        self.thread.join()

class DeviceThread(Thread):
    """The main control loop for a device, managing simulation time-steps."""

    def __init__(self, device):
        """
        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = MyQueue(8)

    def run(self):
        """The main simulation loop.
        
        This loop continuously gets neighbors, processes scripts for a time-step,
        and synchronizes with other devices at a barrier.
        """
        self.queue.device = self.device

        while True:
            # Gets the list of neighbors for the current time-step.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # This inner loop processes all scripts for a single time-step.
            while True:
                # The logic waits for either new scripts or a timepoint completion signal.
                if self.device.scripts_available or self.device.timepoint_done.wait():
                    if self.device.scripts_available:
                        self.device.scripts_available = False
                        
                        # Adds all assigned scripts to the thread pool queue.
                        for (script, location) in self.device.scripts:
                            self.queue.queue.put((neighbours, script, location))
            
                    else:
                        # The timepoint_done event was set.
                        self.device.timepoint_done.clear()
                        self.device.scripts_available = True
                        break
            
            # Invariant: Wait for all tasks in the current step to be processed.
            self.queue.queue.join()
            # Invariant: All devices must synchronize at the barrier before the next step.
            self.device.barrier.wait()
        
        self.queue.finish()
