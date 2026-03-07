"""
This module implements a sophisticated distributed device simulation framework.

It utilizes a thread pool pattern to manage concurrent script execution, ensuring
efficient use of resources. Synchronization between devices is handled by a shared
reusable barrier, while data integrity is maintained through fine-grained,
location-specific locks. This design is robust, scalable, and represents a
mature implementation of the simulation concept. The code appears to be written
for Python 2.
"""
from threading import Event, Thread, Lock
# The following imports suggest modularization, but the classes are defined below
# or are missing from this context. This may reflect a previous file structure.
from barrier import ReusableBarrierCond
from thread_pool import ThreadPool

class Device(object):
    """
    Represents a device in the simulation network.

    This class manages a device's state, including its sensor data and assigned
    scripts, and coordinates with other devices via shared synchronization objects.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): A dictionary of the device's local sensor data.
            supervisor (obj): The supervisor object managing network topology.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        
        # The main synchronization barrier for all devices.
        self.barrier = None

        
        
        # A dictionary of locks, one for each data location, for fine-grained locking.
        self.location_locks = {}

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization primitives.

        The first device to call this method creates a shared barrier and a set of
        location-based locks, then distributes them to all other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        
        if self.barrier is None:
            
            # Assumes ReusableBarrierCond is a barrier implementation, likely
            # based on threading.Condition, imported from a 'barrier' module.
            self.barrier = ReusableBarrierCond(len(devices))

            
            for device in devices:
                device.barrier = self.barrier

                
                
                # Create and distribute a lock for each unique data location.
                for location in device.sensor_data:
                    if location not in self.location_locks:
                        self.location_locks[location] = Lock()
                
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A None script signals the end of assignments
        for the current time step.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data for a given location, or None if not present."""
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data for a given location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        


        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, managing a thread pool for task execution.
    """
    
    
    # Defines the number of worker threads in the pool.
    NO_CORES = 8

    def __init__(self, device):
        """
        Initializes the DeviceThread and its associated ThreadPool.

        Args:
            device (Device): The parent device this thread controls.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        self.thread_pool = ThreadPool(self.device, DeviceThread.NO_CORES)

    def run(self):
        """
        The main simulation loop.

        It waits for scripts to be assigned, submits them as tasks to the thread pool,
        and then synchronizes with all other devices at a barrier before starting
        the next simulation step.
        """
        
        while True:
            

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals shutdown.
                # Submit sentinel tasks to terminate worker threads gracefully.
                for _ in xrange(DeviceThread.NO_CORES):
                    self.thread_pool.submit_task(None, None, None)
                
                self.thread_pool.end_workers()
                break

            
            # Wait for the signal that all scripts for this step have been assigned.
            self.device.timepoint_done.wait()

            
            # Submit all assigned scripts to the thread pool for execution.
            for (script, location) in self.device.scripts:
                self.thread_pool.submit_task(script, location, neighbours)

            
            # Clear the event for the next time step.
            self.device.timepoint_done.clear()

            
            
            # Wait at the barrier for all devices to complete their tasks for this step.
            self.device.barrier.wait()




from threading import Thread
from Queue import Queue

class Worker(Thread):
    """
    A worker thread that executes tasks from a queue.
    """
    

    def __init__(self, device, task_queue):
        """
        Initializes the worker.

        Args:
            device (Device): The parent device, used to access shared resources like locks.
            task_queue (Queue): The queue from which tasks are fetched.
        """
        
        Thread.__init__(self)
        self.device = device
        self.task_queue = task_queue

    def run(self):
        """Continuously fetches and executes tasks from the queue until a sentinel is received."""
        while True:
            
            # Block until a task is available.
            script, location, neighbours = self.task_queue.get()

            
            # A sentinel value (None, None, None) signals the thread to terminate.
            if (script is None and location is None and neighbours is None):
                self.task_queue.task_done()
                break

            
            
            # Use a location-specific lock to ensure thread-safe data access.
            with self.device.location_locks[location]:
                
                self.run_task(script, location, neighbours)

            
            # Signal that the task is complete.
            self.task_queue.task_done()

    def run_task(self, script, location, neighbours):
        """
        Gathers data, runs the script, and updates data on the local device and its neighbors.
        """
        
        script_data = []
        
        # Gather data from neighbors.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        # Gather local data.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            # Execute the script.
            result = script.run(script_data)

            
            
            # Update data on neighbors and the local device.
            for device in neighbours:
                device.set_data(location, result)

            self.device.set_data(location, result)


class ThreadPool(object):
    """
    A simple thread pool implementation.
    """
    

    def __init__(self, device, no_workers):
        """
        Initializes the thread pool.

        Args:
            device (Device): The parent device, passed to workers.
            no_workers (int): The number of worker threads to create.
        """
        
        self.device = device
        self.no_workers = no_workers
        
        self.task_queue = Queue(no_workers)
        self.workers = []
        self.initialize_workers()

    def initialize_workers(self):
        """Creates and starts the worker threads."""
        
        for _ in xrange(self.no_workers):
            self.workers.append(Worker(self.device, self.task_queue))

        for worker in self.workers:
            worker.start()

    def end_workers(self):
        """Waits for all tasks to be completed and then joins all worker threads."""
        
        self.task_queue.join()

        for worker in self.workers:
            worker.join()

    def submit_task(self, script, location, neighbours):
        """Adds a new task to the task queue."""
        
        self.task_queue.put((script, location, neighbours))
