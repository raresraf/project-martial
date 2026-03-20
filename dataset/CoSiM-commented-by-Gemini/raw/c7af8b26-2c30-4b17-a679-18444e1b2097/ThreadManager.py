"""
A framework for managing a pool of worker threads to simulate a distributed system.

This module contains classes for managing thread pools (`ThreadManager`),
synchronization (`ConditionalBarrier`), and simulating network devices (`Device`, `DeviceThread`).
It appears to be designed for a discrete-time simulation where devices execute scripts
on sensor data, communicate with neighbors, and synchronize at time steps.
Note: The use of `from Queue import Queue` indicates this code is for Python 2.
"""

from Queue import Queue
from threading import Thread


class ThreadManager(object):
    """Manages a pool of worker threads to execute tasks from a queue."""
    
    def __init__(self, threads_count):
        """Initializes the thread manager and starts the worker threads.

        Args:
            threads_count (int): The number of worker threads to create in the pool.
        """
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = None
        self.initialize_workers(threads_count)
    
    def create_workers(self, threads_count):
        """Creates the worker threads.

        Args:
            threads_count (int): The number of worker threads.
        """
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)
    
    def start_workers(self):
        """Starts the worker threads."""
        for thread in self.threads:
            thread.start()
    
    def initialize_workers(self, threads_count):
        """Initializes and starts the worker threads.

        Args:
            threads_count (int): The number of worker threads.
        """
        self.create_workers(threads_count)
        self.start_workers()
    
    def set_device(self, device):
        """Sets the device context for the tasks.

        Args:
            device (Device): The device object associated with this thread manager.
        """
        self.device = device
    
    def execute(self):
        """The target function for worker threads, processing tasks from the queue."""
        while True:
            neighbours, script, location = self.queue.get()
            no_neighbours = neighbours is None
            no_scripts = script is None
            if no_neighbours and no_scripts:
                # Sentinel value received, indicating thread should terminate.
                self.queue.task_done()
                return
            self.run_script(neighbours, script, location)
            self.queue.task_done()
    
    @staticmethod
    def is_not_empty(given_object):
        """Checks if a given object is not None.

        Args:
            given_object: The object to check.

        Returns:
            bool: True if the object is not None, False otherwise.
        """
        return given_object is not None
    
    def run_script(self, neighbours, script, location):
        """Executes a script on data gathered from the device and its neighbors.

        Args:
            neighbours (list): A list of neighboring Device objects.
            script (Script): The script to execute.
            location (str): The location from which to get sensor data.
        """
        script_data = []
        
        # Gather data from neighbors
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if ThreadManager.is_not_empty(data):
                    script_data.append(data)
        
        # Gather data from the current device
        data = self.device.get_data(location)
        if ThreadManager.is_not_empty(data):
            script_data.append(data)
        
        # If any data was collected, run the script and distribute the result
        if script_data:
            result = script.run(script_data)
            
            # Distribute result to neighbors
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    continue
                device.set_data(location, result)
            
            # Set result on the current device
            self.device.set_data(location, result)
    
    def submit(self, neighbours, script, location):
        """Submits a new task to the queue.

        Args:
            neighbours (list): A list of neighboring Device objects.
            script (Script): The script to execute.
            location (str): The location for data gathering.
        """
        self.queue.put((neighbours, script, location))
    
    def wait_threads(self):
        """Blocks until all tasks in the queue are processed."""
        self.queue.join()

    def end_threads(self):
        """Gracefully shuts down all worker threads."""
        self.wait_threads()
        
        # Send sentinel values to terminate threads
        for _ in xrange(len(self.threads)):
            self.submit(None, None, None)
            
        for thread in self.threads:
            thread.join()


from threading import Condition


class ConditionalBarrier(object):
    """A reusable barrier implementation for synchronizing a fixed number of threads."""
    
    def __init__(self, num_threads):
        """Initializes the barrier.

        Args:
            num_threads (int): The number of threads to wait for.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
    
    def wait(self):
        """Causes a thread to wait until all threads have reached the barrier."""
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                # Last thread has arrived, notify all waiting threads
                self.cond.notify_all()
                self.count_threads = self.num_threads
            else:
                self.cond.wait()


from threading import Event, Thread, Lock

# These imports suggest the classes were intended to be in separate files.
# from ThreadManager import ThreadManager
# from barriers import ConditionalBarrier


class Device(object):
    """Represents a single device in the simulated network."""
    
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of sensor data for the device.
            supervisor (Supervisor): A supervisor object that manages the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        self.script_received = Event()
        self.timepoint_done = Event()
        
        self.scripts = []
        self.scripts_arrived = False
        
        self.barrier = None
        self.location_locks = {location: Lock() for location in sensor_data}
        
        self.thread = DeviceThread(self)
        self.thread.start()
    
    def __str__(self):
        return "Device %d" % self.device_id
    
    def assign_barrier(self, barrier):
        """Assigns a synchronization barrier to the device.

        Args:
            barrier (ConditionalBarrier): The barrier to use for synchronization.
        """
        self.barrier = barrier
    
    def setup_devices(self, devices):
        """Sets up the synchronization barrier for all devices.
        
        This method appears to be intended to be called by one device (device 0)
        to create and distribute a barrier to all other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        number_of_devices = len(devices)
        if self.device_id == 0:
            self.assign_barrier(ConditionalBarrier(number_of_devices))
            self.broadcast_barrier(devices, self.barrier)
    
    @staticmethod
    def broadcast_barrier(devices, barrier):
        """Broadcasts the barrier to all devices.

        Args:
            devices (list): The list of all devices.
            barrier (ConditionalBarrier): The barrier to broadcast.
        """
        for device in devices:
            if device.device_id == 0:
                continue
            device.assign_barrier(barrier)
    
    def accept_script(self, script, location):
        """Accepts a script to be executed on the device.

        Args:
            script (Script): The script object.
            location (str): The data location for the script.
        """
        self.scripts.append((script, location))
        self.scripts_arrived = True
    
    def assign_script(self, script, location):
        """Assigns a script or signals the end of a timepoint.

        Args:
            script (Script): The script to assign, or None.
            location (str): The data location.
        """
        if script is not None:
            self.accept_script(script, location)
        else:
            self.timepoint_done.set()
    
    def get_data(self, location):
        """Gets sensor data from a specific location, with locking.

        Args:
            location (str): The location of the sensor data.

        Returns:
            The sensor data, or None if the location is invalid.
        """
        data_is_valid = location in self.sensor_data
        if data_is_valid:
            self.location_locks[location].acquire()
        return self.sensor_data[location] if data_is_valid else None
    
    def set_data(self, location, data):
        """Sets sensor data at a specific location, with locking.

        Args:
            location (str): The location to update.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
    
    def shutdown(self):
        """Shuts down the device thread."""
        self.thread.join()

class DeviceThread(Thread):
    """The main execution thread for a Device."""
    
    def __init__(self, device):
        """Initializes the device thread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadManager(8)
    


    def run(self):
        """The main loop of the device thread.
        
        Orchestrates the device's participation in the simulation, including
        script execution and synchronization.
        """
        self.thread_pool.set_device(self.device)
        while True:
            # Get neighbors for the current simulation step
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals end of simulation
                break
            
            # Wait for scripts for the current timepoint
            while True:
                scripts_ready = self.device.scripts_arrived
                done_waiting = self.device.timepoint_done.wait()
                if scripts_ready or done_waiting:
                    if done_waiting and not scripts_ready:
                        self.device.timepoint_done.clear()
                        self.device.scripts_arrived = True
                        break
                    self.device.scripts_arrived = False
                    
                    # Submit received scripts to the thread pool for execution
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit(neighbours, script, location)
            
            # Wait for this device's scripts to finish for the current timepoint
            self.thread_pool.wait_threads()
            
            # Synchronize with all other devices before proceeding to the next timepoint
            self.device.barrier.wait()
        
        # End of simulation, shut down the thread pool
        self.thread_pool.end_threads()
