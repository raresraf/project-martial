

from threading import Event, Thread, Condition, Lock
from Queue import Queue

"""
This module implements a simulation framework for distributed devices,
focusing on concurrent execution of scripts and synchronized data processing.
It utilizes a `Device` class to represent each simulated entity,
a `DeviceThread` for main control flow, and a global `process_scripts` function
that acts as a worker for executing individual scripts from a shared queue.
A `ReusableBarrier` facilitates global synchronization.
"""

from threading import Event, Thread, Condition, Lock
from Queue import Queue # Note: In Python 3, this would be 'queue'


class Device(object):
    """
    Represents a simulated device within a distributed system.

    Each device manages its sensor data, interacts with a central supervisor,
    and dispatches scripts for execution to a pool of worker threads.
    It participates in global synchronization through a `ReusableBarrier`.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor readings
                                (e.g., {location_id: data_value}).
            supervisor (object): An object representing the central supervisor,
                                 used for coordination (e.g., getting neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to temporarily store scripts assigned to this device before dispatching.
        self.locks = {} # Dictionary of Locks, one per data location, to protect sensor data during access.
                                    
        self.no_more_scripts = Event() # Event to signal that no more scripts are being assigned for the current timepoint.
                                            
        self.barrier = None # Reference to the global barrier used for synchronizing all devices.
        # The main thread for the device, responsible for supervisor interaction and dispatching scripts.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the setup of global synchronization primitives (a global barrier).

        This method identifies the "root" device (smallest device_id), which then
        initializes the shared `ReusableBarrier` and distributes it to other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Device with ID 0 acts as the coordinator for global setup.
        # If this device is the root (device_id == 0), it initializes the barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices)) # Creates a global barrier for all devices.

        # Block Logic: Distributes the globally initialized barrier to other devices.
        # Note: This loop assumes `self.barrier` is already set if `self.device_id == 0`.
        # For non-root devices, it expects `self.barrier` to be passed from the root device.
        for device in devices:
            if device is not self: # For every other device, set its barrier.
                device.set_barrier(self.barrier)


    def assign_script(self, script, location):
        """
        Assigns a script for execution at a specific data location.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the sensor data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location)) # Adds the script and its location to a list.
        else:
            # If script is None, it signifies the end of script assignments for the current timepoint.
            # Functional Utility: Signals that no more scripts are being assigned for the current timepoint.
            self.no_more_scripts.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location, acquiring a lock for that location.

        The lock is acquired here and is expected to be released by `set_data`.
        This implies a specific usage pattern where `get_data` and `set_data` are
        called in pairs within a critical section.

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or None if not found.
        """
        if location in self.sensor_data:
            # Functional Utility: Acquires a lock for the specific data location,
            # ensuring exclusive access to the data.
            self.locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location, and releases the lock.

        Pre-condition: A lock for 'location' must have been previously acquired
                       by a call to `get_data(location)`.

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new sensor data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            # Functional Utility: Releases the lock for the specific data location after modification.
            self.locks[location].release()

    
    def set_barrier(self, barrier):
        """
        Sets the global barrier instance for this device.

        This method is used during the setup phase by the root device
        to distribute the globally created barrier to all other devices.

        Args:
            barrier (ReusableBarrier): The shared `ReusableBarrier` instance.
        """
        self.barrier = barrier

    def shutdown(self):
        """
        Performs a graceful shutdown of the main device thread and its child worker threads.
        """
        # Block Logic: Waits for all child worker threads managed by `DeviceThread` to complete.
        for thread in self.thread.child_threads:
            if thread.is_alive(): # Checks if the child thread is still running before joining.
                thread.join()
        # Waits for the main device thread (`DeviceThread`) to complete its execution.
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for coordinating with the supervisor,
    dispatches scripts to a worker pool via a `Queue`, and participates in global synchronization.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The `Device` instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue() # A shared queue for jobs (scripts) to be processed by worker threads.
        self.child_threads = [] # List to keep track of worker threads launched by this `DeviceThread`.
        self.max_threads = 8 # Configures the size of the worker thread pool.


    def run(self):
        """
        Main execution loop for the DeviceThread.

        Architectural Intent: Initializes per-location locks, creates and manages a pool
        of worker threads (`process_scripts`), continuously fetches neighbor information,
        dispatches scripts to the worker pool, and participates in global synchronization.
        Handles graceful shutdown of worker threads.
        """
        # Block Logic: Initializes a `Lock` for each sensor data location of this device.
        # Note: `iteritems()` is Python 2.x specific; in Python 3.x, `items()` should be used.
        for location, data in self.device.sensor_data.iteritems():
            self.device.locks[location] = Lock()

        # Block Logic: Creates and starts a pool of worker threads.
        # Each worker thread runs the `process_scripts` function, consuming jobs from `self.queue`.
        # Note: `xrange` is Python 2.x specific; in Python 3.x, `range` should be used.
        for i in xrange(self.max_threads):
            thread = Thread(target=process_scripts, args=(self.queue,))
            self.child_threads.append(thread)
            thread.start()

        # Block Logic: Main simulation loop.
        while True:
            # Block Logic: Fetches the current list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            # Termination Condition: If no neighbors are returned (None), it signifies
            # that the simulation is ending for this device. It then initiates the
            # graceful shutdown of its worker threads.
            if neighbours is None:
                # Signals all worker threads to terminate by putting `None` into the queue.
                # Each worker thread is designed to exit its loop when it receives `None`.
                for i in xrange(len(self.child_threads)):
                    self.queue.put(None)
                # Functional Utility: Blocks until all items in the queue have been gotten and processed.
                self.queue.join()
                break # Exits the main loop.

            done_scripts = 0 # Counter for scripts already processed from the `self.device.scripts` list.
            
            # Block Logic: Dispatches scripts to the worker thread pool.
            # It iterates through the scripts assigned to this device and puts them as jobs into the queue.
            for (script, location) in self.device.scripts:
                
                job = {} # Constructs a job dictionary for the worker thread.
                job['script'] = script
                job['location'] = location
                job['device'] = self.device
                job['neighbours'] = neighbours
                self.queue.put(job) # Puts the job into the shared queue for workers to process.
                done_scripts += 1 # Increments the count of dispatched scripts.

            # Block Logic: Waits for the `no_more_scripts` event to be set, indicating that
            # all scripts for the current timepoint have been assigned.
            self.device.no_more_scripts.wait()
            
            # Functional Utility: Clears the `no_more_scripts` event to prepare for the next timepoint.
            self.device.no_more_scripts.clear()
            
            # Block Logic: If more scripts were added while waiting for `no_more_scripts`,
            # dispatch those remaining scripts. This handles scripts added concurrently.
            if done_scripts < len(self.device.scripts):
                for (script, location) in self.device.scripts[done_scripts:]:
                    
                    job = {}
                    job['script'] = script
                    job['location'] = location
                    job['device'] = self.device
                    job['neighbours'] = neighbours
                    self.queue.put(job)     

            # Functional Utility: Blocks until all jobs (scripts) dispatched in this timepoint
            # have been processed by the worker threads.
            self.queue.join()

            # Functional Utility: Participates in the global barrier. This ensures all devices
            # are synchronized before proceeding to the next simulation step.
            self.device.barrier.wait()

def process_scripts(queue):
    """
    Worker function for processing scripts from a shared queue.

    This function is intended to be run by multiple child threads launched by `DeviceThread`.
    It continuously retrieves jobs from the queue, collects necessary data (from local
    and neighboring devices), executes the assigned script, and updates data, ensuring
    thread-safe data access through location-specific locks.

    Args:
        queue (Queue): The shared queue from which to retrieve jobs.
    """
    
    while True:
        # Block Logic: Retrieves a job dictionary from the queue. This call blocks until a job is available.
        job = queue.get()
        
        # Termination Condition: If a `None` job is received, it signals the worker thread to terminate gracefully.
        if job is None:
            queue.task_done() # Signals to the queue that this job is finished (for queue's join method).
            break
        
        script = job['script']
        location = job['location']
        mydevice = job['device']
        neighbours = job['neighbours']

        script_data = [] # List to store all data relevant to the script.
        
        # Block Logic: Collects sensor data from neighboring devices.
        for device in neighbours:
            if device is not mydevice: # Ensures not to collect data from itself as a neighbor.
                data = device.get_data(location) # `get_data` acquires location-specific lock.
                if data is not None:
                    script_data.append(data)
        
        # Block Logic: Collects sensor data from its own device.
        data = mydevice.get_data(location)


        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = script.run(script_data) # Functional Utility: Executes the assigned script with the collected data.

            # Block Logic: Updates data on neighboring devices.
            for device in neighbours:
                if device is not mydevice:
                    device.set_data(location, result) # `set_data` releases location-specific lock.
            
            # Block Logic: Updates data on its own device.
            mydevice.set_data(location, result) # `set_data` releases location-specific lock.
        
        queue.task_done() # Signals to the queue that the current job is finished.


class ReusableBarrier(object):
    """
    A reusable barrier synchronization primitive implemented using a `Condition` variable.

    This barrier allows a fixed number of threads (`num_threads`) to wait for
    each other to reach a common point before any can proceed. It is designed
    to be reusable across multiple synchronization points within a larger simulation loop.
    """
    

    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must arrive
                               at the barrier before any can proceed.
        """
        
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Counter for threads currently waiting at the barrier.
        self.cond = Condition() # A condition variable used for synchronization (waiting and notifying).
                                                 
    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        `num_threads` threads have also called `wait()`.

        The last thread to arrive releases all waiting threads and resets the barrier.
        """
        
        self.cond.acquire() # Acquires the lock associated with the condition variable.
        self.count_threads -= 1 # Decrements the counter of threads yet to arrive.
        if self.count_threads == 0: # Conditional Logic: If this is the last thread to arrive.
            self.cond.notify_all() # Notifies all threads waiting on this condition.
            self.count_threads = self.num_threads # Resets the counter for barrier reusability.
        else: # Conditional Logic: If this is not the last thread to arrive.
            self.cond.wait() # Blocks (waits) until notified by the last arriving thread.
        self.cond.release() # Releases the lock associated with the condition variable.
