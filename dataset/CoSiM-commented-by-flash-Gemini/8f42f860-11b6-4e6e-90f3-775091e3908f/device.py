


import threading
from threading import Thread
from Queue import Queue
from cond_barrier import ReusableBarrier


"""
This module implements a simulation framework for distributed devices,
focusing on concurrent execution of scripts and synchronized data processing.
It utilizes a `Device` class to represent each simulated entity,
a `DeviceThread` for main control flow, and `WorkerThread`s for
executing individual scripts from a shared queue. A `ReusableBarrier`
facilitates global synchronization across devices.
"""

import threading
from threading import Thread
from Queue import Queue # Note: In Python 3, this would be 'queue'
from cond_barrier import ReusableBarrier # External dependency: Assumes ReusableBarrier is defined elsewhere.


class Device(object):
    """
    Represents a simulated device within a distributed system.

    Each device manages its sensor data, interacts with a central supervisor,
    and dispatches scripts for execution to a pool of worker threads.
    It provides thread-safe access to its sensor data via location-specific locks
    and participates in global synchronization through a `ReusableBarrier`.
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
        # Dictionary of Locks, one per data location, to protect sensor data during synchronized access.
        self.location_locks = {loc: threading.Lock() for loc in self.sensor_data}
        self.supervisor = supervisor
        self.scripts = [] # List to temporarily store scripts assigned to this device.

        # The main thread for the device, responsible for supervisor interaction and dispatching scripts.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Queue for incoming script assignments for this device.
        self.scripts_queue = Queue()
        
        # Queue for jobs (scripts) dispatched to worker threads for execution.
        self.workers_queue = Queue()

        
        self.barrier = None # Reference to the global barrier used for synchronizing all devices.


    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the setup of global synchronization primitives (a global barrier).

        This method identifies device 0 (root device) which initializes and distributes
        the `ReusableBarrier` to all other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Only device with ID 0 acts as the coordinator for global setup.
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices)) # Creates a global barrier for all devices.
            # Distributes the same barrier instance to all devices.
            for device in devices:
                device.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script for execution at a specific data location by placing it into a queue.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the sensor data location the script operates on.
        """
        # Puts the script and its location into the device's incoming scripts queue.
        self.scripts_queue.put((script, location))

    def get_data(self, location):
        """
        Retrieves sensor data for a given location without acquiring a lock.

        Note: This method is intended for use in contexts where external locking is handled,
        or where read-only access is sufficient without strict synchronization.

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def get_data_synchronize(self, location):
        """
        Retrieves sensor data for a given location, acquiring a location-specific lock.

        The lock is acquired here and is expected to be released by `set_data_synchronize`.
        This implies a specific usage pattern where `get_data_synchronize` and
        `set_data_synchronize` are called in pairs within a critical section.

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or None if not found.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire() # Acquires the lock for this location.
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location without releasing a lock.

        Note: This method is intended for use in contexts where external locking is handled,
        or where write operations do not require strict synchronization.

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new sensor data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def set_data_synchronize(self, location, data):
        """
        Sets or updates sensor data for a given location, and releases the location-specific lock.

        Pre-condition: A lock for 'location' must have been previously acquired
                       by a call to `get_data_synchronize(location)`.

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new sensor data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release() # Releases the lock for this location.

    def shutdown(self):
        """
        Performs a graceful shutdown of the main device thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for fetching neighbor information
    and orchestrating script execution using a pool of `WorkerThread`s.

    It acts as a dispatcher, moving scripts from the device's incoming queue to
    a shared queue for worker threads, and participates in global synchronization.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The `Device` instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.stop = False # Flag to signal thread termination (unused in provided snippet).

    def run(self):
        """
        Main execution loop for the DeviceThread.

        Architectural Intent: Creates and manages a pool of `WorkerThread`s,
        continuously fetches neighbor information, dispatches scripts to the
        worker pool, waits for script completion, and participates in global
        barrier synchronization. Handles graceful shutdown of workers.
        """
        
        num_workers = 16 # Defines the size of the worker thread pool.
        
        workers = [] # List to keep track of active `WorkerThread` instances.
        
        # `workers_queue`: A shared queue for jobs (scripts) to be processed by `WorkerThread`s.
        # This queue is passed to each worker, allowing them to pull jobs.
        workers_queue = Queue()

        
        # Block Logic: Creates and starts a pool of `WorkerThread`s.
        for i in range(num_workers):
            workers.append(WorkerThread(self.device, i, workers_queue))
        for worker in workers:
            worker.start()

        while True:
            # Block Logic: Fetches the current list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Termination Condition: If no neighbors are returned (None), it signifies
            # that the simulation is ending for this device, and the loop breaks.
            if neighbours is None:
                break

            
            # Filters out itself from the neighbors list if present.
            neighbours = [x for x in neighbours if x != self.device]
            # Block Logic: Updates the `neighbours` list for each worker thread.
            # This ensures workers have the latest neighbor information for data collection.
            for worker in workers:
                worker.neighbours = neighbours

            # Block Logic: Dispatches scripts that were initially stored in `self.device.scripts`.
            # These scripts are added to the shared `workers_queue`.
            for script in self.device.scripts:
                workers_queue.put(script)

            # Block Logic: Continuously pulls scripts from the device's incoming `scripts_queue`.
            # This allows new script assignments to be processed asynchronously.
            while True:
                script, location = self.device.scripts_queue.get()
                # Termination Condition: If a `None` script is received from `scripts_queue`,
                # it signifies the end of incoming script assignments for this timepoint.
                if script is None:
                    self.device.scripts_queue.task_done() # Signals queue that this `None` job is done.
                    break
                
                self.device.scripts.append((script, location)) # Appends the script to the device's internal script list.
                workers_queue.put((script, location)) # Puts the script into the `workers_queue` for execution.
                self.device.scripts_queue.task_done() # Signals `scripts_queue` that this script is handled.

            
            # Functional Utility: Blocks until all jobs in the `workers_queue` have been gotten and processed.
            workers_queue.join()
            
            # Functional Utility: Participates in the global barrier. This ensures all devices
            # are synchronized before proceeding to the next simulation step.
            self.device.barrier.wait()

        # Block Logic: Initiates graceful shutdown of worker threads after the main loop terminates.
        for worker in workers:
            workers_queue.put((None, None)) # Sends termination signal (None, None) to each worker.
        for worker in workers:
            worker.join() # Waits for each worker thread to complete its execution.


class WorkerThread(Thread):
    """
    A worker thread responsible for executing scripts obtained from a shared queue.

    Each `WorkerThread` continuously retrieves jobs (scripts) from its queue,
    collects necessary data (from local and neighboring devices), executes the
    assigned script, and updates data, ensuring thread-safe data access.
    """

    def __init__(self, device, worker_id, queue):
        """
        Initializes a WorkerThread.

        Args:
            device (Device): The `Device` instance this worker is associated with.
            worker_id (int): A unique identifier for this worker thread.
            queue (Queue): The shared queue from which to retrieve jobs (scripts).
        """
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.device = device
        self.neighbours = [] # List of neighboring devices, updated by `DeviceThread`.
        self.worker_id = worker_id
        self.queue = queue # Shared queue to get scripts from.

    def run(self):
        """
        Main execution loop for the WorkerThread.

        Architectural Intent: Continuously retrieves scripts from the shared queue.
        For each script, it collects data from itself and neighbors using
        synchronized access methods, executes the script, and updates data.
        Handles termination gracefully upon receiving a `None` job.
        """
        while True:
            # Block Logic: Retrieves a job (script, location) tuple from the queue. This call blocks until a job is available.
            script, location = self.queue.get()
            # Termination Condition: If a `None` script is received, it signals the worker thread to terminate gracefully.
            if script is None:
                self.queue.task_done() # Signals to the queue that this job is finished.
                break

            
            script_data = [] # List to store all data relevant to the script.
            
            # Block Logic: Collects sensor data from neighboring devices.
            for device in self.neighbours:
                # Functional Utility: Retrieves data from neighbor device, acquiring location-specific lock.
                data = device.get_data_synchronize(location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Collects sensor data from its own device.
            # Functional Utility: Retrieves data from its own device, acquiring location-specific lock.
            data = self.device.get_data_synchronize(location)
            if data is not None:
                script_data.append(data)

            
            # Block Logic: If any data was collected, executes the script and updates data.
            if script_data != []:
                
                result = script.run(script_data) # Functional Utility: Executes the assigned script with the collected data.

                
                # Block Logic: Updates data on neighboring devices.
                for device in self.neighbours:
                    device.set_data_synchronize(location, result) # Updates data and releases location-specific lock.
                
                # Block Logic: Updates data on its own device.
                self.device.set_data_synchronize(location, result) # Updates data and releases location-specific lock.
            self.queue.task_done()
