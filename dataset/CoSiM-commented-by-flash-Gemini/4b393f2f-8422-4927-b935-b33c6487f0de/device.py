"""
@4b393f2f-8422-4927-b935-b33c6487f0de/device.py
@brief This module defines a multi-threaded device simulation framework,
including `Device` for managing sensors and scripts, `DeviceThread` for
coordination, and `WorkerThread` for parallel script execution, utilizing
a `ReusableBarrierCond` for synchronization and `Queue` for task distribution.
"""

import Queue
from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a single device in a simulated distributed system.

    Each device manages its own sensor data, processes scripts using a pool
    of worker threads (`WorkerThread`), and coordinates with a supervisor
    and other devices using threads and barriers.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary representing the sensor data this device holds.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to hold scripts assigned to this device.
        self.timepoint_done = Event() # Event to signal when script assignments for a timepoint are complete.

        self.devices = [] # List to store references to all devices in the system.

        self.event_setup = Event() # Event to signal when global setup (barrier, locks) is complete.

        self.barrier_device = None # Global barrier for synchronizing devices, initialized by setup_devices.

        self.locations_lock = [] # List of locks for each data location, shared across devices.

        self.data_set_lock = Lock() # Lock for protecting access to the device's sensor_data during updates.

        self.thread = DeviceThread(self) # Main thread for device operations.
        self.thread.start()

        self.device_shutdown_order = False # Flag to signal worker threads for shutdown.

        self.work_queue = Queue.Queue() # Queue to hold tasks (scripts and their context) for worker threads.

        self.worker_barrier = ReusableBarrierCond(8) # Barrier for synchronizing worker threads within this device.

        self.data_semaphore = Semaphore(value=0) # Semaphore to signal workers that new data (tasks) are available.

        self.worker_semaphore = Semaphore(value=0) # Semaphore for workers to signal completion of a task.

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures global shared resources like the main barrier and location locks.

        This method is typically called by the supervisor during initial setup,
        ensuring all devices share the same synchronization primitives.

        Args:
            devices (list): A list of all Device instances in the simulated system.
        """
        # Block Logic: Only device 0 (the 'master' device) performs the global initialization.
        if self.device_id == 0:
            self.barrier_device = ReusableBarrierCond(len(devices)) # Initialize a global barrier for all devices.

            for _ in range(25): # Initialize 25 shared locks for data locations.
                self.locations_lock.append(Lock())

            # Block Logic: Distribute the shared barrier and location locks to all devices.
            for dev in devices:
                dev.devices = devices
                dev.barrier_device = self.barrier_device
                dev.locations_lock = self.locations_lock
                dev.event_setup.set() # Signal each device that its setup is complete.

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific data location.

        Args:
            script (object): The script object to be executed.
            location (int): The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script to the device's list.
        else:
            self.timepoint_done.set() # If no script, signal that script assignment for this timepoint is complete.

    def get_data(self, location):
        """
        Retrieves data from the device's sensor_data at the specified location.

        Args:
            location (int): The index or key of the data to retrieve.

        Returns:
            any: The data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates data in the device's sensor_data at the specified location.

        Ensures thread-safe access to `sensor_data` during updates.

        Args:
            location (int): The index or key of the data to set.
            data (any): The new data value.
        """
        with self.data_set_lock: # Acquire lock to protect `sensor_data`.
            if location in self.sensor_data:
                self.sensor_data[location] = data # Update the data if the location exists.

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device, waiting for its main thread to finish.
        """
        self.thread.join() # Wait for the main DeviceThread to complete its execution.


class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for coordinating script execution.

    It fetches neighbors, manages timepoint synchronization, dispatches scripts
    to worker threads via a queue, and handles device shutdown.
    """
    

    def __init__(self, device):
        """
        Initializes a DeviceThread instance.

        Args:
            device (Device): The parent Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It waits for initial setup, then continuously fetches neighbors,
        collects assigned scripts, dispatches them to worker threads,
        and participates in global and local barrier synchronization.
        Finally, it manages the shutdown sequence for its worker threads.
        """
        # Pre-condition: Wait until the device's global setup (barrier, locks) is complete.
        self.device.event_setup.wait()

        # Block Logic: Create and start worker threads.
        list_threads = []
        for i in range(8): # Creates 8 worker threads.
            thrd = WorkerThread(self.device, self.device.locations_lock, self.device.work_queue, i)
            list_threads.append(thrd)

        for thrd in list_threads:
            thrd.start()

        script_number = 0 # Counter for the total number of scripts processed.

        while True:
            # Functional Utility: Fetch the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # If no neighbors are returned (e.g., supervisor signals shutdown), break the loop.

            # Block Logic: Wait until scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Distribute assigned scripts to the worker threads via the work queue.
            for (script, location) in self.device.scripts:
                tup = (script, location, neighbours) # Package script context.

                self.device.work_queue.put(tup) # Put the task into the queue.
                self.device.data_semaphore.release() # Signal a worker that a new task is available.

                script_number += 1 # Increment total script count.

            self.device.timepoint_done.clear() # Reset the event for the next timepoint.

            # Block Logic: Synchronize all devices at the global barrier after script assignment.
            self.device.barrier_device.wait()

        # Block Logic: During shutdown, wait for all processed scripts to be completed by workers.
        for _ in xrange(script_number): # Using xrange for Python 2 compatibility.
            self.device.worker_semaphore.acquire() # Wait for each worker to signal completion.

        # Functional Utility: Signal worker threads to shut down.
        self.device.device_shutdown_order = True

        # Block Logic: Release the data semaphore enough times to unblock all worker threads for shutdown.
        for _ in xrange(8): # Release semaphore for each worker thread.
            self.device.data_semaphore.release()

        # Block Logic: Wait for all worker threads to join (terminate).
        for thrd in list_threads:
            thrd.join()


class WorkerThread(Thread):
    """
    A worker thread for a Device, responsible for processing scripts from a shared queue.

    Each worker thread retrieves tasks (script, location, neighbors) from the queue,
    collects data, executes the script, updates relevant data, and signals completion.
    """
    

    def __init__(self, device, locations_lock, work_queue, worker_id):
        """
        Initializes a WorkerThread instance.

        Args:
            device (Device): The parent Device instance this thread belongs to.
            locations_lock (list): A list of shared locks for data locations.
            work_queue (Queue.Queue): The shared queue from which to retrieve tasks.
            worker_id (int): A unique identifier for this worker thread.
        """
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.device = device
        self.locations_lock = locations_lock
        self.work_queue = work_queue
        self.worker_id = worker_id

    def run(self):
        """
        The main execution loop for the WorkerThread.

        It continuously waits for tasks (signaled by `data_semaphore`),
        processes them (collects data, executes script, updates data),
        and then signals task completion (`worker_semaphore`).
        It terminates upon receiving a shutdown signal.
        """
        while True:
            # Pre-condition: Acquire `data_semaphore` to wait for a new task.
            self.device.data_semaphore.acquire()

            # Pre-condition: Check for shutdown signal.
            if self.device.device_shutdown_order is True:
                break # Exit loop if shutdown is ordered.

            # Functional Utility: Retrieve a task from the work queue.
            tup = self.work_queue.get()
            script = tup[0]
            location = tup[1]
            neighbours = tup[2]

            # Block Logic: Acquire location-specific lock for data consistency during script execution.
            with self.locations_lock[location]:
                script_data = [] # List to accumulate data for the current script.
                
                # Block Logic: Collect data from neighboring devices.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collect data from the current device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: If there is data to process, execute the script.
                if script_data != []:
                    # Action: Execute the script.
                    result = script.run(script_data)

                    # Block Logic: Update data in neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Update data in the current device.
                    self.device.set_data(location, result)
            
            # Functional Utility: Signal that this worker has completed a task.
            self.device.worker_semaphore.release()

        # Block Logic: After exiting the main loop (shutdown), synchronize with other worker threads.
        self.device.worker_barrier.wait()
