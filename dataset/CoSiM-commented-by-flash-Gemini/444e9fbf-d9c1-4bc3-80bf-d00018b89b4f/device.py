

"""
This module provides components for simulating a device within a distributed system.
It features a multi-threaded architecture with a `Device` class managing sensor data
and scripts. It utilizes a `DeviceThread` which employs a `ThreadPool` (from `threadpool.py`)
to execute scripts concurrently. Synchronization among devices is managed through
a `Barrier` (from `barrier.py`) and condition variables, while data access is
protected by per-location semaphores.

Key Components:
- `Device`: Represents an individual simulated device, managing sensor data and scripts.
- `DeviceThread`: The main thread for a device, orchestrating script execution via a thread pool.
- `Worker`: Worker threads used by the `ThreadPool` to execute tasks.
- `ThreadPool`: Manages a pool of worker threads for concurrent task execution.
"""

from threading import Thread, Condition, Semaphore
from barrier import Barrier
from threadpool import ThreadPool

class Device(object):
    """
    Represents an individual simulated device within a distributed system.
    Each `Device` instance manages its own sensor data, processes scripts,
    and coordinates with other devices and a supervisor. It employs per-location
    semaphores for data access control and a `Condition` variable for managing
    script assignment and timepoint progression.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping location IDs to their
                                current sensor data values.
            supervisor (object): A reference to the central supervisor managing
                                 the distributed system.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        # A dictionary of semaphores, one for each location, to protect access to sensor data.
        self.data_semaphores = {loc : Semaphore(1) for loc in sensor_data}
        self.scripts = []             # List to store (script, location) tuples.

        self.new_script = False       # Flag to indicate if new scripts have been assigned.
        self.timepoint_end = False    # Flag to indicate the end of a timepoint.
        self.cond = Condition()       # Condition variable for synchronization related to script assignment.

        self.barrier = None           # Shared barrier for synchronizing device threads.
        self.supervisor = supervisor  # Reference to the supervisor.
        self.thread = DeviceThread(self) # Dedicated thread for this device's operations.

        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared `Barrier` instance across a group of devices.
        The device with `device_id == 0` is responsible for creating this barrier
        and distributing it to all other devices in the group.

        Args:
            devices (list): A list of Device objects that are part of the same group.
        """
        # Block Logic: Device with ID 0 is responsible for initializing the shared barrier.
        if self.device_id == 0:
            self.barrier = Barrier(len(devices)) # Create a new Barrier for the group.
            # Block Logic: Distribute the barrier to other devices.
            for neigh in devices:
                if neigh.device_id != self.device_id:
                    neigh.set_barrier(self.barrier) # Assign the shared barrier to neighbors.

    def set_barrier(self, barrier):
        """
        Sets the shared `Barrier` instance for this device. This method is called
        by the device with `device_id == 0` during setup to distribute the barrier.

        Args:
            barrier (Barrier): The shared barrier instance.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location on this device.
        Notifies waiting threads about new scripts or timepoint end.

        Args:
            script (object or None): The script object to execute, or `None` to
                                     signal the end of script assignment for the timepoint.
            location (object): The identifier for the location associated with the script.
        """
        with self.cond: # Acquire the condition variable's lock.
            if script is not None:
                self.scripts.append((script, location)) # Add script to the list.
                self.new_script = True                 # Set flag for new scripts.
            else:
                self.timepoint_end = True              # Set flag for timepoint end.
            self.cond.notifyAll()                      # Notify any waiting threads in `timepoint_ended`.

    def timepoint_ended(self):
        """
        Waits for a signal indicating either new scripts have been assigned or
        the current timepoint has ended. This method is used by the `DeviceThread`
        to synchronize with script assignments.

        Returns:
            bool: True if the timepoint has ended, False if new scripts are available.
        """
        with self.cond: # Acquire the condition variable's lock.
            # Block Logic: Wait until either new scripts are available or the timepoint ends.
            while not self.new_script and \
                  not self.timepoint_end:
                self.cond.wait() # Release lock and wait for notification.

            # Functional Utility: Determine return based on flags and reset them.
            if self.new_script:
                self.new_script = False
                return False # New scripts are available.
            else:
                self.timepoint_end = False
                self.new_script = len(self.scripts) > 0 # Check if there are still scripts from previous cycles.
                return True # Timepoint has ended.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location, acquiring a semaphore to
        ensure exclusive access during the read operation.

        Args:
            location (object): The identifier of the location for which to retrieve data.

        Returns:
            Any: The sensor data associated with the location, or `None` if not found.
        """
        if location in self.sensor_data:
            self.data_semaphores[location].acquire() # Acquire the semaphore for the location.
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates the sensor data for a given location and releases the associated semaphore.

        Args:
            location (object): The identifier of the location to update.
            data (Any): The new sensor data value for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_semaphores[location].release() # Release the semaphore after updating.

    def shutdown(self):
        """
        Initiates the shutdown process for the device's operational thread.
        This method waits for the device's thread to complete its execution.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    The dedicated operational thread for a `Device` instance. It manages the
    execution of assigned scripts by utilizing a `ThreadPool` to run them
    concurrently. This thread orchestrates the simulation's timepoints,
    fetching neighbor information, submitting scripts for execution,
    and synchronizing with other devices via a shared barrier.
    """
    

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    @staticmethod
    def run_script(own_device, neighbours, script, location):
        """
        Static method to encapsulate the logic for executing a single script.
        It collects data from relevant devices, runs the script, and then
        propagates the results. Data access is controlled by semaphores.

        Args:
            own_device (Device): The Device instance that owns this script.
            neighbours (list): A list of neighboring Device objects.
            script (object): The script object to execute.
            location (object): The identifier for the location the script pertains to.
        """
        script_data = [] # List to accumulate data for the script.

        # Block Logic: Gather data from neighboring devices for the script's location.
        # It ensures not to gather data from itself if it's listed as a neighbor.
        for device in neighbours:
            if device is own_device:
                continue # Skip gathering data from itself if already in neighbours list.
            data = device.get_data(location) # Retrieves data, acquiring semaphore internally.
            if data is not None:
                script_data.append(data)

        # Block Logic: Gather data from the current device itself for the script's location.
        data = own_device.get_data(location) # Retrieves data, acquiring semaphore internally.
        if data is not None:
            script_data.append(data)

        # Pre-condition: Check if any data was collected before running the script.
        if script_data != []:
            # Functional Utility: Execute the script with the collected data.
            result = script.run(script_data)

            # Block Logic: Propagate the result of the script execution back to all neighboring devices.
            for device in neighbours:
                if device is not own_device:
                    device.set_data(location, result) # Updates data, releasing semaphore internally.

            # Block Logic: Update the current device's own sensor data with the result.
            own_device.set_data(location, result) # Updates data, releasing semaphore internally.

    def run(self):
        """
        The main execution loop for the device's thread.
        It initializes a thread pool, enters a continuous simulation loop,
        fetches tasks (scripts), submits them to the pool, and synchronizes
        with other devices using a barrier.
        """
        
        # Functional Utility: Initialize a ThreadPool with a fixed size.
        pool_size = 8
        pool = ThreadPool(pool_size)

        # Block Logic: Main loop for continuous simulation timepoints.
        while True:
            # Pre-condition: Fetch information about neighboring devices from the supervisor.
            # This also serves as a signal for the simulation's continuation or termination.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Exit the loop if the supervisor signals simulation end.

            # Block Logic: Process scripts assigned to this device for the current timepoint.
            offset = 0
            # Invariant: Loop continues as long as the timepoint has not ended and new scripts are available.
            while not self.device.timepoint_ended():
                scripts = self.device.scripts[offset:] # Get new scripts since last check.
                # Block Logic: Add each script as a task to the thread pool.
                for (script, location) in scripts:
                    pool.add_task(DeviceThread.run_script, self.device,
                                  neighbours, script, location)

                # Functional Utility: Update offset to prevent reprocessing already queued scripts.
                offset = len(self.device.scripts)

            # Block Logic: Wait for all tasks submitted to the pool for this timepoint to complete.
            pool.wait()

            # Functional Utility: Synchronize with all other device threads using the shared barrier.
            # All device threads must reach this point before any can proceed to the next timepoint.
            self.device.barrier.wait()

        # Functional Utility: Terminate the thread pool gracefully.
        pool.terminate()


from Queue import Queue
from threading import Thread

class Worker(Thread):
    """
    A worker thread designed to be part of a `ThreadPool`. It continuously fetches
    tasks (functions and their arguments) from a shared queue and executes them.
    It handles potential errors during task execution and signals task completion.
    """
    

    def __init__(self, tasks):
        """
        Initializes a Worker thread.

        Args:
            tasks (Queue): The shared queue from which this worker fetches tasks.
        """
        Thread.__init__(self)
        self.tasks = tasks

    def run(self):
        """
        The main execution loop for the worker thread.
        It continuously retrieves tasks, executes them, and marks them as done.
        """
        while True:
            # Block Logic: Retrieve a task from the shared queue. Blocks until a task is available.
            func, args, kargs = self.tasks.get()
            try:
                # Functional Utility: Execute the retrieved function with its arguments.
                func(*args, **kargs)
            except ValueError:
                # If a ValueError is caught (e.g., from a shutdown signal), exit the loop.
                return
            finally:
                # Always signal that the task is done, regardless of success or failure.
                self.tasks.task_done()

class ThreadPool(object):
    """
    A simple thread pool implementation that manages a fixed number of worker threads.
    It allows submitting tasks for asynchronous execution and provides mechanisms
    to wait for all tasks to complete and to gracefully terminate the pool.
    """
    

    def __init__(self, num_threads):
        """
        Initializes the ThreadPool with a specified number of worker threads.

        Args:
            num_threads (int): The number of worker threads to create in the pool.
        """
        self.tasks = Queue(num_threads) # A bounded queue to hold tasks.
        # Block Logic: Create and initialize worker threads.
        self.workers = [Worker(self.tasks) for _ in range(num_threads)]

        # Block Logic: Start all worker threads.
        for worker in self.workers:
            worker.start()

    def add_task(self, func, *args, **kargs):
        """
        Adds a new task to the thread pool's queue.

        Args:
            func (callable): The function to be executed by a worker.
            *args: Positional arguments to pass to the function.
            **kargs: Keyword arguments to pass to the function.
        """
        self.tasks.put((func, args, kargs)) # Add task as a tuple (function, args, kwargs).

    def wait(self):
        """
        Blocks until all tasks currently in the queue have been processed by the workers.
        """
        self.tasks.join()

    def terminate(self):
        """
        Gracefully terminates all worker threads in the pool.
        It adds special "dummy" tasks that cause workers to exit their loops.
        """
        self.wait() # Ensure all current tasks are finished before terminating.

        def raising_dummy():
            """A dummy function that raises an exception to signal worker shutdown."""
            raise ValueError

        # Block Logic: Add a shutdown signal (dummy task) for each worker.
        for _ in range(len(self.workers)):
            self.add_task(raising_dummy)
        # Block Logic: Wait for all worker threads to fully terminate.
        for worker in self.workers:
            worker.join()
