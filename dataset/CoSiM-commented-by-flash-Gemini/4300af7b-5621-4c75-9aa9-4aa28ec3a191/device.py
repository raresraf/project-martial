

"""
This module defines the components for simulating a device within a distributed system.
It employs a multi-threaded architecture using Python's `threading` module,
`Queue` for task management, and a `ReusableBarrierCond` (assumed to be from `barrier.py`)
for synchronization among multiple devices.

Key components include:
- `Device`: Represents an individual simulated device, managing its sensor data and scripts.
- `MyThread`: Worker threads that execute individual scripts assigned to a device.
- `DeviceThread`: The main thread for a device, managing a pool of `MyThread` workers
                  and orchestrating the execution of scripts and synchronization.
"""

from threading import Event, Thread, Semaphore
from Queue import Queue
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents an individual simulated device in a distributed system.
    Each device manages its own sensor data, communicates with a central
    supervisor, and processes assigned scripts. It utilizes a dedicated
    `DeviceThread` for concurrent operations and participates in synchronization
    with other devices using a shared barrier and location-specific locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping location IDs to their
                                current sensor data values.
            supervisor (Supervisor): A reference to the central supervisor
                                     managing the distributed system.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when a script has been assigned.
        self.scripts = []             # List to store (script, location) tuples.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's tasks.
        # Each device runs its operations within a dedicated thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.locations = []             # List of all unique locations relevant to this device group.
        self.location_locks = None      # Dictionary of Semaphores for location-specific data access.
        self.barrier = None             # Shared barrier for synchronizing device threads.

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization primitives (barrier and location locks)
        across a group of devices. This method ensures that these shared resources
        are created only once and then distributed among the participating devices.

        Args:
            devices (list): A list of Device objects that are part of the same group.
        """
        lock = {} # Temporary dictionary for location locks.
        barrier = ReusableBarrierCond(len(devices)) # Create a new barrier for the group.

        # Pre-condition: Check if the barrier and location_locks have not been initialized yet for this device.
        if self.barrier is None:
            # Functional Utility: Gather all unique locations across all devices in the group.
            self.get_all_locations(devices)
            # Block Logic: Create a Semaphore for each unique location to control access to its data.
            for location in self.locations:
                lock[location] = Semaphore(1) # Initialize with count 1 for mutual exclusion.

            # Block Logic: Distribute the newly created barrier and location locks to all devices in the group.
            for device in devices:
                # Invariant: Only assign if not already set, ensuring single initialization.
                if device.barrier is None and device.location_locks is None:
                    device.barrier = barrier
                    device.location_locks = lock

    def get_all_locations(self, devices):
        """
        Populates the `self.locations` list with all unique location IDs present
        across the sensor data of all devices in the provided list.

        Args:
            devices (list): A list of Device objects to scan for locations.
        """
        # Block Logic: Iterate through each device and its sensor data to find unique locations.
        for device in devices:
            for location in device.sensor_data:
                if location not in self.locations:
                    self.locations.append(location)

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location on this device.
        If `script` is `None`, it signals the end of scripts for the current timepoint.

        Args:
            script (object or None): The script object to execute, or `None`.
            location (str): The identifier for the location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a script has been received.
        else:
            self.timepoint_done.set() # Signal that all scripts for this timepoint have been assigned.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's internal state.

        Args:
            location (str): The identifier of the location for which to retrieve data.

        Returns:
            Any: The sensor data associated with the location, or `None` if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a given location on this device.

        Args:
            location (str): The identifier of the location to update.
            data (Any): The new sensor data value for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device's operational thread.
        This method waits for the device's thread to complete its execution.
        """
        self.thread.join()


class MyThread(Thread):
    """
    A worker thread within a `DeviceThread` that is responsible for executing
    individual scripts. It fetches tasks from a shared queue, acquires
    location-specific locks, gathers data from neighboring devices and its
    own device, executes the assigned script, and propagates the results.
    """

    def __init__(self, device, tasks):
        """
        Initializes a MyThread worker.

        Args:
            device (Device): The Device instance that this worker thread serves.
            tasks (Queue): The shared queue from which this thread fetches tasks.
        """
        Thread.__init__(self, name="MyThread %d" % device.device_id)
        self.device = device
        self.tasks = tasks

    def run(self):
        """
        The main execution loop for the worker thread.
        It continuously retrieves tasks from the queue, processes them,
        and signals task completion.
        """
        while True:
            # Block Logic: Retrieve a task (neighbours, script, location) from the shared queue.
            # This call blocks until a task is available.
            neighbours, script, location = self.tasks.get()

            # Pre-condition: Check if the task is a shutdown signal (neighbours is None).
            if neighbours is None:
                self.tasks.task_done() # Signal that this "shutdown task" is done.
                return # Exit the thread's main loop.

            # Block Logic: Acquire the semaphore for the specific location.
            # This ensures exclusive access to the sensor data for this location
            # while the script is executing and data is being updated.
            self.device.location_locks[location].acquire()

            script_data = [] # List to accumulate data for the script.

            # Block Logic: Gather data from neighboring devices for the script's location.
            # Invariant: Each 'device' in 'neighbours' is a valid Device object.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Gather data from the current device itself for the script's location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Pre-condition: Check if any data was collected before running the script.
            if script_data != []:
                # Functional Utility: Execute the script with the collected data.
                result = script.run(script_data)

                # Block Logic: Propagate the script's result back to all neighboring devices.
                # Invariant: Each 'device' in 'neighbours' will have its data updated.
                for device in neighbours:
                    device.set_data(location, result)
                
                # Block Logic: Update the current device's own sensor data with the result.
                self.device.set_data(location, result)

            # Block Logic: Release the semaphore for the specific location.
            self.device.location_locks[location].release()

            # Functional Utility: Signal to the queue that the current task is complete.
            self.tasks.task_done()

class DeviceThread(Thread):
    """
    The primary operational thread for a `Device` instance. It manages a pool
    of `MyThread` worker threads to execute scripts concurrently, orchestrates
    the simulation flow by interacting with the supervisor, queues scripts for
    execution, and ensures synchronization with other devices at key timepoints
    using a shared barrier.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.max_threads = 8 # Maximum number of worker threads to use concurrently.
        # A Queue to hold tasks for worker threads, with a size limit.
        self.tasks = Queue(self.max_threads)

        self.thread_list = [] # List to store references to worker threads.
        # Block Logic: Create and populate the pool of worker threads.
        for _ in range(self.max_threads):
            self.thread_list.append(MyThread(self.device, self.tasks))

    def run(self):
        """
        The main execution loop for the device's thread.
        It starts worker threads, enters a continuous loop to fetch tasks
        (scripts), queues them for its workers, waits for their completion,
        and synchronizes with other device threads. It handles shutdown.
        """
        # Block Logic: Start all worker threads.
        for thread in self.thread_list:
            thread.start()

        # Block Logic: Main simulation loop for processing timepoints.
        while True:
            # Pre-condition: Fetch neighbors from the supervisor. This also acts as a signal
            # for simulation continuation or termination.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # If neighbours is None, it's a shutdown signal from the supervisor.
                # Block Logic: Put 'None' tasks into the queue for each worker to signal them to shut down.
                for _ in self.thread_list:
                    self.tasks.put((None, None, None))
                self.tasks.join() # Wait for all shutdown tasks to be processed by workers.
                break # Exit the main simulation loop.

            # Block Logic: Wait until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Queue up all assigned scripts for the worker threads to process.
            # Each script becomes a task (neighbours, script_object, location) in the queue.
            for (script, location) in self.device.scripts:
                self.tasks.put((neighbours, script, location))

            # Block Logic: Wait for all currently queued tasks (scripts) to be completed by worker threads.
            self.tasks.join()

            # Functional Utility: Synchronize with other device threads using the shared barrier.
            # All device threads must reach this point before any can proceed.
            self.device.barrier.wait()

            # Functional Utility: Clear the timepoint_done event, resetting it for the next timepoint.
            self.device.timepoint_done.clear()

        # Block Logic: Join all worker threads to ensure they have finished execution before the main thread exits.
        for thread in self.thread_list:
            thread.join()
        self.thread_list = [] # Clear the list of worker threads.
