"""
This module implements a distributed device simulation framework using a producer-consumer pattern
for script execution and a condition-based barrier for synchronization.

Algorithm:
- Producer-Consumer: `DeviceThread` acts as a producer, populating a script queue,
  while `ScriptThread` instances act as consumers, processing scripts from this queue.
- ReusableBarrierCond: Ensures all participating devices reach a specific point before proceeding.
- Data Consistency: Uses locks (`lcks`) for exclusive access to sensor data locations.
"""

import Queue
from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

class ScriptThread(Thread):
    """
    A worker thread responsible for executing assigned scripts on sensor data.

    This thread continuously retrieves scripts from a shared queue (`scripts_queue`)
    managed by its associated `Device` and executes them, ensuring data consistency
    through location-specific locks.

    Attributes:
        device (Device): The Device object associated with this script thread,
                         providing access to the script queue, locks, and data.
    """

    def __init__(self, device):
        """
        Initializes a new ScriptThread instance.

        Args:
            device (Device): The Device object that this thread will serve.
        """
        # Initialize the base Thread class with a descriptive name.
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the ScriptThread.

        Invariant: The loop continuously processes scripts from the device's queue
                   until a sentinel value (None, None) is encountered, signaling termination.
                   Each script execution involves acquiring a lock for the relevant
                   data location, collecting data, executing the script, updating
                   data, and then releasing the lock.
        """
        while True:
            # Retrieve a script and its location from the device's shared queue.
            # This operation blocks until an item is available.
            (script, location) = self.device.scripts_queue.get()

            # Block Logic: Check for a sentinel value to terminate the thread.
            if (script, location) == (None, None):
                # Put the sentinel back for other script threads to terminate.
                self.device.scripts_queue.put((None, None))
                break

            script_data = []

            # Acquire a lock for the specific data location to ensure exclusive access
            # during data collection and modification.
            with self.device.lcks[location]:
                # Block Logic: Collect data from neighboring devices.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Collect data from the current device.
                data = self.device.get_data(location)

                if data is not None:
                    script_data.append(data)

                # Block Logic: If data is available, execute the script and update data.
                if script_data != []:
                    # Execute the script with the collected data.
                    result = script.run(script_data)

                    # Update the sensor data in neighboring devices.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    # Update the sensor data in the current device.
                    self.device.set_data(location, result)

            # Signal that the current task from the queue has been completed.
            self.device.scripts_queue.task_done()


class Device(object):
    """
    Represents a simulated device in a distributed environment.

    Each device has a unique ID, manages its sensor data, and interacts
    with a supervisor. It manages a pool of `ScriptThread` workers to
    execute scripts concurrently and coordinates with other devices
    using a shared barrier and data-specific locks.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings,
                            where keys represent locations.
        supervisor (Supervisor): A reference to the central supervisor managing devices.
        script_received (threading.Event): Event to signal when script assignments are complete.
        scripts (list): A list to temporarily store assigned scripts (script, location) tuples.
        start_scripts (threading.Event): Event to signal that script execution can begin.
        timepoint_done (ReusableBarrierCond): A shared barrier for synchronizing all devices.
        used_barrier (bool): Flag to indicate if the barrier has been initialized for this device.
        neighbours (list): A list of neighboring Device objects.
        scripts_queue (Queue.Queue): A thread-safe queue for scripts to be processed by `ScriptThread`s.
        lcks (dict): A dictionary of `threading.Lock` objects, one for each
                     data location, to ensure exclusive access during data manipulation.
        thread (DeviceThread): The dedicated orchestrating thread for this device.
        thread_pool (list): A list of `ScriptThread` instances forming the worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): Initial sensor data for the device.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Events for managing script assignment and execution flow.
        self.script_received = Event()
        self.scripts = []
        self.start_scripts = Event()
        # Initialize barrier (count set later by setup_devices).
        self.timepoint_done = ReusableBarrierCond(0)
        self.used_barrier = False
        self.neighbours = []

        # Thread-safe queue for scripts to be processed by ScriptThreads.
        self.scripts_queue = Queue.Queue()
        # Dictionary to hold location-specific locks, initialized by setup_devices.
        self.lcks = {}

        # Create and start the orchestrating thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Initialize the pool of worker ScriptThreads.
        self.thread_pool = []
        self.init_thread_pool(self.thread_pool)

    def init_thread_pool(self, pool):
        """
        Initializes a pool of 8 `ScriptThread` instances and starts them.

        Args:
            pool (list): The list to which the initialized `ScriptThread` objects will be added.
        """
        # Block Logic: Create and append 8 ScriptThread instances to the pool.
        for i in xrange(8): # xrange is Python 2, in Python 3 it would be range
            thread = ScriptThread(self)
            pool.append(thread)

        # Block Logic: Start all threads in the pool.
        for i in xrange(len(pool)):
            pool[i].start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization primitives (location-specific locks and barrier)
        across all devices. This method ensures that these resources are only initialized once.

        Pre-condition: This method should be called by a central entity (e.g., supervisor).
        Invariant: Location-specific locks (`lcks`) are propagated across all devices for
                   shared sensor data. The `timepoint_done` barrier's count is set based on
                   the total number of devices, and the barrier object is shared among them.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Initialize and propagate location-specific locks.
        for i in self.sensor_data.keys():
            # If a lock for this location doesn't exist yet, create it.
            if i not in self.lcks: # has_key is Python 2, in Python 3 it would be 'in'
                self.lcks[i] = Lock()
                # Propagate this newly created lock to all other devices.
                for j in xrange(len(devices)):
                    if devices[j].device_id != self.device_id:
                        devices[j].lcks[i] = self.lcks[i]

        # Block Logic: Initialize and propagate the shared barrier.
        if self.used_barrier is False:
            self.timepoint_done.count_threads = len(devices)
            self.timepoint_done.num_threads = len(devices)
            # Propagate the initialized barrier and mark it as used for all devices.
            for i in xrange(len(devices)):
                devices[i].used_barrier = True
                if devices[i].device_id != self.device_id:
                    devices[i].timepoint_done = self.timepoint_done

    def assign_script(self, script, location):
        """
        Assigns a script to be executed on data at a specific location for this device.
        If `script` is None, it signals that no more scripts are coming for the current timepoint.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (int): The data location where the script should be applied.
        """
        if script is not None:
            # Append the script and its target location to a temporary list.
            self.scripts.append((script, location))
        else:
            # Signal that script assignment for the current timepoint is complete.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (int): The location from which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location
                 does not exist in the device's sensor_data.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        Args:
            location (int): The location at which to set data.
            data (any): The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device by joining its worker
        threads and the orchestrating `DeviceThread`.
        """
        # Block Logic: Wait for all worker threads to complete any pending tasks.
        for i in xrange(len(self.thread_pool)):
            self.thread_pool[i].join()

        # Wait for the orchestrating DeviceThread to complete.
        self.thread.join()

class DeviceThread(Thread):
    """
    Manages the lifecycle and script distribution for a single Device.

    This thread continuously checks for new script assignments, acts as a producer
    by adding scripts to the device's queue, and synchronizes with other device
    threads using a shared barrier. It represents the orchestrator for a device
    in the simulation.

    Attributes:
        device (Device): The Device object associated with this thread.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        # Initialize the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.

        Invariant: The loop continues until the supervisor signals termination
                   by returning None for neighbors. Within each iteration, the
                   thread retrieves neighbor information, waits for new script
                   assignments, pushes these scripts to the device's queue,
                   waits for all queued scripts to be processed, and then
                   synchronizes with other devices via a barrier.
        """
        while True:
            # Retrieve information about neighboring devices from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: If no neighbors are returned (e.g., simulation termination signal),
            # put a sentinel value in the queue for worker threads to terminate and break the loop.
            if self.device.neighbours is None:
                self.device.scripts_queue.put((None, None))
                break

            # Wait for the `script_received` event, which signals that script assignments
            # for the current timepoint are complete and ready for processing.
            self.device.script_received.wait()

            # Clear the `script_received` event, resetting it for the next timepoint.
            self.device.script_received.clear()

            # Block Logic: Populate the scripts queue with all assigned scripts.
            # This acts as the producer part of the producer-consumer pattern.
            for (script, location) in self.device.scripts:
                self.device.scripts_queue.put((script, location))

            # Block Logic: Wait until all scripts currently in the queue have been processed
            # by the worker ScriptThreads. This ensures all work for the current timepoint is done.
            self.device.scripts_queue.join()

            # Invariant: If the barrier is used for this device, wait at the barrier.
            # This synchronizes all DeviceThreads, ensuring they all complete their
            # timepoint processing before proceeding to the next.
            if self.device.used_barrier is True:
                self.device.timepoint_done.wait()
