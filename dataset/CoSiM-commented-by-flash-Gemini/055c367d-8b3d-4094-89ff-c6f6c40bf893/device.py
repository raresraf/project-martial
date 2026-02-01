
"""
This module implements a simulation framework for distributed devices,
focusing on concurrent script execution and synchronization.
It defines Device, Helper, and DeviceThread classes, utilizing a thread pool
for parallel processing and a condition-based reusable barrier for synchronization.

Algorithm:
- Device: Represents a distributed entity with sensor data and communication capabilities.
- Helper: Manages a thread pool to concurrently execute scripts on device data.
- DeviceThread: Orchestrates device operations, script execution, and synchronization.
- ReusableBarrierCond: A synchronization primitive ensuring all participating devices
  reach a specific point before proceeding.
"""

from threading import Event, Thread, Lock
from multiprocessing.dummy import Pool as ThreadPool
from reusablebarrier import ReusableBarrierCond

class Device(object):
    """
    Represents a simulated device in a distributed environment.

    Each device has a unique ID, manages its sensor data, and interacts
    with a supervisor. It is capable of receiving and executing scripts
    on its data, coordinating with other devices using a shared barrier
    and data-specific locks.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings,
                            where keys represent locations.
        supervisor (Supervisor): A reference to the central supervisor managing devices.
        script_received (threading.Event): Event to signal when new scripts are available.
        scripts (list): A list to store assigned scripts (script, location) tuples.
        timepoint_done (threading.Event): Event to signal completion of a timepoint's tasks.
        barrier (ReusableBarrierCond): A shared barrier for synchronizing all devices.
        data_locks (dict): A dictionary of `threading.Lock` objects, one for each
                           data location, to ensure exclusive access during data manipulation.
        thread (DeviceThread): The dedicated thread for this device's operations.
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

        # Events for signaling and managing script execution flow.
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None  # Initialized by setup_devices
        # Create a lock for each data location to protect concurrent access.
        self.data_locks = {}
        for location in sensor_data:
            self.data_locks[location] = Lock()
        # Create and start a dedicated thread for this device's operations.
        self.thread = DeviceThread(self)
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
        Initializes the shared synchronization barrier across all devices.
        This method is typically called once by a designated device (e.g., device with ID 0)
        to set up the common barrier for all participating devices.

        Pre-condition: This method should be called once by the orchestrator.
        Invariant: If `device_id` is 0, a new `ReusableBarrierCond` is created
                   and then propagated to all other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Only the device with ID 0 is responsible for initializing
        # the shared barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            # Propagate the initialized barrier to all other devices.
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed on data at a specific location for this device.
        If `script` is None, it signals that no more scripts are coming for the current
        timepoint, and sets the `timepoint_done` event.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (int): The data location where the script should be applied.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Signal that a new script has been received.
            self.script_received.set()
        else:
            # Signal that script assignment for the current timepoint is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.
        This method acquires a lock before returning data.

        Args:
            location (int): The location from which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location
                 does not exist in the device's sensor_data.
        """
        if location in self.sensor_data:
            # Acquire the lock for the specific data location.
            self.data_locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location and releases the corresponding lock.

        Args:
            location (int): The location at which to set data.
            data (any): The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            # Release the lock for the specific data location after modification.
            self.data_locks[location].release()

    def shutdown(self):
        """
        Joins the device's dedicated thread, effectively waiting for it to complete
        its execution before the program exits.
        """
        self.thread.join()

class Helper(object):
    """
    Manages the concurrent execution of scripts for a Device using a thread pool.

    This class encapsulates the logic for distributing scripts to a pool of worker
    threads and collecting their results, ensuring that script execution is
    parallelized and efficient.

    Attributes:
        device (Device): The Device object for which this helper is managing scripts.
        pool (multiprocessing.dummy.Pool): A thread pool used for concurrent script execution.
        neighbours (list): A list of neighboring Device objects from which to fetch data.
        scripts (list): A list of (script, location) tuples to be executed.
    """

    def __init__(self, device):
        """
        Initializes a new Helper instance.

        Args:
            device (Device): The Device object associated with this helper.
        """
        self.device = device
        # Initialize a thread pool with a fixed size of 8 threads.
        self.pool = ThreadPool(8)
        self.neighbours = None
        self.scripts = None

    def set_neighbours_and_scripts(self, neighbours, scripts):
        """
        Sets the neighbors and scripts to be processed by the helper.

        Args:
            neighbours (list): A list of neighboring Device objects.
            scripts (list): A list of (script, location) tuples.
        """
        self.neighbours = neighbours
        self.scripts = scripts

    def script_run(self, script_location_tuple):
        """
        Executes a single script on the data at a specified location.
        This method is designed to be run by a worker thread in the thread pool.

        Args:
            script_location_tuple (tuple): A tuple containing (script, location).
        """
        script, location = script_location_tuple
        script_data = []
        # Block Logic: Collect data from neighboring devices, excluding itself.
        for device in self.neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        # Collect data from the current device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        # Block Logic: If data is available, execute the script and update data in devices.
        if script_data != []:
            result = script.run(script_data)
            # Update data in neighboring devices, excluding itself.
            for device in self.neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)
            # Update data in the current device.
            self.device.set_data(location, result)

    def run(self):
        """
        Asynchronously maps the `script_run` method to the assigned scripts
        using the thread pool.
        """
        self.pool.map_async(self.script_run, self.scripts)

    def close_pool(self):
        """
        Closes the thread pool and waits for all active threads to complete.
        """
        self.pool.close()
        self.pool.join()


class DeviceThread(Thread):
    """
    Manages the lifecycle and script execution for a single Device.

    This thread continuously checks for new scripts, utilizes a `Helper` to
    process them concurrently, and synchronizes with other device threads
    using a shared barrier. It acts as the main processing loop for a device
    in the simulation.

    Attributes:
        device (Device): The Device object associated with this thread.
        helper (Helper): An instance of the Helper class to manage concurrent script execution.
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
        self.helper = None # Initialized within run method

    def run(self):
        """
        The main execution loop for the DeviceThread.

        Invariant: The loop continues until the supervisor signals termination
                   by returning None for neighbors. Within each iteration, the
                   thread waits for either `script_received` or `timepoint_done`
                   events, processes scripts using the Helper, and then synchronizes
                   with other devices via a barrier.
        """
        while True:
            # Retrieve information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If no neighbors are returned (e.g., simulation termination signal), break the loop.
            if neighbours is None:
                break
            # Initialize the Helper for concurrent script execution.
            self.helper = Helper(self.device)

            # Block Logic: Loop to handle script reception and timepoint completion.
            while True:
                # Pre-condition: Either new scripts are received or the timepoint is marked as done.
                if (self.device.script_received.is_set() or
                self.device.timepoint_done.is_set()):

                    # Block Logic: If scripts are received, process them.
                    if self.device.script_received.is_set():
                        self.device.script_received.clear() # Clear the event after processing.
                        # Set neighbors and scripts for the helper to process.
                        self.helper.set_neighbours_and_scripts(neighbours,
							self.device.scripts)
                        self.helper.run() # Start asynchronous script execution.
                    # Block Logic: If timepoint is done, break inner loop to synchronize.
                    else:
                        self.device.timepoint_done.clear() # Clear the event.
                        self.device.script_received.set() # Set script_received for next cycle.
                        break
            # Close the thread pool managed by the helper.
            self.helper.close_pool()
            # Synchronize with other DeviceThreads using the shared barrier.
            self.device.barrier.wait()
