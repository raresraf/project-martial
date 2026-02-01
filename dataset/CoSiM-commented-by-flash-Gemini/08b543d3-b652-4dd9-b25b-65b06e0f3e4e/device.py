"""
This module implements a distributed device simulation framework, emphasizing concurrent script
execution and robust synchronization using a condition-variable-based reusable barrier.

Algorithm:
- Device: Represents a simulated physical device with sensor data and script processing capabilities.
- ScriptWorker: A worker thread responsible for executing a single script on collected data and updating
  device states, ensuring data consistency with shared locks.
- DeviceThread: The orchestrating thread for each `Device`, which distributes scripts to `ScriptWorker`
  instances and manages overall synchronization using shared barriers and locks.
- ReusableBarrierCond: A condition-variable-based synchronization primitive ensuring all participating threads
  reach a specific point before proceeding.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

class Device(object):
    """
    Represents a simulated device in a distributed environment.

    Each device has a unique ID, manages its sensor data, and can receive
    and execute scripts. It uses shared locks and a reusable barrier for
    thread-safe operations and synchronization with other devices.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings,
                            where keys represent locations.
        supervisor (Supervisor): A reference to the central supervisor managing devices.
        scripts (list): A list to store assigned scripts (script, location) tuples.
        timepoint_done (threading.Event): Event to signal completion of script assignments for a timepoint.
        lock (threading.Lock): A general lock for protecting modifications to `sensor_data`.
        threads (list): A list containing the single `DeviceThread` instance for this device.
        barrier (ReusableBarrierCond): A shared reusable barrier for synchronizing all devices.
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
        # List to hold (script, location) tuples assigned to this device.
        self.scripts = []
        # Event for signaling that script assignments for a timepoint are complete.
        self.timepoint_done = Event()
        # General lock for protecting access to device's sensor data.
        self.lock = Lock()
        # Create and start a dedicated orchestrating thread for this device.
        self.threads = [DeviceThread(self)]
        for thread in self.threads:
            thread.start()
        self.barrier = None # Will be set by setup_devices.

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
        This method is typically called once by the device with `device_id == 0`
        to create the barrier, which is then shared among all participating devices.

        Pre-condition: This method should be called by a central entity (e.g., supervisor)
                       after all devices are instantiated.
        Invariant: If the current device is the designated initializer, a new `ReusableBarrierCond`
                   is created, and then propagated to all `Device` instances.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Only the device with ID 0 initializes the shared barrier.
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            # Propagate the created barrier to all devices.
            for device in devices:
                device.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed on data at a specific location for this device.
        If `script` is None, it signals that script assignments for the current timepoint are complete.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (int): The data location where the script should be applied.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that script assignment for the current timepoint is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (int): The location from which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location
                 does not exist in the device's sensor_data.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

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
        Joins the device's dedicated orchestrating thread, effectively waiting for it
        to complete its execution before the program exits.
        """
        for thread in self.threads:
            thread.join()

class ScriptWorker(Thread):
    """
    A worker thread responsible for executing a single script on collected data
    and updating device states.

    It gathers data from the current device and its neighbors, executes the script,
    updates data in all relevant devices (current and neighbors), and then
    synchronizes with other `ScriptWorker` instances via a local barrier.

    Attributes:
        device (Device): The Device object associated with this worker thread.
        script (Script): The script object to execute.
        location (int): The data location relevant to this script execution.
        neighbours (list): A list of neighboring Device objects.
        barrier (ReusableBarrierCond): A shared barrier for synchronizing
                                       `ScriptWorker` instances within a timepoint.
    """

    def __init__(self, data):
        """
        Initializes a new ScriptWorker instance.

        Args:
            data (dict): A dictionary containing 'device', 'script', 'location',
                         'neighbours', and 'barrier' for this worker.
        """
        Thread.__init__(self)
        self.device = data['device']
        self.script = data['script']
        self.location = data['location']
        self.neighbours = data['neighbours']
        self.barrier = data['barrier']

    def run(self):
        """
        The main execution method for the ScriptWorker.

        Invariant: Gathers data from the current device and its neighbors, executes
                   the assigned script, and then updates the sensor data in all
                   involved devices (current and neighbors) in a thread-safe manner
                   using `device.lock`. Finally, it waits on a local barrier.
        """
        script_data = []

        # Block Logic: Collect data from neighboring devices for the current location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Collect data from the current device for the current location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)


        # Block Logic: If data is available, execute the script and update results.
        if script_data != []:
            # Execute the script with the collected data.
            result = self.script.run(script_data)

            # Block Logic: Update the sensor data in neighboring devices in a thread-safe manner.
            for device in self.neighbours:
                # Acquire the device's general lock before setting data.
                device.lock.acquire()
                device.set_data(self.location, result)
                device.lock.release() # Release the lock.

            # Update the sensor data in the current device in a thread-safe manner.
            self.device.lock.acquire()
            self.device.set_data(self.location, result)
            self.device.lock.release() # Release the lock.

        # Wait on the local barrier, synchronizing with other ScriptWorkers for this timepoint.
        self.barrier.wait()


class DeviceThread(Thread):
    """
    The dedicated orchestrating thread for a `Device` object.

    This thread is responsible for:
    1. Interacting with the supervisor to get neighbor information.
    2. Waiting for script assignments for a timepoint.
    3. Spawning `ScriptWorker` instances to execute scripts concurrently.
    4. Managing the synchronization of `ScriptWorker` instances via a local barrier.
    5. Synchronizing with other `DeviceThread` instances using a shared global barrier.

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

        Invariant: The loop continues until the supervisor signals termination.
                   Within each iteration, it retrieves neighbor information,
                   waits for script assignments, creates and manages a pool
                   of `ScriptWorker` instances to execute these scripts,
                   synchronizes these workers, and then synchronizes with
                   other `DeviceThread` instances via a global barrier.
        """
        while True:
            # Retrieve information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: If no neighbors are returned (e.g., simulation termination signal), break the loop.
            if neighbours is None:
                break

            # Wait for the `timepoint_done` event, which signals that script assignments
            # for the current timepoint are complete and ready for processing.
            self.device.timepoint_done.wait()

            no_scripts = len(self.device.scripts)

            # Create a local barrier for synchronizing the ScriptWorker instances for this timepoint.
            # The count is no_scripts + 1 because the DeviceThread itself will also wait on it.
            worker_barrier = ReusableBarrierCond(no_scripts + 1)
            workers = []

            # Block Logic: Create a ScriptWorker for each assigned script.
            for (script, location) in self.device.scripts:
                workers.append(
                ScriptWorker(
                {
                'device' : self.device,
                'script' : script,
                'location' : location,
                'neighbours' : neighbours,
                'barrier' : worker_barrier
                }
                ))

            # Block Logic: Start all ScriptWorker instances.
            for worker in workers:
                worker.start()

            # Wait on the local worker barrier, ensuring all ScriptWorkers complete before proceeding.
            worker_barrier.wait()

            # Block Logic: Join all ScriptWorker instances to ensure their completion.
            for worker in workers:
                worker.join()

            # Clear the `timepoint_done` event, resetting it for the next timepoint.
            self.device.timepoint_done.clear()
            # Clear the list of scripts after processing.
            self.device.scripts = [] # Functional utility: Clear the processed scripts list.

            # Synchronize with other DeviceThreads using the shared global barrier, ensuring
            # all devices complete their timepoint processing before proceeding.
            self.device.barrier.wait()
