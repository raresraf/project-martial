"""
This module implements a distributed device simulation framework using Python threading
and a condition-variable-based reusable barrier for synchronization.

Algorithm:
- ReusableBarrier: A synchronization primitive that ensures all participating threads
  reach a specific point before any can proceed. It is designed to be reusable.
- Device: Represents a simulated entity managing sensor data and script execution.
- DeviceThread: The dedicated thread for each device, responsible for orchestrating
  script execution and synchronizing with other devices using shared resources.
"""

from threading import Event, Thread, Condition, Lock

class ReusableBarrier(object):
    """
    Implements a reusable barrier synchronization mechanism using a Condition variable.

    This barrier allows a specified number of threads to wait until all have
    reached a certain point, after which all are released simultaneously. It
    is designed for repeated use without reinitialization.

    Attributes:
        num_threads (int): The total number of threads expected to participate in the barrier.
        count_threads (int): The current count of threads waiting at the barrier.
        cond (threading.Condition): The condition variable used for signaling and waiting.
    """

    def __init__(self, num_threads):
        """
        Initializes a new instance of the ReusableBarrier.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can pass.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        # Initialize a Condition variable, which internally uses a Lock.
        self.cond = Condition()

    def wait(self):
        """
        Causes the calling thread to wait until all `num_threads` threads
        have reached this barrier.

        Pre-condition: The thread holds the `cond` lock implicitly before waiting.
        Invariant: The `count_threads` is atomically decremented. When it reaches zero,
                   all waiting threads are notified, and the `count_threads` is reset.
        """
        self.cond.acquire()
        try:
            self.count_threads -= 1
            # If this thread is the last to reach the barrier:
            if self.count_threads == 0:
                self.cond.notify_all() # Release all waiting threads.
                self.count_threads = self.num_threads # Reset for next use.
            else:
                self.cond.wait() # Wait until signaled by the last thread.
        finally:
            self.cond.release()


class Device(object):
    """
    Represents a simulated device in a distributed environment.

    Each device has a unique ID, manages its sensor data, and interacts
    with a supervisor. It is capable of receiving and executing scripts
    on its data, coordinating with other devices using shared synchronization
    primitives (reusable barrier and locks).

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings,
                            where keys represent locations.
        supervisor (Supervisor): A reference to the central supervisor managing devices.
        script_received (threading.Event): Event to signal when new scripts are assigned.
        scripts (list): A list to store assigned scripts (script, location) tuples.
        timepoint_done (threading.Event): Event to signal completion of a timepoint's tasks.
        thread (DeviceThread): The dedicated orchestrating thread for this device.
        barr (ReusableBarrier): A shared barrier for synchronizing all devices.
        lock (threading.Lock): A shared lock for protecting concurrent access to resources.
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
        # Event for signaling when new scripts have been received.
        self.script_received = Event()
        # List to hold (script, location) tuples assigned to this device.
        self.scripts = []
        # Event for signaling that the processing for a timepoint is done.
        self.timepoint_done = Event()
        # Create and start a dedicated thread for this device's operations.
        self.thread = DeviceThread(self)
        self.barr = None # Will be set by setup_devices.
        self.lock = None # Will be set by setup_devices.
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
        Initializes shared synchronization primitives (barrier and a general lock)
        across all devices. This method ensures these resources are only initialized once
        by the device with the lowest ID.

        Pre-condition: This method should be called by a central entity (e.g., supervisor)
                       after all devices are instantiated.
        Invariant: If the current device is the designated initializer, a new `ReusableBarrier`
                   and `Lock` are created, and then propagated to all `Device` instances.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Only the device with the lowest ID initializes shared resources.
        if devices[0].barr is None and devices[0].device_id == self.device_id:
            bariera = ReusableBarrier(len(devices))
            lock = Lock()
            # Propagate the created barrier to all devices.
            for i in devices:
                i.barr = bariera
            # Propagate the created lock to all devices.
            for j in devices:
                j.lock = lock

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
            # Signal that a new script has been received.
            self.script_received.set()
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
        Joins the device's dedicated thread, effectively waiting for it to complete
        its execution before the program exits.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The dedicated orchestrating thread for a `Device` object.

    This thread is responsible for:
    1. Interacting with the supervisor to get neighbor information.
    2. Waiting for script assignments.
    3. Executing assigned scripts, potentially updating device states.
    4. Synchronizing with other `DeviceThread` instances using a shared barrier.

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
                   waits for the current timepoint's tasks to be ready,
                   executes the scripts sequentially, and then synchronizes
                   with other `DeviceThread` instances via a barrier.
        """
        while True:
            # Retrieve information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: If no neighbors are returned (e.g., simulation termination signal), break the loop.
            if neighbours is None:
                break

            # Wait for the `timepoint_done` event, which signals that script assignments
            # for the current timepoint are complete and ready for processing by this device.
            self.device.timepoint_done.wait()

            # Block Logic: Iterate through each assigned script and execute it.
            # This section processes scripts sequentially.
            for (script, location) in self.device.scripts:
                # Acquire the shared lock to protect access to sensor data during script execution.
                self.device.lock.acquire()
                script_data = []

                # Block Logic: Collect data from neighboring devices for the current location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Collect data from the current device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: If there is data collected, execute the script and update results.
                if script_data != []:
                    # Execute the script with the collected data.
                    result = script.run(script_data)

                    # Update the sensor data in neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    # Update the sensor data in the current device.
                    self.device.set_data(location, result)
                self.device.lock.release() # Release the shared lock.

            # Clear the `timepoint_done` event, resetting it for the next timepoint.
            self.device.timepoint_done.clear()
            # Clear the scripts list for the next timepoint.
            self.device.scripts = []

            # Synchronize with other DeviceThreads using the shared barrier, ensuring
            # all devices complete their timepoint processing before proceeding.
            self.device.barr.wait()
