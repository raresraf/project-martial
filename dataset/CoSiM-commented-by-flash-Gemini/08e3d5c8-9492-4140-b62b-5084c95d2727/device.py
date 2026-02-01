"""
This module implements a distributed device simulation framework, focusing on concurrent script
execution and robust synchronization using a semaphore-based reusable barrier.

Algorithm:
- Device: Represents a simulated physical device with sensor data and script processing capabilities.
- ParallelScript: A worker thread responsible for executing a single script on collected data
  and updating device states, ensuring data consistency with location-specific locks.
- DeviceThread: The orchestrating thread for each `Device`, which distributes scripts to `ParallelScript`
  instances and manages overall synchronization using shared barriers and locks.
- ReusableBarrierSem: A semaphore-based synchronization primitive ensuring all participating threads
  reach a specific point before proceeding.
- Producer-Consumer Pattern: `DeviceThread` acts as a producer, adding tasks (scripts) to a shared
  list (`to_procces`) and signaling `sem_scripts`, while `ParallelScript` instances act as
  consumers, retrieving tasks from this list.
"""

from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a simulated device in a distributed environment.

    Each device has a unique ID, manages its sensor data, and can receive
    and execute scripts. It uses shared locks and reusable barriers for
    thread-safe operations and synchronization with other devices.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings,
                            where keys represent locations.
        supervisor (Supervisor): A reference to the central supervisor managing devices.
        script_received (threading.Event): Event to signal when new scripts are assigned.
        scripts (list): A list to temporarily store assigned scripts (script, location) tuples.
        timepoint_done (threading.Event): (Unused in provided code snippet, but declared).
        thread (DeviceThread): The dedicated orchestrating thread for this device.
        time_bar (ReusableBarrierSem): A shared barrier for synchronizing devices
                                       at the end of a timepoint.
        script_bar (ReusableBarrierSem): A shared barrier for synchronizing devices
                                        after scripts have been assigned.
        devloc (list): A list of `threading.Lock` objects, one for each data location,
                       to provide fine-grained access control.
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
        # Event for signaling that the processing for a timepoint is done (currently unused).
        self.timepoint_done = Event()

        # Create and start a dedicated orchestrating thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

        self.time_bar = None # Will be set by setup_devices.
        self.script_bar = None # Will be set by setup_devices.
        # List of location-specific locks, initialized by setup_devices.
        self.devloc = []

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization primitives (barriers and location-specific locks)
        across all devices. This method ensures these resources are only initialized once
        by the device with `device_id == 0`.

        Pre-condition: This method should be called by a central entity (e.g., supervisor)
                       after all devices are instantiated.
        Invariant: If `device_id` is 0, new `ReusableBarrierSem` instances are created
                   for `time_bar` and `script_bar`. A list of `threading.Lock` objects
                   (`devloc`) is initialized for a range of possible data locations.
                   These shared resources are then propagated to all `Device` instances.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Only the device with ID 0 initializes shared resources.
        if self.device_id == 0:
            # Initialize two reusable barriers for different synchronization points.
            self.time_bar = ReusableBarrierSem(len(devices))
            self.script_bar = ReusableBarrierSem(len(devices))

            # Propagate the created barriers to all devices.
            for device in devices:
                device.time_bar = self.time_bar
                device.script_bar = self.script_bar

            # Determine the maximum location ID across all devices to size the devloc list.
            maxim = 0
            for device in devices:
                loc_list = device.sensor_data.keys()
                # Ensure loc_list is not empty before attempting to sort and access last element.
                if loc_list:
                    loc_list.sort() # Sort to find the maximum location.
                    if loc_list[-1] > maxim:
                        maxim = loc_list[-1]

            # Initialize location-specific locks up to the maximum location ID found.
            while maxim >= 0:
                self.devloc.append(Lock())
                maxim -= 1

            # Propagate the created location locks to all devices.
            for device in devices:
                device.devloc = self.devloc


    def assign_script(self, script, location):
        """
        Assigns a script to be executed on data at a specific location for this device.
        If `script` is None, it signals that script assignments for the current timepoint
        are complete and then waits on `script_bar` for other devices.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (int): The data location where the script should be applied.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that script assignment for the current timepoint is complete.
            self.script_received.set()
            # Wait on the script barrier, ensuring all devices have finished assigning scripts.
            self.script_bar.wait()

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
        Joins the device's dedicated orchestrating thread, effectively waiting for it
        to complete its execution before the program exits.
        """
        self.thread.join()


class ParallelScript(Thread):
    """
    A worker thread responsible for executing a single script on collected data
    and updating device states.

    It acquires a location-specific lock, gathers data from the device and its neighbors,
    executes the script, updates the data, and then releases the lock.

    Attributes:
        device_thread (DeviceThread): The `DeviceThread` instance that created this worker.
    """

    def __init__(self, device_thread):
        """
        Initializes a new ParallelScript instance.

        Args:
            device_thread (DeviceThread): The parent `DeviceThread` that created this worker.
        """
        Thread.__init__(self)
        self.device_thread = device_thread

    def run(self):
        """
        The main execution loop for the ParallelScript worker.

        Invariant: The loop continuously tries to process script tasks from the
                   `device_thread.to_procces` list until a sentinel `None` task is received.
                   Each script execution involves acquiring a location-specific lock,
                   collecting data, executing the script, updating data, and
                   then releasing the lock.
        """
        while True:
            # Acquire a semaphore permit, which is released by DeviceThread when a script is available.
            self.device_thread.sem_scripts.acquire()

            # Retrieve the script task from the shared list.
            nod = self.device_thread.to_procces[0]

            # Remove the processed task from the list.
            del self.device_thread.to_procces[0]

            # Block Logic: Check for a sentinel value to terminate the worker thread.
            if nod is None:
                break

            neighbours, script, location = nod[0], nod[1], nod[2]

            # Acquire a location-specific lock to ensure exclusive access for data at this location.
            self.device_thread.device.devloc[location].acquire()

            script_data = []

            # Block Logic: Collect data from neighboring devices for the current location.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            # Collect data from the current device for the current location.
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: If data is available, execute the script and update results.
            if script_data != []:
                # Execute the script with the collected data.
                result = script.run(script_data)

                # Update the sensor data in neighboring devices.
                for device in neighbours:
                    device.set_data(location, result)
                # Update the sensor data in the current device.
                self.device_thread.device.set_data(location, result)

            # Release the location-specific lock.
            self.device_thread.device.devloc[location].release()


class DeviceThread(Thread):
    """
    The dedicated orchestrating thread for a `Device` object.

    This thread is responsible for:
    1. Interacting with the supervisor to get neighbor information.
    2. Managing a pool of `ParallelScript` worker threads.
    3. Distributing scripts to the worker pool.
    4. Synchronizing with other `DeviceThread` instances using shared barriers.

    Attributes:
        device (Device): The Device object associated with this thread.
        sem_scripts (threading.Semaphore): Semaphore to control access to `to_procces`
                                            and signal script availability to workers.
        numar_procesoare (int): The number of `ParallelScript` worker threads to create.
        pool (list): A list of `ParallelScript` worker thread instances.
        to_procces (list): A shared list acting as a queue for script tasks to be processed
                           by `ParallelScript` workers.
    """

    def create_pool(self, device_thread):
        """
        Creates and starts a pool of `ParallelScript` worker threads.

        Args:
            device_thread (DeviceThread): The parent `DeviceThread` instance.

        Returns:
            list: A list of the created `ParallelScript` thread instances.
        """
        pool = []
        # Block Logic: Create and start a specified number of ParallelScript worker threads.
        for _ in xrange(self.numar_procesoare): # xrange for Python 2 compatibility.
            aux_t = ParallelScript(device_thread)
            pool.append(aux_t)
            aux_t.start()
        return pool

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        # Initialize the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Semaphore to signal script availability to worker threads.
        self.sem_scripts = Semaphore(0)
        self.numar_procesoare = 8 # Number of worker threads.
        # Create and start the pool of worker threads.
        self.pool = self.create_pool(self)
        # Shared list to hold script tasks for worker threads.
        self.to_procces = []

    def run(self):
        """
        The main execution loop for the DeviceThread.

        Invariant: The loop continues until the supervisor signals termination.
                   Within each iteration, it retrieves neighbor information,
                   waits for new script assignments, pushes these scripts
                   into a shared list, signals worker threads, and then
                   synchronizes with other `DeviceThread` instances via shared barriers.
                   Upon termination, it signals and waits for worker threads to exit.
        """
        while True:
            # Retrieve information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: If no neighbors are returned (e.g., simulation termination signal),
            # signal worker threads to terminate and break the loop.
            if neighbours is None:
                # Send sentinel values to all worker threads.
                for _ in range(self.numar_procesoare):
                    self.to_procces.append(None)
                    self.sem_scripts.release()
                # Wait for all worker threads to join.
                for item in self.pool:
                    item.join()
                break

            # Wait for the `script_received` event, which signals that script assignments
            # for the current timepoint are complete and ready for processing.
            self.device.script_received.wait()

            # Block Logic: Populate the shared list (`to_procces`) with all assigned scripts
            # and signal worker threads.
            for (script, location) in self.device.scripts:
                nod = (neighbours, script, location)
                self.to_procces.append(nod)
                self.sem_scripts.release() # Release semaphore to notify worker.

            # Wait on the script barrier, ensuring all DeviceThreads have distributed their scripts.
            self.device.script_bar.wait()

            # Wait on the time barrier, ensuring all DeviceThreads complete their processing
            # for the current timepoint.
            self.device.time_bar.wait()

            # Clear the `script_received` event, resetting it for the next timepoint.
            self.device.script_received.clear()
            # Clear the list of scripts after processing.
            self.device.scripts = [] # Functional utility: Clear the processed scripts list.
