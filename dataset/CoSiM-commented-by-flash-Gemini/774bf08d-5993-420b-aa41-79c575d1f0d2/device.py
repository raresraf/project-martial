


"""
This module defines a distributed device simulation framework, featuring
`Device` and `DeviceThread` classes, along with a custom `ReusableBarrierCond`
for synchronization. It manages device-specific data, script execution across
multiple threads, inter-device communication, and uses threading primitives
like Events and Locks for coordination.
"""

from threading import Event, Thread, Condition, Lock

class ReusableBarrierCond(object):
    """
    Implements a reusable barrier using a `Condition` variable.

    This barrier allows a fixed number of threads to wait for each other
    at a synchronization point and then proceed together. It can be
    reused multiple times for subsequent synchronization points.
    """

    def __init__(self, num_threads):
        """
        Initializes a new ReusableBarrierCond instance.

        Args:
            num_threads (int): The number of threads that must reach the
                                barrier before any can proceed.
        """
        self.num_threads = num_threads # Total number of threads expected to reach the barrier.
        self.count_threads = self.num_threads # Current count of threads yet to reach the barrier.
        self.cond = Condition() # Condition variable used for blocking and unblocking threads.

    def wait(self):
        """
        Main entry point for threads to wait at the barrier.

        A thread calling this method will block until `num_threads`
        threads have also called `wait()`, after which all waiting
        threads are released.
        """
        self.cond.acquire() # Acquires the lock associated with the condition variable.
        self.count_threads -= 1 # Decrements the count of threads yet to reach the barrier.
        # Block Logic: Checks if this is the last thread to reach the barrier.
        if self.count_threads == 0:
            self.cond.notify_all() # Notifies all waiting threads to resume execution.
            self.count_threads = self.num_threads # Resets the counter for the next use of the barrier.
        else:
            self.cond.wait() # Waits (blocks) until notified by the last thread to reach the barrier.
        self.cond.release() # Releases the lock after proceeding.


class Device(object):
    """
    Represents a single device or processing unit within a distributed system.
    Each device manages its own sensor data, executes scripts, and interacts
    with other devices through a supervisor. It employs a multi-threaded
    architecture with shared locks and barriers for synchronization across
    multiple `DeviceThread` instances.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): Initial sensor data for the device,
                                mapping locations to data values.
            supervisor (Supervisor): The supervisor managing this device
                                     and its interactions with others.
        """
        self.root_device = 0 # Identifier for the root or coordinating device.
        self.init_lock = Lock() # Lock for protecting initialization of shared resources.
        self.finalize_lock = Lock() # Lock for protecting finalization steps at the end of a timepoint.
        self.max_threads = 8 # Maximum number of worker threads per device.
        # Barrier for synchronizing worker threads within this specific device.
        self.device_barrier = ReusableBarrierCond(self.max_threads)

        # Shared resources, initialized by the root device during setup.
        self.neighbours = None # List of neighboring devices.
        self.barrier = None # Global barrier for synchronizing all devices.
        self.locks = None # Dictionary of locks for data locations.
        self.dict_lock = None # Lock for protecting access to the `locks` dictionary.

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when a script has been assigned.
        self.scripts = [] # List of scripts assigned to this device.
        self.timepoint_done = Event() # Event to signal that the timepoint's tasks are completed.
        # List of DeviceThread workers for parallel script execution.
        self.threads = [DeviceThread(self, i) for i in range(self.max_threads)]

        # Block Logic: Starts all worker threads for this device.
        for thread in self.threads:
            thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the device with a list of other devices in the system
        and establishes shared synchronization primitives (global barrier and location-specific locks).

        This method is designed to be called once by a "root" device (device_id 0)
        to initialize shared resources for all participating devices.

        Args:
            devices (list): A list of Device objects representing
                            all devices in the system.
        """
        # Block Logic: This block is executed only by the root device (device_id 0),
        # acting as a coordinator to set up shared resources for all devices.
        if self.device_id == self.root_device:
            self.locks = {} # Initializes an empty dictionary to hold location-specific locks.
            self.dict_lock = Lock() # Initializes a lock to protect access to the 'locks' dictionary.
            # Initializes a global barrier for synchronizing all devices, considering all worker threads.
            self.barrier = ReusableBarrierCond(self.max_threads * len(devices))

            # Block Logic: Propagates the shared dictionary of locks, the dictionary lock,
            # and the global barrier to all other non-root devices.
            for device in devices:
                if device.device_id != self.root_device:
                    device.dict_lock = self.dict_lock
                    device.locks = self.locks
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.

        Args:
            script (Script): The script object to be executed.
                             If None, it signifies that all scripts for the
                             current timepoint have been assigned.
            location (int): The identifier for the data location
                            the script operates on.
        """
        # Block Logic: If a script is provided, it's added to the list of scripts
        # and the 'script_received' event is set to signal availability.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signals that a script has been received.
        # Block Logic: If no script is provided (script is None), it signals that
        # the timepoint's script assignment is complete, allowing worker threads to proceed.
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location.

        Args:
            location (int): The identifier of the data location.

        Returns:
            Any: The sensor data at the given location, or None if the
                 location does not exist in the device's sensor_data.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location.

        Args:
            location (int): The identifier of the data location to update.
            data (Any): The new data value to set for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device,
        waiting for all its background worker threads to complete.
        """
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread operating within a Device.

    Each `DeviceThread` is responsible for processing a subset of the
    assigned scripts for the device, specifically those distributed to it
    based on its `thread_id`. It handles data retrieval, script execution,
    and data propagation, coordinating with other threads via shared locks and barriers.
    """

    def __init__(self, device, thread_id):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The parent Device object.
            thread_id (int): A unique identifier for this specific worker thread
                             within its parent device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        The main execution loop for the DeviceThread worker.

        Each worker thread is responsible for a subset of the assigned scripts.
        It continuously retrieves neighbor information, waits for scripts,
        executes its assigned scripts for each timepoint, handles data
        synchronization, and then waits at global barriers.
        """
        while True:
            # Block Logic: Acquires a lock to safely initialize neighbors for the current timepoint.
            self.device.init_lock.acquire()
            # Pre-condition: `self.device.neighbours` is None, indicating new neighbors need to be fetched.
            # Invariant: `self.device.neighbours` will hold the current list of neighbors.
            if self.device.neighbours is None:
                self.device.neighbours = self.device.supervisor.get_neighbours()
            neighbours = self.device.neighbours

            self.device.init_lock.release() # Releases the initialization lock.

            # Block Logic: If neighbours are None, it indicates a shutdown request, so the thread breaks its loop.
            if neighbours is None:
                break

            # Block Logic: Waits until the timepoint is marked as done, signifying all scripts are ready to be processed.
            self.device.timepoint_done.wait()

            # Block Logic: Iterates through the assigned scripts, processing only those designated for this thread based on its `thread_id`.
            for i in range(self.thread_id, len(self.device.scripts), self.device.max_threads):
                (script, location) = self.device.scripts[i]

                # Block Logic: Acquires a lock for the shared `locks` dictionary before checking/creating a lock for the current `location`.
                self.device.dict_lock.acquire()
                if location not in self.device.locks:
                    self.device.locks[location] = Lock() # Creates a new lock if one doesn't exist for this location.
                self.device.dict_lock.release()

                # Block Logic: Acquires the specific lock for the current data `location` to ensure exclusive access.
                self.device.locks[location].acquire()

                script_data = []
                # Block Logic: Gathers sensor data from all known neighboring devices for the specified location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Gathers sensor data from the local device for the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: There is collected data for the script to process.
                # Invariant: If data is present, the script is run and its results are propagated to the devices.
                if script_data != []:
                    # Block Logic: Executes the script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Updates the sensor data on all neighboring devices with the result of the script execution.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Updates the sensor data on the local device with the result of the script execution.
                    self.device.set_data(location, result)

                self.device.locks[location].release() # Releases the location-specific lock.

            # Block Logic: Waits at a device-internal barrier to ensure all worker threads within this device complete their script processing.
            self.device.device_barrier.wait()

            # Block Logic: Acquires a lock to safely finalize the timepoint (resetting neighbors and timepoint_done event).
            self.device.finalize_lock.acquire()
            # Pre-condition: `self.device.neighbours` is not None, indicating a timepoint was processed.
            if self.device.neighbours is not None:
                self.device.neighbours = None # Resets neighbors for the next timepoint.
                self.device.timepoint_done.clear() # Clears the timepoint_done event.
            self.device.finalize_lock.release() # Releases the finalization lock.

            # Block Logic: Waits at the global barrier to synchronize with all other devices (including their worker threads)
            # before proceeding to the next timepoint.
            self.device.barrier.wait()
