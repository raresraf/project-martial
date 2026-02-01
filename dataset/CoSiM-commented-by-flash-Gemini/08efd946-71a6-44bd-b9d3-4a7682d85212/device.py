"""
This module implements a distributed device simulation framework, utilizing a multi-threaded
approach and a semaphore-based reusable barrier for synchronization.

Algorithm:
- ReusableBarrierSem: A semaphore-based synchronization primitive ensuring all participating threads
  reach a specific point before proceeding. It uses a double-phase mechanism for reusability.
- Device: Represents a simulated physical device with sensor data and script processing capabilities.
- DeviceThread: Worker threads for each `Device`, responsible for executing assigned scripts
  concurrently across different data locations and coordinating with other threads/devices
  using shared barriers and location-specific locks.
"""

from threading import Lock, Event, Thread, Semaphore, Condition


class ReusableBarrierSem():
    """
    Implements a reusable barrier synchronization mechanism using Semaphores.

    This barrier allows a specified number of threads to wait until all have
    reached a certain point, after which all are released simultaneously. It
    uses a double-phase approach to allow the barrier to be reused multiple times
    without requiring reinitialization.

    Attributes:
        num_threads (int): The total number of threads expected to participate in the barrier.
        count_threads1 (int): The current count of threads waiting in phase 1.
        count_threads2 (int): The current count of threads waiting in phase 2.
        counter_lock (threading.Lock): A lock to protect access to the `count_threads` variables.
        threads_sem1 (threading.Semaphore): A semaphore used to release threads from phase 1.
        threads_sem2 (threading.Semaphore): A semaphore used to release threads from phase 2.
    """

    def __init__(self, num_threads):
        """
        Initializes a new instance of the ReusableBarrierSem.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can pass.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait until all `num_threads` threads
        have passed through both phases of this barrier.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        Manages the first phase of the barrier synchronization.

        Pre-condition: `count_threads1` holds the number of remaining threads for this phase.
        Invariant: `count_threads1` is decremented atomically. When it reaches zero,
                   all threads are released via `threads_sem1.release()`, and `count_threads1` is reset.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """
        Manages the second phase of the barrier synchronization.

        Pre-condition: `count_threads2` holds the number of remaining threads for this phase.
        Invariant: `count_threads2` is decremented atomically. When it reaches zero,
                   all threads are released via `threads_sem2.release()`, and `count_threads2` is reset.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a simulated device in a distributed environment.

    Each device has a unique ID, manages its sensor data, and can receive
    and execute scripts. It uses shared synchronization primitives (a reusable
    barrier and location-specific locks) for thread-safe operations and
    synchronization with other devices. Each device manages a pool of `DeviceThread`s.

    Class Attributes:
        location_locks (list): A shared list of (location_id, threading.Lock) tuples
                               to provide fine-grained access control to data locations.
        barrier (ReusableBarrierSem): A shared reusable barrier for synchronizing all `DeviceThread`s.
        nr_t (int): The number of `DeviceThread` instances per device.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings,
                            where keys represent locations.
        supervisor (Supervisor): A reference to the central supervisor managing devices.
        script_received (threading.Event): Event to signal when new scripts are assigned.
        scripts (list): A list to store assigned scripts (script, location) tuples.
        timepoint_done (threading.Event): Event to signal completion of script processing for a timepoint.
        neighbours_event (threading.Event): Event to signal when neighbour information is ready for workers.
        threads (list): A list of `DeviceThread` instances associated with this device.
    """

    location_locks = [] # Shared class attribute for location-specific locks.
    barrier = None      # Shared class attribute for the global barrier.
    nr_t = 8            # Number of DeviceThread instances per device.

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
        # Event for signaling when neighbour information is available for worker threads.
        self.neighbours_event = Event()
        # Create and start a pool of DeviceThread instances for this device.
        self.threads = []
        for i in xrange(Device.nr_t): # xrange for Python 2 compatibility.
            self.threads.append(DeviceThread(self, i))
        for i in xrange(Device.nr_t): # xrange for Python 2 compatibility.
            self.threads[i].start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization primitives (a global barrier) across all `DeviceThread`s.
        This method is called once by a coordinating entity (e.g., supervisor).

        Pre-condition: This method should be called once by a central entity.
        Invariant: `Device.barrier` is initialized as a `ReusableBarrierSem` whose count is
                   the total number of `DeviceThread` instances across all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Initialize the global barrier for all DeviceThread instances across all devices.
        Device.barrier = ReusableBarrierSem(Device.nr_t * len(devices))

    def assign_script(self, script, location):
        """
        Assigns a script to be executed on data at a specific location for this device.
        Initializes location-specific locks if they don't exist. If `script` is None,
        it signals that script assignments for the current timepoint are complete.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (int): The data location where the script should be applied.
        """
        # If a lock for this location doesn't exist, create and add it to the shared list.
        if location not in [elem[0] for elem in Device.location_locks]:
            Device.location_locks.append((location, Lock()))

        if script is not None:
            self.scripts.append((script, location))
            # Signal that new scripts have been received.
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
        Joins all of the device's dedicated `DeviceThread`s, effectively waiting
        for them to complete their execution before the program exits.
        """
        for i in xrange(Device.nr_t): # xrange for Python 2 compatibility.
            self.threads[i].join()

class DeviceThread(Thread):
    """
    A worker thread for a `Device` object, responsible for executing a subset
    of assigned scripts concurrently.

    It interacts with the supervisor (via `device.threads[0]`), waits for script
    assignments, processes its share of scripts, and synchronizes with other
    `DeviceThread` instances using shared barriers and location-specific locks.

    Attributes:
        device (Device): The Device object associated with this thread.
        index (int): The unique index of this `DeviceThread` within its parent `Device`'s pool.
        neighbours (list): A list of neighboring Device objects (retrieved by `index == 0` thread).
    """

    def __init__(self, device, index):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
            index (int): The index of this thread in the device's thread pool.
        """
        # Initialize the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device Thread %d - Worker %d" % (device.device_id, index))
        self.device = device
        self.index = index
        self.neighbours = None

    def run(self):
        """
        The main execution loop for the DeviceThread worker.

        Invariant: The loop continues until the supervisor signals termination
                   (by returning None for neighbors). Within each iteration,
                   the thread synchronizes to get neighbor information, waits
                   for timepoint completion signal, processes its assigned scripts
                   in a thread-safe manner using location-specific locks, and then
                   synchronizes with all other `DeviceThread`s via the global barrier.
        """
        while True:
            # Block Logic: Only the thread with index 0 retrieves neighbour information
            # and signals other threads for this device when it's ready.
            if self.index == 0:
                self.neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours_event.set() # Signal that neighbours are ready.
            else:
                self.device.neighbours_event.wait() # Wait for neighbours to be ready.
                self.neighbours = self.device.threads[0].neighbours # Get neighbours from thread 0.

            # Block Logic: If no neighbors are returned (e.g., simulation termination signal), break the loop.
            if self.neighbours is None:
                break

            # Wait for the `timepoint_done` event, which signals that script assignments
            # for the current timepoint are complete and ready for processing by this device.
            self.device.timepoint_done.wait()

            # Block Logic: This thread processes a subset of the scripts assigned to its device.
            # It iterates through the device's scripts list with a step equal to Device.nr_t.
            for j in range(self.index, len(self.device.scripts), Device.nr_t):
                location = self.device.scripts[j][1]
                script = self.device.scripts[j][0]

                # Block Logic: Acquire the location-specific lock before accessing/modifying data.
                for i in range(len(Device.location_locks)):
                    if location == Device.location_locks[i][0]:
                        Device.location_locks[i][1].acquire()

                script_data = []
                # Block Logic: Collect data from neighboring devices for the current location.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                # Collect data from the current device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: If data is available, execute the script and update results.
                if script_data != []:
                    # Execute the script with the collected data.
                    result = script.run(script_data)

                    # Update the sensor data in neighboring devices.
                    for device in self.neighbours:
                        device.set_data(location, result)
                    # Update the sensor data in the current device.
                    self.device.set_data(location, result)

                # Release the location-specific lock.
                for i in range(len(Device.location_locks)):
                    if location == Device.location_locks[i][0]:
                        Device.location_locks[i][1].release()

            # Synchronize with all other DeviceThread instances using the global barrier.
            Device.barrier.wait()
            # Block Logic: Only thread with index 0 clears events and resets for the next timepoint.
            if self.index == 0:
                self.device.timepoint_done.clear()
                self.device.scripts = [] # Functional utility: Clear the processed scripts list.

            if self.index == 0:
                self.device.neighbours_event.clear() # Reset neighbours event for next timepoint.
            # Synchronize again to ensure all threads are ready for the next iteration.
            Device.barrier.wait()
