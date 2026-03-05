

"""
This module implements a distributed device simulation framework with advanced
synchronization mechanisms. It defines `Device` objects, a `DeviceThread` for
orchestrating script execution, and custom `ReusableBarrierSem` and `MyLock`
classes for managing thread synchronization and resource access.
"""

from threading import Event, Thread, RLock, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    Implements a reusable barrier synchronization mechanism using semaphores,
    allowing multiple threads to wait for each other to reach a common point
    before proceeding. This barrier can be reset and reused.
    It employs a two-phase approach to ensure proper synchronization even
    if threads arrive at different times in subsequent cycles.
    """

    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrierSem instance.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can proceed.
        """
        self.num_threads = num_threads
        # Functional Utility: `count_threads1` tracks threads in the first phase of the barrier.
        self.count_threads1 = self.num_threads

        # Functional Utility: `count_threads2` tracks threads in the second phase of the barrier.
        self.count_threads2 = self.num_threads
        
        # Functional Utility: `counter_lock` protects access to the thread counters.
        self.counter_lock = Lock()
        
        # Functional Utility: `threads_sem1` is used to block threads in the first phase
        # until all threads have arrived.
        self.threads_sem1 = Semaphore(0)
        
        # Functional Utility: `threads_sem2` is used to block threads in the second phase
        # until all threads have arrived.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait until all other threads have also
        called `wait()` on this barrier. Implements a two-phase synchronization.
        """
        
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        First phase of the reusable barrier. Threads decrement a counter and
        wait on `threads_sem1` until all threads have reached this phase.
        """
        
        with self.counter_lock:
            # Block Logic: Decrements the counter for threads in phase 1.
            self.count_threads1 -= 1
            # Pre-condition: Checks if the current thread is the last one to reach this phase.
            if self.count_threads1 == 0:
                # Block Logic: Releases all threads waiting on `threads_sem1` if this is the last thread.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Functional Utility: Resets the first phase counter for the next barrier cycle.
                self.count_threads1 = self.num_threads
        # Functional Utility: Acquires a permit from `threads_sem1`, blocking until
        # all threads have reached phase 1.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        Second phase of the reusable barrier. Threads decrement a counter and
        wait on `threads_sem2` until all threads have reached this phase.
        """
        
        with self.counter_lock:
            # Block Logic: Decrements the counter for threads in phase 2.
            self.count_threads2 -= 1
            # Pre-condition: Checks if the current thread is the last one to reach this phase.
            if self.count_threads2 == 0:
                # Block Logic: Releases all threads waiting on `threads_sem2` if this is the last thread.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Functional Utility: Resets the second phase counter for the next barrier cycle.
                self.count_threads2 = self.num_threads
        # Functional Utility: Acquires a permit from `threads_sem2`, blocking until
        # all threads have reached phase 2.
        self.threads_sem2.acquire()

class MyLock(object):
    """
    A custom lock wrapper that associates an RLock with a specific
    device ID and a zone (location). This allows for fine-grained
    locking based on the origin and data location.
    """

    def __init__(self, deviceId, zone):
        """
        Initializes a MyLock instance.

        Args:
            deviceId (int): The ID of the device associated with this lock.
            zone (int): The zone or location associated with this lock.
        """
        
        self.lock = RLock() # Functional Utility: An RLock (reentrant lock) is used, allowing
                            # the same thread to acquire the lock multiple times.
        self.dev = deviceId
        self.zone = zone

    def acquire(self):
        """
        Acquires the underlying RLock. Blocks until the lock is acquired.
        """
        
        self.lock.acquire()

    def release(self):
        """
        Releases the underlying RLock.
        """
        
        self.lock.release()

def get_leader(devices):
    """
    Identifies the device with the smallest `device_id` from a list of devices,
    designating it as the leader.

    Args:
        devices (list): A list of Device instances.

    Returns:
        int: The `device_id` of the leader device.
    """
    
    leader = devices[0].device_id
    for dev in devices:
        if dev.device_id < leader:
            leader = dev.device_id
    return leader

class Device(object):
    """
    Represents a simulated device in a distributed system. Each device manages
    its own sensor data, processes assigned scripts through a dedicated thread,
    and coordinates its operations with other devices via a supervisor and
    shared synchronization primitives including global and location-specific locks,
    and a reusable barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor data, keyed by location.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Signals that a script has been assigned or timepoint done.
        self.scripts = [] # List of (script, location) tuples assigned to this device.
        self.timepoint_done = Event() # Signals completion of script assignments for a timepoint.
        self.thread = DeviceThread(self) # The main thread for this device.
        self.thread.start()
        self.barrier = None # Shared barrier for global synchronization.
        self.global_lock = None # A global RLock for critical sections.
        self.gl1 = None # Another global RLock, possibly for a different critical section.
        self.lock_list = None # A list to store `MyLock` instances for dynamic, location-specific locking.


    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the device's shared synchronization mechanisms.
        The leader device initializes the global locks, the lock list,
        and the shared barrier, which are then distributed to all other devices.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        # Functional Utility: Determines the leader device based on the smallest ID.
        leader = get_leader(devices)
        # Block Logic: The leader device (device with the smallest ID) is responsible
        # for initializing shared resources.
        if self.device_id == leader:
            # Functional Utility: Initializes global reentrant locks for critical sections.
            global_lock = RLock()
            gl1 = RLock()
            # Functional Utility: Initializes an empty list to store dynamically created `MyLock` instances.
            lock_list = []
            # Functional Utility: Initializes a reusable barrier for global synchronization
            # across all device threads.
            barrier = ReusableBarrierSem(len(devices))
            # Block Logic: Propagates the initialized shared resources to all devices in the simulation.
            for dev in devices:
                dev.barrier = barrier
                dev.global_lock = global_lock
                dev.gl1 = gl1
                dev.lock_list = lock_list

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location for this device.
        If a script is provided, it is added to the list of scripts.
        If no script is provided (None), it signals that a script has been received
        and that the timepoint is done.

        Args:
            script (object or None): The script object to assign, or None if the timepoint is complete.
            location (int): The numerical identifier for the location associated with the script.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Functional Utility: Signals that all script assignments for the current timepoint are complete.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location for which to retrieve data.

        Returns:
            any: The sensor data for the specified location, or None if not found.
        """
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location.

        Args:
            location (int): The location for which to set data.
            data (any): The new data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device thread, waiting for its completion.
        """
        
        self.thread.join()

class MyThread(Thread):
    """
    A worker thread responsible for executing a list of scripts for a device.
    It collects necessary sensor data from the local device and its neighbors,
    executes each script, and updates the sensor data accordingly.
    """

    def __init__(self, device, scripts, neighbours):
        """
        Initializes a MyThread instance.

        Args:
            device (Device): The Device instance this thread is associated with.
            scripts (list): A list of (script, location) tuples for this thread to execute.
            neighbours (list): A list of neighboring Device instances.
        """
        
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):
        """
        The main execution method for the MyThread. It iterates through
        its assigned scripts, collects data from local and neighbor devices,
        executes the script, and updates sensor data.
        """

        dev = self.device
        scripts = self.scripts
        neighbours = self.neighbours

        # Block Logic: Iterates through each script-location pair assigned to this worker thread.
        for (script, location) in scripts:
            script_data = [] # Accumulator for all data relevant to the script.

            # Block Logic: Gathers sensor data from neighboring devices at the specified location.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Block Logic: Retrieves the local device's sensor data for the current location.
            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)

            # Pre-condition: Checks if there is any data collected for the script to run.
            if script_data != []:
                # Functional Utility: Executes the assigned script with the collected data,
                # simulating sensor data processing.
                result = script.run(script_data)
                
                # Block Logic: Updates the sensor data of neighboring devices with the script's result.
                for device in neighbours:
                    device.set_data(location, result)
                # Block Logic: Updates the local device's sensor data with the script's result.
                dev.set_data(location, result)

def contains(my_list, searched):
    """
    Checks if a given element is present in a list.

    Args:
        my_list (list): The list to search within.
        searched (object): The element to search for.

    Returns:
        int: 1 if the element is found, 0 otherwise.
    """
    
    for elem in my_list:
        if elem == searched:
            return 1
    return 0

class DeviceThread(Thread):
    """
    The main thread for a Device. It orchestrates the collection of neighbor
    information, the dynamic management of shared locks, the distribution
    and execution of scripts via `MyThread` instances, and synchronization
    at timepoints.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device instance this thread is managing.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def existent_lock(self, new_lock):
        """
        Checks if a given `MyLock` object already exists in the device's
        `lock_list` based on `device_id` and `zone`.

        Args:
            new_lock (MyLock): The `MyLock` object to check for existence.

        Returns:
            int: 1 if an equivalent lock exists, 0 otherwise.
        """
        
        for lock in self.device.lock_list:
            if new_lock.dev == lock.dev:
                if new_lock.zone == lock.zone:
                    return 1
        return 0

    def get_index(self, dev, zone):
        """
        Retrieves the index of a `MyLock` object in the device's `lock_list`
        based on `device_id` and `zone`.

        Args:
            dev (int): The device ID associated with the lock.
            zone (int): The zone or location associated with the lock.

        Returns:
            int: The index of the lock in `lock_list` if found, -1 otherwise.
        """
        
        my_list = self.device.lock_list
        for i in range(len(my_list)):
            if dev == my_list[i].dev:
                if zone == my_list[i].zone:
                    return i
        
        # Post-condition: Returns -1 if no matching lock is found.
        return -1

    def run(self):
        """
        The main execution loop for the DeviceThread. It continuously
        processes neighbor data, dynamically manages locks, dispatches
        scripts to worker threads, and synchronizes with other devices.
        """

        while True:
            # Block Logic: Fetches the current set of neighboring devices from the supervisor.
            
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: Checks if a shutdown signal has been received from the supervisor.
            if neighbours is None:
                break

            # Block Logic: Waits until the device has received all scripts for the current timepoint
            # or a timepoint-done signal. Clears the events for the next cycle.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Functional Utility: Acquires a global lock (`global_lock`) to protect the critical
            # section where the shared `lock_list` is being updated and `MyLock` instances are managed.
            self.device.global_lock.acquire()

            # Functional Utility: `my_list` stores indices of locks acquired by this device and its neighbors
            # for the current timepoint's scripts.
            my_list = []

            # Block Logic: Iterates through each assigned script to identify all necessary locks
            # (for the current device and its neighbors) and add them to the global `lock_list` if new.
            for (script, location) in self.device.scripts:

                # Block Logic: Manages the lock for the current device and location.
                new_lock = MyLock(self.device.device_id, location)
                # Pre-condition: Checks if a lock for this device and location already exists in the shared list.
                if self.existent_lock(new_lock) == 0:
                    # Block Logic: If not, adds the new lock to the shared `lock_list`.
                    self.device.lock_list.append(new_lock)
                # Functional Utility: Retrieves the index of the lock for the current device and location.
                index = self.get_index(self.device.device_id, location)

                # Block Logic: Iterates through neighboring devices to ensure their locks are also managed.
                for device in neighbours:
                    new_lock = MyLock(device.device_id, location)
                    # Pre-condition: Checks if a lock for this neighbor device and location already exists.
                    if self.existent_lock(new_lock) == 0:
                        # Block Logic: If not, adds the new lock to the shared `lock_list`.
                        self.device.lock_list.append(new_lock)
                    # Functional Utility: Retrieves the index of the lock for the neighbor device and location.
                    index = self.get_index(device.device_id, location)
                    # Block Logic: Adds the index of the lock to `my_list` if it's not already there,
                    # ensuring each relevant lock is acquired only once.
                    if contains(my_list, index) == 0:
                        my_list.append(index)

            # Functional Utility: Releases the global lock (`global_lock`) after `lock_list` updates are complete.
            self.device.global_lock.release()

            # Functional Utility: Acquires another global lock (`gl1`) to protect the section where
            # the necessary `MyLock` instances from `lock_list` are acquired.
            self.device.gl1.acquire()
            # Block Logic: Acquires all necessary `MyLock` instances for the current scripts.
            for index in my_list:
                self.device.lock_list[index].acquire()
            self.device.gl1.release()

            # Functional Utility: Determines the number of scripts to be executed in this timepoint.
            length = len(self.device.scripts)
            # Pre-condition: Checks if there is only one script to execute.
            if length == 1:
                # Block Logic: Creates and starts a single `MyThread` for the script, then waits for its completion.
                trd = MyThread(self.device, self.device.scripts, neighbours)
                trd.start()
                trd.join()
            else:
                # Block Logic: Creates multiple `MyThread` instances, one for each script,
                # starts them, and then waits for all of them to complete.
                tlist = []
                for i in range(length):
                    lst = [self.device.scripts[i]] # Creates a list with a single script for each thread.
                    trd = MyThread(self.device, lst, neighbours)
                    trd.start()
                    tlist.append(trd)
                for i in range(length):
                    tlist[i].join()

            # Block Logic: Releases all `MyLock` instances acquired for the current scripts.
            for index in my_list:
                self.device.lock_list[index].release()

            # Functional Utility: Waits until the timepoint is explicitly marked as done.
            self.device.timepoint_done.wait()
            # Functional Utility: Clears the `timepoint_done` event for the next cycle.
            self.device.timepoint_done.clear()
            # Functional Utility: Synchronizes with all other `DeviceThread` instances
            # across devices using a shared barrier, ensuring all devices complete
            # their current timepoint processing before proceeding.
            self.device.barrier.wait()
