

"""
This module implements a distributed device simulation framework.
It defines `Device` objects, `DeviceThread` for orchestrating device operations,
`SlaveThread` for executing individual scripts, and a custom `ReusableBarrierSem`
for inter-thread synchronization using semaphores.
"""

from threading import *


class Device(object):
    """
    Represents a simulated device in a distributed system. Each device
    manages its own sensor data, processes assigned scripts, and
    synchronizes its operations with other devices through a supervisor
    and shared synchronization primitives (locks and barriers).
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
        self.script_received = Event() # Signals that a script has been assigned.
        self.scripts = [] # List of (script, location) tuples assigned to this device.
        self.timepoint_done = Event() # Signals completion of script assignments for a timepoint.
        self.thread = DeviceThread(self)
        self.thread.start()

        self.lock_data = Lock() # Lock for protecting access to `sensor_data`.
        self.lock_location = [] # List of locks, one per location, for fine-grained access control.
        self.time_barrier = None # Barrier for synchronizing device threads at timepoints.

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the device's synchronization mechanisms.
        Device 0 initializes the shared barrier and per-location locks,
        which are then distributed to all other devices.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        # Block Logic: Device 0 acts as the coordinator to initialize the shared barrier.
        if self.device_id == 0:
            # Functional Utility: `ReusableBarrierSem` ensures all device threads synchronize
            # at discrete timepoints before proceeding.
            self.time_barrier = ReusableBarrierSem(len(devices)) 

            # Block Logic: Propagates the initialized barrier to all other devices.
            for device in devices:
                device.time_barrier = self.time_barrier

            loc_num = 0
            # Block Logic: Determines the maximum location index to prepare for
            # creating enough location-specific locks.
            for device in devices:
                for location in device.sensor_data:
                    loc_num = max(loc_num, location) 
            # Block Logic: Initializes a list of `Lock` objects, one for each possible location,
            # for fine-grained access control to sensor data.
            for i in range(loc_num + 1):
                self.lock_location.append(Lock()) 

            # Block Logic: Propagates the newly created location-specific locks to all devices.
            for device in devices:
                device.lock_location = self.lock_location 

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location for this device.
        If a script is provided, it signals that a script has been received.
        If no script is provided (None), it signals that the timepoint is done.

        Args:
            script (Script or None): The script object to assign, or None if the timepoint is complete.
            location (int): The numerical identifier for the location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Functional Utility: Signals to the device thread that a script is available for processing.
            self.script_received.set()
        else:
            # Functional Utility: Signals that all script assignments for the current timepoint are complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location for which to retrieve data.

        Returns:
            any: The sensor data for the specified location, or None if not found.
        """
        # Pre-condition: Checks if the location exists in the device's sensor data.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location. Access to `sensor_data` is protected
        by `self.lock_data`.

        Args:
            location (int): The location for which to set data.
            data (any): The new data to set.
        """
        # Functional Utility: Acquires a lock to ensure exclusive access to the shared
        # `sensor_data` dictionary during modification.
        with self.lock_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device thread, waiting for its completion.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the operational lifecycle of a Device, including fetching neighbor
    information, orchestrating script execution using `SlaveThread`s,
    and synchronizing with other DeviceThreads at each timepoint.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread. It continuously
        processes sensor data, executes scripts via `SlaveThread`s,
        and synchronizes with other devices until a shutdown signal is received.
        """

        while True:
            slaves = [] # List to hold SlaveThread instances.
            
            # Block Logic: Fetches the current set of neighboring devices from the supervisor.
            # A `None` value indicates a shutdown signal.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Functional Utility: Blocks until all scripts for the current timepoint have been
            # assigned to the device.
            self.device.timepoint_done.wait()
            # Functional Utility: Resets the event for the next timepoint.
            self.device.timepoint_done.clear() 

            # Block Logic: For each assigned script, creates and starts a `SlaveThread`
            # to handle its execution concurrently.
            for (script, location) in self.device.scripts:
                slave = SlaveThread(script, location, neighbours, self.device) 
                slaves.append(slave)
                slave.start()

            # Block Logic: Joins all `SlaveThread`s, waiting for their completion
            # before proceeding, ensuring all scripts are executed for the timepoint.
            for i in range(len(slaves)):
                slaves.pop().join()

            # Functional Utility: Synchronizes with all other DeviceThreads in the simulation
            # using a barrier, ensuring all devices complete their current timepoint
            # processing before proceeding.
            self.device.time_barrier.wait() 

class SlaveThread(Thread):
    """
    A dedicated thread for executing a single script at a specific location
    for a device, interacting with neighbor data and ensuring thread-safe
    access to shared resources.
    """
    def __init__(self, script, location, neighbours, device):
        """
        Initializes a new SlaveThread instance.

        Args:
            script (Script): The script to be executed.
            location (int): The numerical identifier for the location of the script.
            neighbours (list): A list of neighboring Device instances.
            device (Device): The Device instance that owns this slave thread.
        """

        Thread.__init__(self, name="Slave Thread of Device %d" % device.device_id)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """
        The main execution method for the SlaveThread. It collects data
        from its device and neighbors, executes the assigned script, and
        updates the relevant sensor data.
        """
        
        device = self.device
        script = self.script
        location = self.location
        neighbours = self.neighbours
        
        # Block Logic: Retrieves the local device's sensor data for the current location.
        data = device.get_data(location)
        input_data = [] # Accumulator for all data relevant to the script.
        # Functional Utility: Acquires the specific lock for the current location,
        # ensuring exclusive access to its data during script execution.
        this_lock = device.lock_location[location]

        if data is not None:
            input_data.append(data) 

        with this_lock: 
            # Block Logic: Collects sensor data from neighboring devices at the specified location,
            # ensuring each neighbor's data is retrieved safely.
            for neighbour in neighbours:
                temp = neighbour.get_data(location) 

                if temp is not None:
                    input_data.append(temp)

            # Pre-condition: Ensures there is data to process before executing the script.
            if input_data != []: 
                # Functional Utility: Executes the assigned script with the collected data,
                # simulating sensor data processing.
                result = script.run(input_data) 

                # Block Logic: Updates the sensor data of neighboring devices with the script's result.
                for neighbour in neighbours:
                    neighbour.set_data(location, result) 

                # Block Logic: Updates the local device's sensor data with the script's result.
                device.set_data(location, result) 


class ReusableBarrierSem():
    """
    Implements a reusable barrier synchronization mechanism using semaphores,
    allowing multiple threads to wait for each other to reach a common point
    before proceeding. This barrier can be reset and reused.
    It uses a two-phase approach to ensure proper synchronization even if
    threads arrive at different times in subsequent cycles.
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
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            # Functional Utility: Resets the second phase counter for the next barrier cycle.
            self.count_threads2 = self.num_threads
         
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
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            # Functional Utility: Resets the first phase counter for the next barrier cycle.
            self.count_threads1 = self.num_threads
         
        # Functional Utility: Acquires a permit from `threads_sem2`, blocking until
        # all threads have reached phase 2.
        self.threads_sem2.acquire()
