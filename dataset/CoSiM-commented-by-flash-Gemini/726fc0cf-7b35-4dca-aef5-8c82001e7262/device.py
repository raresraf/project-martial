"""
This module implements a multi-threaded device simulation framework.

It defines a reusable barrier for thread synchronization, a `Device` class
representing simulated entities, a `DeviceThread` to manage device operations,
and `MyScriptThread` threads to execute scripts. The framework supports
inter-device communication and script processing in a concurrent manner,
utilizing a shared, semaphore-based barrier for global synchronization.
"""

from threading import Event, Semaphore, Lock, Thread
import Queue # Imported but not explicitly used in this file's direct code; might be used by a dependency.


class ReusableBarrierSem(object):
    """
    Implements a reusable barrier synchronization primitive using semaphores.

    This barrier ensures that a fixed number of participating threads wait
    at a designated point until all threads have arrived. Once all threads
    reach the barrier, they are all released simultaneously. It uses a
    two-phase mechanism (`phase1` and `phase2`) to allow for reuse across
    multiple synchronization points.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                before any are released.
        """
        self.num_threads = num_threads
        # Counters for threads in phase 1 and phase 2.
        # These are simple integers as they are protected by `counter_lock`.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock() # Lock to protect the counters during modifications.
        
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase, initially blocking all threads.
        
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase, initially blocking all threads.

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        participating threads have also called wait(). This method orchestrates
        the two phases of the barrier to enable reuse.
        """
        self.phase1() # Execute the first barrier phase.
        self.phase2() # Execute the second barrier phase.

    def phase1(self):
        """
        Manages the first synchronization phase of the barrier.

        Threads acquire the `counter_lock`, decrement their count, and if they
        are the last thread to reach this phase, they release all waiting threads
        in `threads_sem1` and reset the counter. Otherwise, they wait on `threads_sem1`.
        """
        with self.counter_lock: # Protect access to the counter.
            self.count_threads1 -= 1 # Decrement the count of threads yet to reach the barrier.
            
            if self.count_threads1 == 0: # Check if this is the last thread to reach the barrier.
                for i in range(self.num_threads):
                    self.threads_sem1.release() # Release all threads waiting on this phase's semaphore.
                self.count_threads1 = self.num_threads # Reset the counter for the next use of the barrier.
        self.threads_sem1.acquire() # All threads wait here until released by the last thread in this phase.

    def phase2(self):
        """
        Manages the second synchronization phase of the barrier.

        Similar to `phase1`, but uses a separate counter and semaphore (`count_threads2`, `threads_sem2`)
        to allow the barrier to be reused for subsequent synchronization points.
        """
        with self.counter_lock: # Protect access to the counter.
            self.count_threads2 -= 1 # Decrement the count of threads yet to reach the barrier.
            if self.count_threads2 == 0: # Check if this is the last thread to reach the barrier.
                for i in range(self.num_threads):
                    self.threads_sem2.release() # Release all threads waiting on this phase's semaphore.
                self.count_threads2 = self.num_threads # Reset the counter for the next use of the barrier.
        self.threads_sem2.acquire() # All threads wait here until released by the last thread in this phase.


class Device(object):
    """
    Represents a single simulated device in the system.

    Each device manages its own `sensor_data`, interacts with a `supervisor`,
    and processes scripts. It coordinates with other devices through a shared
    barrier and protects its sensor data with a local lock.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary mapping locations (int) to sensor data values.
            supervisor (object): A reference to the supervisor object managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # `result_queue` is declared but not explicitly used in the provided code.
        self.result_queue = Queue.Queue()
        self.set_lock = Lock()          # Lock to protect `self.sensor_data` during `set_data` operations.
        # `neighbours_lock` is intended for global neighbor list protection, but is not used in this specific implementation.
        self.neighbours_lock = None
        self.barrier = ReusableBarrierSem(0) # Placeholder barrier, will be set during `setup_devices`.

        self.script_received = Event()  # Event to signal when new scripts have been assigned.
        self.scripts = []               # List of (script, location) tuples assigned to this device.
        self.timepoint_done = Event()   # Event to signal that all scripts for a timepoint have been assigned.

        # The main thread responsible for this device's control flow and script delegation.
        self.thread = DeviceThread(self)

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global shared synchronization primitives (specifically the barrier).
        This method is designed to be called once by all devices, but global initialization
        logic for the barrier is handled by the device with the lowest `device_id` (assumed to be `devices[0]`).

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Global initialization: Only the first device (assumed `device_id` 0) creates
        # and distributes the global barrier.
        if self.device_id == devices[0].device_id:
            # Initializes the global barrier with the total number of devices.
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            # Other devices receive the shared barrier from the first device.
            self.barrier = devices[0].barrier

        self.thread.start() # Start the `DeviceThread` after global resources are set up.

    def assign_script(self, script, location):
        """
        Assigns a processing script and its associated data location to this device.
        If `script` is None, it signals that script assignment for the current
        timepoint is complete.

        Args:
            script (object): The script object (must have a `run` method) to be executed,
                             or None to signal timepoint completion.
            location (int): The data location this script pertains to.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the new script to the list.
        else:
            # If no script is provided, it indicates the end of script assignments for this timepoint.
            self.script_received.set() # Signal script processing can begin.
            self.timepoint_done.set()  # Signal that all scripts for the current timepoint have been assigned.

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location within this device.

        Note: This method does *not* acquire any locks to protect `self.sensor_data`.
        If `get_data` is called concurrently with `set_data` (which uses `set_lock`),
        or by multiple worker threads on the same device, this could lead to race
        conditions and inconsistent data reads. External synchronization is required
        before calling this method for data consistency.

        Args:
            location (int): The location ID for which to retrieve data.

        Returns:
            Any: The data at the specified location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location within this device.
        Access to `self.sensor_data` is protected by `self.set_lock`.

        Args:
            location (int): The location ID for which to set data.
            data (Any): The new data value to be set.
        """
        # The `my_lock` is acquired and released by `MyScriptThread` externally.
        # This `set_data` method itself doesn't acquire `self.set_lock` directly.
        # This is an inconsistency in the current implementation.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the graceful shutdown sequence for the device's main thread.
        """
        self.thread.join()


class MyScriptThread(Thread):
    """
    A dedicated thread responsible for executing a single script task.

    Instances of `MyScriptThread` are created and managed by `DeviceThread`
    to process individual scripts concurrently. It handles collecting input
    data and distributing results while ensuring data consistency using locks.
    """

    def __init__(self, script, location, device, neighbours):
        """
        Initializes a MyScriptThread instance.

        Args:
            script (object): The script object (must have a `run` method) to be executed.
            location (int): The data location relevant to this script.
            device (Device): The Device instance this worker operates for.
            neighbours (list): The list of neighboring devices for the current timepoint.
        """
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        The main execution logic for `MyScriptThread`.

        It collects data from neighbors and the local device for its assigned
        location, executes the script, and then updates the data in relevant
        devices. It uses `self.device.my_lock` to protect data writes.
        """
        script_data = [] # List to collect all relevant data for the script.

        # Gathers data from all neighboring devices for the current location.
        # Note: `device.get_data` itself does not use locks, relying on external synchronization.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Gathers data from its own device for the current location.
        # Note: `self.device.get_data` itself does not use locks, relying on external synchronization.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # If any data was collected, run the script and update devices.
        if script_data != []:
            result = self.script.run(script_data) # Execute the script.

            # Updates the data in neighboring devices with the script's result.
            # Each neighbor's `my_lock` is acquired before updating its `sensor_data`.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            # Updates its own device's data with the script's result.
            # `self.device.my_lock` is acquired before updating its `sensor_data`.
            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()


class DeviceThread(Thread):
    """
    The main control thread for a `Device`.

    It is responsible for interacting with the supervisor to get neighbor
    information, managing timepoint progression, and directly spawning
    `MyScriptThread` instances for script execution. It also participates
    in global synchronization using a shared barrier.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously processes timepoints:
        1. Retrieves neighbor information from the supervisor.
        2. Participates in a global barrier synchronization.
        3. Waits for scripts to be assigned.
        4. Spawns `MyScriptThread` instances for each assigned script.
        5. Waits for all `MyScriptThread` instances to complete.
        6. Participates in another global barrier synchronization after script execution.
        7. Clears the `script_received` event for the next timepoint.
        """
        while True:
            # Retrieves the list of neighboring devices from the supervisor.
            # Note: `device.neighbours_lock` (global) is not used here to protect this read.
            neighbours = self.device.supervisor.get_neighbours()
            
            # If no neighbors are returned (e.g., shutdown signal from supervisor),
            # break the loop and terminate this thread.
            if neighbours is None:
                break;
            
            # First global barrier synchronization: All DeviceThreads wait here
            # before starting script execution for the current timepoint.
            self.device.barrier.wait()

            # Waits until new scripts have been assigned to this device for the current timepoint.
            self.device.script_received.wait()
            
            script_threads = [] # List to hold `MyScriptThread` instances for the current timepoint.
            
            # Creates a `MyScriptThread` for each assigned script.
            for (script, location) in self.device.scripts:
                script_threads.append(MyScriptThread(script,
                    location, self.device, neighbours))
            
            # Starts all `MyScriptThread` instances.
            for thread in script_threads:
                thread.start()
            
            # Waits for all `MyScriptThread` instances to complete their tasks.
            for thread in script_threads:
                thread.join()
            
            # `timepoint_done.wait()` is present in `assign_script` but not waited on here.
            # The `timepoint_done` event is not actively used in this `run` method.
            self.device.timepoint_done.wait()
            
            # Second global barrier synchronization: All DeviceThreads wait here
            # after completing script execution for the current timepoint.
            self.device.barrier.wait()
            
            self.device.script_received.clear() # Clear the event for the next script assignment cycle.