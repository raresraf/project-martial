


from threading import Thread, Lock, Condition, Event



"""
This module implements a simulation framework for distributed devices,
focusing on concurrent execution of scripts and synchronized data processing.
It utilizes a `Device` class to represent each simulated entity,
a `DeviceThread` for main control flow, and `ExecuteSript` threads for
executing individual scripts. A distributed locking mechanism is employed
to ensure consistent data access across multiple devices using `Lock` objects
for specific data locations.
"""


from threading import Event, Thread, Semaphore, Lock, RLock, Condition # Condition is specifically used by ReusableBarrier.
from reusable_barrier import ReusableBarrier # External dependency: Assumes ReusableBarrier is defined elsewhere.
import multiprocessing # Imported but not directly used in the provided snippet.

class Device(object):
    """
    Represents a simulated device within a distributed system.

    Each device manages its sensor data, interacts with a central supervisor,
    and dispatches scripts for execution. It launches a dedicated `DeviceThread`
    to handle control flow and participates in global synchronization.
    A distributed locking mechanism using `lock_set` is employed for data locations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor readings
                                (e.g., {location_id: data_value}).
            supervisor (object): An object representing the central supervisor,
                                 used for coordination (e.g., getting neighbors).
        """
        # Dictionary to store script execution results (unused in provided snippet).
        self.results = {}
        # Local lock for protecting access to this device's `sensor_data` (unused in provided snippet).
        self.lock = None # Note: This `lock` is later set to a `Lock()` in `setup_devices` (for device 0).
        # Dictionary of reentrant locks (`RLock`), one per data location, shared across *all* devices
        # to ensure distributed exclusive access to specific data locations.
        self.dislocksdict = None # Note: This is later set in `setup_devices`.
        # Reference to the global barrier used for synchronizing all devices.
        self.barrier = None
        # Semaphore used for general synchronization, initialized with a count of 1 (unused in snippet).
        self.sem = Semaphore(1)
        # Semaphore used specifically for coordinating the setup phase, initialized with a count of 0.
        self.sem2 = Semaphore(0)
        # List to store references to all devices in the system (unused in provided snippet).
        self.all_devices = [] # This attribute is not populated within the class.
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a script has been received (usage is unclear in `assign_script`).
        self.script_received = Event()
        # List to temporarily store scripts assigned to this device before dispatching.
        self.scripts = []
        # Event to signal when the current timepoint's script assignments are done.
        self.timepoint_done = Event()
        # The main thread for the device, responsible for supervisor interaction and dispatching scripts.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Dictionary of Locks, one per data location, used for protecting sensor data during script execution.
        self.lock_set = None # Note: This is later set in `set_barriers`.

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the setup of global synchronization primitives (barrier and distributed data locks).

        This method identifies the "root" device (smallest device_id), which then calls a helper
        function `set_barriers` to initialize and distribute the shared resources.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Checks if the current device is the "root" device (i.e., has the smallest device_id).
        if self.is_root_device(devices) == 0:
            # Functional Utility: If it's the root device, it initiates the global barrier and lock setup.
            set_barriers(devices)

    def assign_script(self, script, location):
        """
        Assigns a script for execution at a specific data location.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the sensor data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signals that a script has been received.
        else:
            # If script is None, it signifies the end of script assignments for the current timepoint.
            # Functional Utility: Signals that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()

    def is_root_device(self, devices):
        """
        Determines if the current device is the "root" device among a list of devices.

        The root device is defined as the device with the smallest `device_id`.
        This is used for coordinating initialization tasks.

        Args:
            devices (list): A list of all Device objects in the simulation.

        Returns:
            int: 0 if the current device has the smallest `device_id` (is root), 1 otherwise.
        """
        is_root = 0
        for current_device in devices:
            if current_device.device_id < self.device_id:
                is_root = 1
                break
        return is_root

    def get_data(self, loc):
        """
        Retrieves sensor data for a given location.

        Note: This method does NOT acquire any locks (`self.lock` or `self.lock_set[loc]`).
        Locking for data access is expected to be handled by the caller (`ExecuteSript`).

        Args:
            loc (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or None if not found.
        """
        # Returns `sensor_data[loc]` if `loc` exists in `sensor_data`, otherwise `None`.
        return self.sensor_data[loc] if loc in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Note: This method does NOT acquire any locks (`self.lock` or `self.lock_set[location]`).
        Locking for data modification is expected to be handled by the caller (`ExecuteSript`).

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new sensor data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
        else:
            pass # No action if location is not in sensor_data.

    def shutdown(self):
        """
        Performs a graceful shutdown of the main device thread.
        """
        # Waits for the main device thread (`DeviceThread`) to complete its execution.
        self.thread.join()

def set_barriers(devices):
    """
    Helper function to initialize and distribute shared synchronization primitives
    (a global barrier and location-specific locks) among all devices.

    This function is intended to be called only once by the root device during setup.

    Args:
        devices (list): A list of all Device objects in the simulation.
    """
    # Dictionary to hold one `Lock` object for each unique data location across all devices.
    lock_set = {}
    # Functional Utility: Creates a global reusable barrier for all devices.
    barrier = ReusableBarrier(len(devices))
    # Block Logic: Assigns the created barrier and initializes the `lock_set` for each device.
    for current_device in devices:
        current_device.barrier = barrier
        for current_location in current_device.sensor_data:
            # For each unique location across devices, a dedicated lock is created.
            lock_set[current_location] = Lock()
                current_device.lock_set = lock_set
        
        class DeviceThread(Thread):
            """
            The main thread for a Device, responsible for coordinating with the supervisor,
            dispatching scripts to `ExecuteSript` workers, and participating in global synchronization.
            """
        
            def __init__(self, device):
                """
                Initializes a DeviceThread.
        
                Args:
                    device (Device): The `Device` instance this thread is associated with.
                """
                Thread.__init__(self, name="Device Thread %d" % device.device_id)
                self.device = device
        
            def run(self):
                """
                Main execution loop for the DeviceThread.
        
                Architectural Intent: Continuously fetches neighbor information from the supervisor.
                It orchestrates the execution of assigned scripts by launching `ExecuteSript`
                for each, waits for them to complete, and then participates in a global barrier
                for timepoint synchronization.
                """
                nr_threads = 8 # Represents a fixed number of worker threads to potentially utilize (unused in current dispatch logic).
                while True:
                    # Functional Utility: Clears the `timepoint_done` event to prepare for the next cycle.
                    self.device.timepoint_done.clear()
                    # Block Logic: Fetches the current list of neighboring devices from the supervisor.
                    neigh = self.device.supervisor.get_neighbours()
                    # Functional Utility: Participates in the global barrier. This likely marks the start
                    # of a new timepoint, ensuring all devices are synchronized before proceeding.
                    self.device.barrier.wait()
                    # Termination Condition: If no neighbors are returned (None), it signifies
                    # that the simulation is ending for this device, and the loop breaks.
                    if neigh is None:
                        break
                    # Block Logic: Waits until the current timepoint's script assignments are done.
                    # This ensures all scripts for this timepoint have been added to `self.device.scripts`.
                    self.device.timepoint_done.wait()
                    perform_s = [] # List to hold scripts that need to be executed.
                    # Block Logic: Copies scripts from `self.device.scripts` to a local list for processing.
                    for script in self.device.scripts:
                        perform_s.append(script)
                    
                    # Functional Utility: Clears the device's script list as they are about to be processed.
                    self.device.scripts = []
        
                    threads = [] # List to store `ExecuteSript` worker threads.
                    # Block Logic: Creates and appends `ExecuteSript` threads.
                    # Note: `xrange` is Python 2.x specific; for Python 3, `range` should be used.
                    # Note: `perform_s` is passed to each thread, but scripts are popped within `ExecuteSript.run()`,
                    # indicating `perform_s` is treated as a shared queue.
                    for i in xrange(nr_threads):
                        threads.append(ExecuteSript(self.device, neigh, perform_s))
                    # Block Logic: Starts all created `ExecuteSript` threads concurrently.
                    for i in xrange(nr_threads):
                        threads[i].start()
                    # Block Logic: Waits for all launched `ExecuteSript` instances to complete their execution.
                    for i in xrange(nr_threads):
                        threads[i].join()
                    # Functional Utility: Participates in the global barrier again. This likely ensures
                    # all devices have finished executing their scripts before proceeding to the next timepoint.
                    self.device.barrier.wait()
                    # Functional Utility: Clears the `timepoint_done` event to prepare for the next timepoint.
                    self.device.timepoint_done.clear() # This line seems redundant as it's also done at the beginning of the loop.
        
        class ReusableBarrier(object):
            """
            A reusable barrier synchronization primitive implemented using a `Condition` variable.
        
            This barrier allows a fixed number of threads (`num_threads`) to wait for
            each other to reach a common point before any can proceed. It is designed
            to be reusable across multiple synchronization points within a larger simulation loop.
            """
            
            def __init__(self, num_threads):
                """
                Initializes a ReusableBarrier.
        
                Args:
                    num_threads (int): The total number of threads that must arrive
                                       at the barrier before any can proceed.
                """
                self.num_threads = num_threads
                # `count_threads`: Counter for threads currently waiting at the barrier.
                self.count_threads = self.num_threads
                # `cond`: A condition variable used for synchronization (waiting and notifying).
                self.cond = Condition()
            
            def wait(self):
                """
                Causes the calling thread to wait at the barrier until all other
                `num_threads` threads have also called `wait()`.
        
                The last thread to arrive releases all waiting threads and resets the barrier.
                """
                
                self.cond.acquire() # Acquires the lock associated with the condition variable.
                self.count_threads -= 1 # Decrements the counter of threads yet to arrive.
                if self.count_threads == 0: # Conditional Logic: If this is the last thread to arrive.
                    
                    self.cond.notify_all() # Notifies all threads waiting on this condition.
                    
                    self.count_threads = self.num_threads # Resets the counter for barrier reusability.
                else: # Conditional Logic: If this is not the last thread to arrive.
                    
                    self.cond.wait() # Waits (blocks) until notified by the last arriving thread.
                
                self.cond.release() # Releases the lock associated with the condition variable.
        
            def print_barrier(self):
                """
                Prints the current state of the barrier (number of threads and current count).
                """
                print self.num_threads, self.count_threads # Note: `print` is a statement in Python 2.x, function in Python 3.x.
        
        class ExecuteSript(Thread):
            """
            A worker thread responsible for executing a single script at a specific data location.
        
            This thread collects data from its associated device and its neighbors,
            executes the provided script, and updates the relevant sensor data,
            ensuring data consistency via location-specific locks.
            """
        
        
            def __init__(self, device, neighbours, perform_script):
                """
                Initializes an ExecuteSript thread.
        
                Args:
                    device (Device): The `Device` instance this thread is associated with.
                    neighbours (list): A list of neighboring Device objects.
                    perform_script (list): A shared list of scripts (tuples of (script_object, location)) to be processed.
                """
                Thread.__init__(self)
                self.device = device
                self.neighbours = neighbours
                self.perform_script = perform_script # This is the shared list from `DeviceThread`.
        
            def run(self):
                """
                Main execution logic for the ExecuteSript thread.
        
                Architectural Intent: Atomically extracts a script from the shared `perform_script` list,
                then ensures thread-safe access to a specific data location across devices.
                It collects data, executes the assigned script, and updates the result on
                all relevant devices (self and neighbors) using location-specific locks.
                """
                # Block Logic: Extracts a script to execute from the shared list.
                # This implies `perform_script` is treated as a queue/stack, and this operation needs
                # to be thread-safe if multiple `ExecuteSript` threads are started concurrently.
                if len(self.perform_script) != 0: # Checks if there are scripts left to process.
                    (script, location) = self.perform_script.pop() # Atomically removes and gets the last script.
                    collected = [] # List to store all data relevant to the script.
                    
                    # Functional Utility: Acquires the location-specific lock.
                    # This ensures exclusive access to the data at `location` across all devices.
                    self.device.lock_set[location].acquire()
        
                    # Block Logic: Collects sensor data from neighboring devices.
                    for current_neigh in self.neighbours:
                        data = current_neigh.get_data(location) # Note: `get_data` does not acquire locks internally.
                        collected.append(data)
                    # Block Logic: Collects sensor data from its own device.
                    data = self.device.get_data(location)
                    collected.append(data)
        
                    # Block Logic: If any data was collected, executes the script and updates data.
                    if collected != []:
                        # Functional Utility: Executes the assigned script with the collected data.
                        result = script.run(collected)
                        
                        # Block Logic: Updates data on neighboring devices.
                        for current_neigh in self.neighbours:
                            current_neigh.set_data(location, result) # Note: `set_data` does not acquire locks internally.
                        # Block Logic: Updates data on its own device.
                        self.device.set_data(location, result)
        
                    # Functional Utility: Releases the location-specific lock.
                    self.device.lock_set[location].release()
        
