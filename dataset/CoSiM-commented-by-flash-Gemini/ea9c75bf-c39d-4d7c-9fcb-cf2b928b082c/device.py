"""
This module implements a simulated distributed system designed for cooperative
data processing among interconnected 'devices'. It defines the core components
and their interactions, leveraging Python's `threading` module for concurrency
and synchronization.

The key components include:
- `ReusableBarrier`: A custom barrier synchronization primitive that allows a
  fixed number of threads to wait for each other at a common point and then
  proceed. This is crucial for coordinating across multiple devices or threads
  within a device.
- `Device`: Represents an individual node in the distributed system. Each device
  manages its own sensor data, holds a list of scripts to execute, and orchestrates
  its main processing thread (`DeviceThread`) and potentially other worker threads.
- `ScriptThread`: A specialized thread responsible for executing a single script
  for a specific sensor location. It fetches data from the parent device and its
  neighbors, runs the script, and then updates the relevant data on the devices.
- `DeviceThread`: The main execution thread for each `Device`. It handles neighbor
  discovery, manages the lifecycle of `ScriptThread`s for assigned tasks, and
  participates in global synchronization using the `ReusableBarrier`.

The system models a scenario where devices collaboratively process data, often
requiring data exchange with adjacent nodes and synchronized progression through
processing timepoints.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue


class ReusableBarrier(object):
    """
    Implements a reusable barrier synchronization primitive for a fixed number of threads.

    This barrier allows a group of `num_threads` to halt execution at a specific
    point (`wait` method) until all threads in the group have arrived. Once all
    threads have reached the barrier, they are all released simultaneously.
    The barrier can then be reset and reused for subsequent synchronization points.

    The implementation uses a two-phase approach to ensure reusability without
    deadlock. Each phase involves:
    - A `count_lock` to atomically decrement a thread counter.
    - A `Semaphore` to block arriving threads and release them when the counter reaches zero.

    Attributes:
        num_threads (int): The total number of threads expected to participate in the barrier.
        count_threads1 (list): Internal counter for the first phase of the barrier.
        count_threads2 (list): Internal counter for the second phase of the barrier.
        count_lock (Lock): A lock to protect access to the thread counters.
        threads_sem1 (Semaphore): Semaphore for the first phase of thread synchronization.
        threads_sem2 (Semaphore): Semaphore for the second phase of thread synchronization.
    """
    def __init__(self, num_threads):
        """
        Initializes a new `ReusableBarrier` instance.

        Sets up the internal state required for barrier synchronization,
        including thread counters, a lock for atomic updates, and two
        semaphores for managing thread blocking and release across two phases.

        Args:
            num_threads (int): The total number of threads that will
                                participate in this barrier.
        """
        
        self.num_threads = num_threads
        # `count_threads1` and `count_threads2` are lists to allow mutation within the `phase` method.
        # They track the number of threads yet to reach the barrier in each phase.
        self.count_threads1 = [self.num_threads] 
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock() # Protects `count_threads` from race conditions during decrement/reset.
        
        self.threads_sem1 = Semaphore(0) # Initially blocks all threads for the first phase.
        
        self.threads_sem2 = Semaphore(0) # Initially blocks all threads for the second phase.

    def wait(self):
        """
        Causes the calling thread to wait until all other threads (`num_threads`)
        have also called this method.

        This method implements the two-phase barrier synchronization. A thread
        must successfully pass through both phases before being allowed to proceed.
        This design prevents a "last thread in" from potentially getting ahead
        of "first thread out" in a subsequent barrier use.
        """
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Manages one phase of the barrier synchronization.

        This internal method is called twice by the `wait` method to implement
        the two-phase barrier. It atomically decrements the count of threads
        yet to reach the barrier and, if it's the last thread, releases all
        waiting threads. Otherwise, it blocks the calling thread.

        Args:
            count_threads (list): A list containing the current count of threads
                                  that have not yet reached this phase of the barrier.
                                  (Wrapped in a list to allow in-place modification within a function).
            threads_sem (Semaphore): The semaphore associated with this phase, used
                                     to block and release threads.
        """
        
        with self.count_lock:
            count_threads[0] -= 1
            
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Represents an individual node (device) within the simulated distributed system.

    Each `Device` manages its own sensor data, holds a list of scripts to execute,
    and orchestrates its main processing thread (`DeviceThread`). It acts as a
    central hub for its own operations, including receiving script assignments,
    accessing local sensor data, and interacting with neighbor devices.
    It utilizes synchronization primitives like `Event` and `ReusableBarrier`
    to coordinate its activities within the broader distributed environment.

    Attributes:
        device_id (int): A unique integer identifier for this device.
        sensor_data (dict): A dictionary holding the device's local sensor readings.
                            Keys are sensor locations/types, and values are the corresponding data.
        supervisor (Supervisor): A reference to the central supervisor orchestrating the devices.
        scripts (list): A list of tuples, where each tuple is `(script_object, location)`,
                        representing scripts assigned to this device for execution.
        timepoint_done (Event): An `Event` object used to signal when a timepoint's
                                script assignments have been fully processed.
        barrier (ReusableBarrier): A reference to a `ReusableBarrier` object, used
                                   for global synchronization across devices.
        thread (DeviceThread): The main `DeviceThread` instance responsible for
                               this device's primary operational loop.
        location_locks (list): A list of `Lock` objects, one for each unique sensor location,
                               to protect concurrent access to `sensor_data` at that location.
                               This is initialized by the master device (`device_id == 0`).
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new `Device` instance.

        Sets up the device's unique identifier, its local sensor data, and a reference
        to the central `supervisor`. It also initializes internal state variables and
        starts its dedicated `DeviceThread`.

        Args:
            device_id (int): A unique integer identifier for this device.
            sensor_data (dict): A dictionary representing the sensor readings
                                collected by this device. Keys are sensor locations
                                or types, and values are the corresponding data.
            supervisor (object): An object representing the central supervisor
                                 that orchestrates the devices.
        
        Attributes Initialized:
            self.device_id (int): Unique ID of the device.
            self.sensor_data (dict): Stores sensor data.
            self.supervisor (object): Reference to the supervisor.
            self.scripts (list): An empty list to store assigned scripts. Each element is `(script, location)`.
            self.timepoint_done (Event): An Event object, initially cleared, to signal completion of timepoint processing.
            self.barrier (ReusableBarrier): Initialized to None, will hold a reference to the shared barrier.
            self.thread (DeviceThread): An instance of `DeviceThread` for this device, which is immediately started.
            self.location_locks (list): Initialized to None, will hold a list of Locks for sensor locations.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()
        self.location_locks = None

    def __str__(self):
        """
        Provides a human-readable string representation of the Device object.

        This method is primarily used for logging, debugging, and identification
        purposes within the simulated distributed system.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the device with a list of all devices in the system and initializes
        shared synchronization primitives, specifically when called on the master device (`device_id == 0`).

        This method is crucial for establishing the topology of the distributed system
        and preparing the necessary synchronization mechanisms (barriers and per-location locks)
        before the devices begin their main operational loops.

        Args:
            devices (list): A list of all `Device` objects participating in the simulation.

        Details:
        - The `devices` list is implicitly used to determine the total number of participants
          for barrier initialization.
        - If `self.device_id` is 0 (acting as the master device):
            - It initializes a `ReusableBarrier` (`self.barrier`) with the total number of devices.
            - It aggregates all unique sensor locations across all devices in the system.
            - It creates a list of `Lock` objects (`self.location_locks`), one for each unique
              sensor location. These locks are vital for protecting concurrent read/write
              access to data at specific locations across different devices.
            - It then propagates the initialized `barrier` and `location_locks` references to
              all other devices in the system, ensuring they all share the same synchronization objects.
        """
        
        # self.devices = devices # This line is not present in the current code, but commented here for clarity in docstring.

        # The following lines appear to be from a previous iteration or a different version
        # of the code and are not directly functional in this current `setup_devices` context.
        # self.count_threads = [len(self.devices)] 

        if 0 == self.device_id:
            
            # Initialize a ReusableBarrier for all devices.
            self.barrier = ReusableBarrier(len(devices))
            
            locations = []
            # Collect all unique sensor locations from all devices.
            for device in devices:
                for location in device.sensor_data:
                    if location not in locations:
                        locations.append(location)
            
            # Initialize a list of locks for each unique sensor location.
            self.location_locks = []
            for _ in range(len(locations)):
                self.location_locks.append(Lock())
            
            # Propagate the shared barrier and location locks to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific sensor location.

        This method is used by the supervisor to provide the device with tasks.
        It appends the script and its associated location to an internal queue
        (`self.scripts`). A special case exists when `script` is `None`, which
        acts as a signal to indicate the completion of a processing timepoint.

        Args:
            script (object or None): The script object to be executed, or `None`
                                     to signal a control event (e.g., end of timepoint).
            location (int): The sensor location or identifier relevant to the script.

        Details:
        - The `(script, location)` tuple is appended to `self.scripts`.
        - If `script` is `None`:
            - The `self.timepoint_done` event is set. This signals to the
              `DeviceThread` that script assignments for the current timepoint
              are complete, prompting it to process the accumulated scripts.
        """
        

        if script is not None:
            self.scripts.append((script, location))
        else:
            
            
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location from the device's local store.

        This method provides read access to the `sensor_data` dictionary.
        It checks for the existence of the `location` in `sensor_data` before
        attempting to return data.

        Args:
            location (int): The identifier for the sensor location whose data is to be retrieved.

        Returns:
            Any: The sensor data associated with the given `location`, or `None` if
                 the `location` is not found in `self.sensor_data`.
        """
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location in the device's local store.

        This method provides write access to the `sensor_data` dictionary.
        It only updates the data if the specified `location` already exists
        within the `sensor_data`, ensuring that new locations are not implicitly
        created through this method.

        Args:
            location (int): The identifier for the sensor location whose data is to be updated.
            data (Any): The new data value to be stored for the given `location`.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates a graceful shutdown of the device by joining its main `DeviceThread`.

        This method ensures that the device's primary operational thread completes
        its execution before the device is fully decommissioned. Any scripts or
        operations managed by the `DeviceThread` should be completed or handled
        within that thread's logic prior to calling `shutdown`.
        """
        
        self.thread.join()


class ScriptThread(Thread):
    """
    A specialized thread responsible for executing a single script for a specific
    sensor location within a device.

    `ScriptThread`s are spawned by the `DeviceThread` to parallelize the execution
    of individual scripts. Each `ScriptThread` encapsulates the logic for:
    - Acquiring the necessary lock for the sensor data location.
    - Collecting relevant sensor data from its parent `device` and its `neighbours`.
    - Running the assigned `script` with the collected data.
    - Updating the sensor data on the parent `device` and its `neighbours` with
      the results of the script execution.

    This class helps in distributing the script execution workload and managing
    concurrent access to shared sensor data.
    """
    

    def __init__(self, device, script, location, neighbours):
        """
        Initializes a new `ScriptThread` instance.

        Sets up the thread with references to its parent `Device`, the `script`
        to execute, the target `location`, and a list of `neighbours` to interact with.

        Args:
            device (Device): The parent `Device` instance that spawned this thread.
            script (object): The script object to be executed.
            location (int): The sensor location relevant to this script's execution.
            neighbours (list): A list of neighboring `Device` objects that this
                                script might interact with (e.g., fetch data from).
        """
        
        Thread.__init__(self)


        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        The main execution logic for the `ScriptThread`.

        This method encapsulates the process of executing a single script:
        - It acquires a lock for the specific sensor `location` to ensure
          exclusive access to the data during script execution and update.
        - It collects sensor data relevant to the `location` from all
          `neighbours` and from the parent `device` itself.
        - If collected data is not empty, it executes the assigned `script`
          with the aggregated data.
        - Finally, it propagates the result of the script execution back to
          all `neighbours` and updates the parent `device`'s own sensor data
          for that `location`.
        """
        with self.device.location_locks[self.location]:
            script_data = []
            
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            if script_data != []:
                
                result = self.script.run(script_data)
                


                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    The main execution thread for a `Device` instance.

    This thread is responsible for the overall operation of a device within the
    simulated distributed system. Its primary tasks include:
    - Periodically discovering or updating its list of neighboring devices from the `supervisor`.
    - Waiting for `timepoint_done` events, which signal the assignment of new scripts.
    - Orchestrating the execution of assigned scripts by spawning `ScriptThread`s
      for each script and waiting for their completion.
    - Synchronizing with other `DeviceThread`s using a shared `ReusableBarrier`
      to ensure coordinated progression through simulation timepoints.
    - Handling termination signals from the `supervisor`.

    This thread acts as the control center for a device's distributed data processing activities.
    """
    

    def __init__(self, device):
        """
        Initializes a new `DeviceThread` instance.

        Sets up the thread with a descriptive name and establishes a strong
        reference to the `Device` object it is responsible for managing.

        Args:
            device (Device): The `Device` instance that this thread will serve.
                             This reference allows the thread to access and
                             manipulate the parent device's state and resources.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the `DeviceThread`.

        This method defines the continuous operation of a device within the
        simulated distributed system. It performs the following sequence of operations
        in a loop until a termination condition is met:

        1.  **Neighbor Discovery**: Fetches the latest list of neighbors from the
            `supervisor`. If `None` is returned, it signals a termination,
            breaks the loop, and ends the thread.
        2.  **Wait for Scripts**: Blocks until the `timepoint_done` event is set,
            indicating that scripts for the current timepoint have been assigned.
        3.  **Script Execution**:
            - If neighbors are available, it iterates through all assigned scripts
              in `self.device.scripts`.
            - For each script, it creates a new `ScriptThread` and starts it,
              effectively parallelizing the script execution.
            - It then waits for all spawned `ScriptThread`s to complete their
              execution using `thread.join()`.
        4.  **Reset Events**: Clears the `timepoint_done` event to prepare for
            the next timepoint's script assignments.
        5.  **Global Synchronization**: Waits at the `self.device.barrier` to
            synchronize with all other `DeviceThread`s in the system, ensuring
            that all devices complete their timepoint processing before proceeding
            to the next.
        """
        while True:
            
            vecini = self.device.supervisor.get_neighbours()
            if vecini is None:
                break
            
            self.device.timepoint_done.wait()
            threads = []
            
            if len(vecini) != 0:
                for (script, locatie) in self.device.scripts:
                    thread = ScriptThread(self.device, script, locatie, vecini)
                    threads.append(thread)
                    thread.start()
                for thread in threads:
                    thread.join()
            
            
            self.device.timepoint_done.clear()
            
            
            
            self.device.barrier.wait()
