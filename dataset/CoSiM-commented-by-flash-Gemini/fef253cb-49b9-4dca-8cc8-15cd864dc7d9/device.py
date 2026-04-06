"""
This module implements a simulated distributed system, modeling the behavior
of interconnected 'devices' that process sensor data and coordinate through
a supervisor. It leverages Python's `threading` module to manage concurrency
within and between devices, utilizing `Event` objects and `Lock`s for
synchronization, along with a custom `barrier.ReusableBarrierCond` for global
synchronization points.

The system is composed of three main classes:
- `Device`: Represents an individual node in the distributed system, holding
  sensor data, managing its execution threads, and handling script assignments
  and data access.
- `DeviceThread`: The primary execution thread for each `Device`, responsible
  for discovering neighbors, continuously waiting for and executing assigned
  scripts, and synchronizing with other device threads.
- `WorkerThread`: Auxiliary threads spawned by each `Device` to assist the
  `DeviceThread` in executing scripts concurrently, particularly when scripts
  involve fetching and setting data from/to neighbors.

The architecture aims to simulate a cooperative processing environment where
devices execute scripts that might read data from their own sensors and
from neighboring devices, process it, and then update their own or neighbors'
sensor data. Synchronization is critical to ensure data consistency and
coordinated execution across the distributed nodes.
"""

import barrier
from threading import Event, Thread, Lock


class Device(object):
    """
    Represents an individual node within the simulated distributed system.

    Each `Device` instance is responsible for managing its local sensor data,
    executing assigned scripts, and coordinating with other devices and a central
    `supervisor`. It orchestrates multiple threads (`DeviceThread` and `WorkerThread`s)
    to handle concurrent operations, and employs various synchronization primitives
    to ensure data consistency and controlled execution flow across the system.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's local sensor readings,
                            where keys represent sensor locations/types and values are data.
        supervisor (Supervisor): A reference to the central supervisor orchestrating the devices.
        script_received (Event): Signals when a new script has been assigned to the device.
        script_taken (Event): Signals when a script has been taken for execution.
        assign_script_none (Event): An event related to script assignment, likely for specific scenarios.
        script_de_orice_fel (Event): Generic script event.
        assign_script_not_none (Event): Another event related to script assignment.
        bariera (barrier.ReusableBarrierCond): A synchronization barrier for coordinating main device threads.
        bariera_join (barrier.ReusableBarrierCond): A synchronization barrier for coordinating all worker threads.
        barrier_time (barrier.ReusableBarrierCond): A synchronization barrier for timing-related coordination.
        flag_terminate (bool): A flag indicating if the device's threads should terminate.
        script_sent (Lock): A lock to protect access to the `scripts` list during script assignment.
        script_sent_thread (Lock): Another lock related to script sending, potentially for thread-specific contexts.
        barrier_lock (Lock): A lock to protect access to barrier-related operations.
        counter (int): A counter used in barrier synchronization, typically initialized to the number of devices.
        flag_received (Event): Signals internal state about receiving something.
        got_neighbours (Event): Signals when the device has successfully received its neighbors list from the supervisor.
        barrier_clear_events (barrier.ReusableBarrierCond): A barrier for clearing events in a coordinated manner.
        flag_script_received (bool): Internal flag indicating if a script was received.
        flag_script_taken (bool): Internal flag indicating if a script was taken.
        flag_assign_script (int): 2. Internal state for script assignment (e.g., 0, 1, 2).
        flag_get_neigbours (bool): False. Internal flag for neighbor fetching state.
        get_neighbours_lock (Lock): A lock to protect neighbor-related operations.
        index_lock (Lock): A lock to protect the script index (`i`) during concurrent script processing.
        i (int): An index used to iterate through the assigned scripts.
        scripts (list): A list of tuples, where each tuple is `(script_object, location)`.
        neighbours (list): A list of other `Device` objects that are considered neighbors.
        devices (list): A list of all `Device` objects in the system (only populated for device_id 0).
        count_threads (list): A list containing the number of devices.
        locations_locks (list): A list of `Lock` objects, one for each unique sensor location,
                                to protect concurrent access to `sensor_data` at that location.
        timepoint_done (Event): Signals when a timepoint's processing is done.
        initialize (Event): Signals that the device's setup is complete and threads can proceed.
        put_take_data (Lock): A lock to protect concurrent read/write access to `sensor_data`.
        thread (DeviceThread): The main `DeviceThread` instance for this device.
        threads (list): A list of `WorkerThread` instances spawned by this device.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new `Device` instance.

        Sets up the device's unique identifier, sensor data, and a reference to
        its supervisor. It also initializes various synchronization primitives
        (Events and Locks) and internal state flags crucial for managing
        concurrent operations and inter-thread communication within the simulated
        distributed system.

        Args:
            device_id (int): A unique integer identifier for this device.
            sensor_data (dict): A dictionary representing the sensor readings
                                collected by this device. Keys are sensor locations
                                or types, and values are the corresponding data.
            supervisor (Supervisor): An object representing the central supervisor
                                     that orchestrates the devices.

        Attributes Initialized:
            self.device_id (int): Unique ID of the device.
            self.sensor_data (dict): Stores sensor data.
            self.supervisor (Supervisor): Reference to the supervisor.
            self.script_received (Event): Initially cleared. Set when a script is assigned.
            self.script_taken (Event): Initially cleared. Set when a script starts execution.
            self.assign_script_none (Event): Initially cleared. Specific script assignment event.
            self.script_de_orice_fel (Event): Initially cleared. Generic script event.
            self.assign_script_not_none (Event): Initially cleared. Specific script assignment event.
            self.bariera (barrier.ReusableBarrierCond): Placeholder for a barrier, set during setup.
            self.bariera_join (barrier.ReusableBarrierCond): Placeholder for a barrier, set during setup.
            self.barrier_time (barrier.ReusableBarrierCond): Placeholder for a barrier, set during setup.
            self.flag_terminate (bool): False. Indicates the device's threads should not terminate.
            self.script_sent (Lock): Initialized as an unlocked Lock. Protects script assignment.
            self.script_sent_thread (Lock): Initialized as an unlocked Lock. Related to script sending.
            self.barrier_lock (Lock): Initialized as an unlocked Lock. Protects barrier operations.
            self.counter (int): 0. Used in barrier synchronization.
            self.flag_received (Event): Initially cleared. Signals internal receive state.
            self.got_neighbours (Event): Initially cleared. Signals neighbor list reception.
            self.barrier_clear_events (barrier.ReusableBarrierCond): Placeholder for a barrier.
            self.flag_script_received (bool): False. Internal flag for script reception state.
            self.flag_script_taken (bool): False. Internal flag for script execution state.
            self.flag_assign_script (int): 2. Internal state for script assignment.
            self.flag_get_neigbours (bool): False. Internal flag for neighbor fetching state.
            self.get_neighbours_lock (Lock): Initialized as an unlocked Lock. Protects neighbor access.
            self.index_lock (Lock): Initialized as an unlocked Lock. Protects script index `i`.
            self.i (int): 0. Index for iterating through assigned scripts.
            self.scripts (list): Empty list. Stores assigned scripts and their locations.
            self.neighbours (list): None. Stores references to neighbor devices.
            self.devices (list): Empty list. Stores references to all devices (for device_id 0).
            self.count_threads (list): A list containing the number of devices.
            self.locations_locks (list): Empty list. Stores locks for sensor data locations.
            self.timepoint_done (Event): Initially cleared. Signals timepoint processing completion.
            self.initialize (Event): Initially cleared. Signals device setup completion.
            self.put_take_data (Lock): Initialized as an unlocked Lock. Protects `sensor_data` access.
            self.thread (DeviceThread): An instance of `DeviceThread` for this device.
            self.threads (list): Empty list. Stores `WorkerThread` instances.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.script_taken = Event()
        self.assign_script_none = Event()
        self.script_de_orice_fel = Event()
        self.assign_script_not_none = Event()
        self.bariera = None
        self.bariera_join = None
        self.barrier_time = None
        self.flag_terminate = False
        self.script_sent = Lock()
        self.script_sent_thread = Lock()
        self.barrier_lock = Lock()
        self.counter = 0
        self.flag_received = Event()
        self.got_neighbours = Event()
        self.barrier_clear_events = None
        self.flag_script_received = False
        self.flag_script_taken = False
        self.flag_assign_script = 2
        self.flag_get_neigbours = False
        self.get_neighbours_lock = Lock()
        self.index_lock = Lock()
        self.i = 0
        self.scripts = []
        self.neighbours = None
        self.devices = []
        self.count_threads = []
        self.locations_locks = []
        self.timepoint_done = Event()
        self.initialize = Event()
        self.put_take_data = Lock()
        self.thread = DeviceThread(self)
        self.threads = []

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
        shared synchronization primitives, especially when called on the master device (`device_id == 0`).

        This method is crucial for establishing the topology of the distributed system
        and preparing the necessary synchronization mechanisms (barriers and per-location locks)
        before starting the device's operational threads.

        Args:
            devices (list): A list of all `Device` objects participating in the simulation.

        Details:
        - Stores the list of all devices locally (`self.devices`).
        - If `self.device_id` is 0 (master device):
            - It aggregates all unique sensor locations from all devices.
            - It initializes a list of `Lock` objects (`self.locations_locks`), one for each
              unique sensor location, to protect concurrent access to sensor data at those locations.
            - It initializes several `ReusableBarrierCond` instances (`self.bariera`,
              `self.bariera_join`, `self.barrier_time`, `self.barrier_clear_events`)
              which are essential for coordinating execution phases across multiple device threads.
            - It propagates these initialized barriers and `locations_locks` to all other devices
              in the system, ensuring they all share the same synchronization objects.
        - The main `DeviceThread` (`self.thread`) is started.
        - Seven `WorkerThread` instances are spawned and started for this device, which will
          assist in script execution.
        - Finally, `self.initialize` event is set to signal that setup is complete.
        """
        
        self.devices = devices
        self.count_threads = [len(self.devices)]

        if self.device_id == 0:
            
            locations = []
            for device in self.devices:
                l = []
                for key in device.sensor_data.keys():
                    l.append(key)
                for locatie in l:
                    locations.append(locatie)
            maxim = max(locations)
            self.locations_locks = [None] * (maxim + 1)
            for locatie in locations:
                if self.locations_locks[locatie] is None:
                    lock = Lock()
                    self.locations_locks[locatie] = lock

            self.bariera = barrier.ReusableBarrierCond(len(self.devices))
            num_threads = len(self.devices) * 8
            self.bariera_join = barrier.ReusableBarrierCond(num_threads)
            self.barrier_time = barrier.ReusableBarrierCond(num_threads)
            self.barrier_clear_events = barrier.ReusableBarrierCond(num_threads)

            for device in self.devices:
                device.i = 0
                device.bariera = self.bariera
                device.counter = len(self.devices)
                device.barrier_time = self.barrier_time
                device.barrier_clear_events = self.barrier_clear_events
                device.locations_locks = self.locations_locks

        self.thread.start()
        i = 0
        while i < 7:
            dev = WorkerThread(self)
            dev.start()
            self.threads.append(dev)
            i = i + 1
        
        self.initialize.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific sensor location.

        This method is used by the supervisor to provide the device with tasks.
        It appends the script and its associated location to an internal queue.
        A special case exists when `script` is `None`, which acts as a signal
        to indicate the completion of a processing timepoint or a control command,
        triggering specific events.

        Args:
            script (object or None): The script object to be executed, or `None`
                                     to signal a control event.
            location (int): The sensor location or identifier relevant to the script.

        Details:
        - The `self.script_sent` lock is acquired to ensure thread-safe access to
          the `self.scripts` list, preventing race conditions during script assignment.
        - The `(script, location)` tuple is appended to `self.scripts`.
        - If `script` is `None`:
            - `self.script_received` event is set, indicating that a script (or a control signal)
              has been assigned, potentially waking up waiting worker threads.
            - `self.timepoint_done` event is set, signaling the completion of a timepoint's
              script assignments.
        """
        
        with self.script_sent:
            if script is not None:
                self.scripts.append((script, location))
            else:
                self.scripts.append((script, location))
                self.script_received.set()
                self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location from the device's local store.

        This method provides thread-safe access to the `sensor_data` dictionary,
        allowing scripts and other components to read current sensor values.

        Args:
            location (int): The identifier for the sensor location whose data is to be retrieved.

        Returns:
            Any: The sensor data associated with the given `location`, or `None` if
                 the `location` is not found in `self.sensor_data`.

        Details:
        - The `self.put_take_data` lock is acquired to prevent race conditions when
          accessing `self.sensor_data` concurrently.
        - It checks if the `location` exists as a key in `self.sensor_data`.
        - If the location exists, its corresponding value (sensor data) is returned.
        - Otherwise, `None` is returned, indicating no data for that location.
        """
        
        with self.put_take_data:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location in the device's local store.

        This method provides thread-safe access to the `sensor_data` dictionary,
        allowing scripts and other components to modify sensor values. The update
        only occurs if the `location` already exists in the `sensor_data`.

        Args:
            location (int): The identifier for the sensor location whose data is to be updated.
            data (Any): The new data value to be stored for the given `location`.

        Details:
        - The `self.put_take_data` lock is acquired to prevent race conditions when
          accessing `self.sensor_data` concurrently.
        - It checks if the `location` exists as a key in `self.sensor_data`. This
          prevents adding new locations through this method, ensuring data integrity.
        - If the location exists, its corresponding value (sensor data) is updated with `data`.
        """
        
        with self.put_take_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates a graceful shutdown of the device and its associated threads.

        This method ensures that all `WorkerThread` instances and the main
        `DeviceThread` complete their current operations and terminate cleanly
        before the `Device` object is fully decommissioned.

        Details:
        - It iterates through all `WorkerThread` instances stored in `self.threads`
          and calls `join()` on each, blocking until each worker thread finishes execution.
        - After all worker threads have joined, it calls `join()` on the main
          `DeviceThread` (`self.thread`), waiting for its termination.
        - This prevents abrupt termination and potential data corruption or
          incomplete operations from ongoing threads.
        """
        
        for thread in self.threads:
            thread.join()
        self.thread.join()

class DeviceThread(Thread):
    """
    The main dedicated thread of execution for a `Device` instance.

    Each `Device` object has one `DeviceThread` which is responsible for
    the primary operational loop of the device. This includes continuously
    polling for network neighbors, processing assigned scripts (or delegating
    to worker threads), and coordinating through synchronization barriers
    and events managed by its parent `Device`.

    This thread maintains the active state of the device within the simulated
    distributed environment.
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
        simulated distributed system. It repeatedly performs neighbor discovery,
        waits for new scripts to be assigned, processes these scripts (which
        may involve inter-device data exchange), and synchronizes with other
        threads/devices using events and barriers. The loop terminates when
        a termination signal is received from the supervisor.
        """
        while True:
            # Fetch neighbors from the supervisor. This is a critical step for inter-device communication.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            # Check for a termination signal from the supervisor (e.g., if no more neighbors or a global shutdown).
            if self.device.neighbours is None:
                self.device.flag_terminate = True # Set the termination flag for this device.
                self.device.got_neighbours.set() # Signal that neighbor acquisition is complete (or terminated).
                break # Exit the main loop, terminating the thread.
            
            self.device.got_neighbours.set() # Signal that neighbors have been successfully acquired.
            
            # Wait for a script to be received by the device. This blocks until `script_received` is set.
            self.device.script_received.wait()
            while True:
                # Protect the script index `i` during concurrent access from multiple worker threads.
                with self.device.index_lock:
                    # If all scripts currently assigned to the device have been processed, break this inner loop.
                    if self.device.i >= len(self.device.scripts):
                        break
                    # Retrieve a script and its associated location using modulo to handle script cycling or partitioning.
                    (script, location) = self.device.scripts[self.device.i % 8]
                    self.device.i = self.device.i + 1 # Advance the script index.
                # Process the script if it is not a termination/control signal.
                if script is not None:
                    # Acquire the lock specific to the sensor data location to ensure exclusive access.
                    lock = self.device.locations_locks[location]
                    with lock:
                        
                        script_data = [] # Initialize a list to aggregate data for the script.
                        
                        # Collect data from neighboring devices for the specified location.
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        # Collect data from the device's own sensor for the specified location.
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        # If there is data to process, run the script.
                        if script_data != []:
                            
                            result = script.run(script_data) # Execute the script with collected data.

                            # Update sensor data on neighboring devices with the script's result.
                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            # Update the device's own sensor data with the script's result.
                            self.device.set_data(location, result)
                # If the script is None, it acts as a signal to break the script processing loop.
                if script is None:
                    break

            # Wait at a barrier to clear events, synchronizing with other worker threads.
            self.device.barrier_clear_events.wait()
            self.device.script_received.clear() # Clear the script received event, waiting for the next assignment.
            self.device.got_neighbours.clear() # Clear the got_neighbours event for the next cycle.
            # Wait at a time barrier, synchronizing the progression of timepoints across devices.
            self.device.barrier_time.wait()

class WorkerThread(Thread):
    """
    An auxiliary worker thread designed to assist a `Device` instance in script execution.

    Each `Device` can spawn multiple `WorkerThread`s to parallelize the execution
    of assigned scripts. These threads primarily focus on processing scripts that
    involve retrieving data from the parent `Device` and its neighbors, running
    a script with the collected data, and then updating the sensor data on the
    parent and neighboring devices.

    `WorkerThread`s participate in the same synchronization events and barriers
    as the `DeviceThread` to ensure coordinated data processing across the
    simulated distributed system. They are responsible for handling their
    share of the scripts assigned to the parent device.
    """

    def __init__(self, device):
        """
        Initializes a new `WorkerThread` instance.

        Sets up the thread with a descriptive name and establishes a strong
        reference to the `Device` object it is intended to serve. This allows
        the worker thread to access and manipulate the parent device's shared
        state and resources, participating in its script execution and
        synchronization mechanisms.

        Args:
            device (Device): The `Device` instance that this worker thread will serve.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for an auxiliary `WorkerThread`.

        This method defines the continuous operation of a worker thread,
        assisting its parent `Device` in processing assigned scripts. It
        repeatedly waits for neighbor information, then for script assignments,
        executes its share of scripts (which may involve fetching data from
        or updating data on neighboring devices), and participates in global
        synchronization through events and barriers. The loop terminates
        when a `flag_terminate` signal is received from the parent device.
        """
        while True:
            # Wait until neighbor information is available from the main device thread.
            self.device.got_neighbours.wait()
            # Check for a termination signal from the parent device.
            if self.device.flag_terminate == True:
                break # Exit the loop, terminating the worker thread.
            
            # Wait for a script to be received by the device. This blocks until `script_received` is set.
            self.device.script_received.wait()
            while True:
                # Protect the script index `i` during concurrent access from multiple worker threads.
                with self.device.index_lock:
                    # If all scripts currently assigned to the device have been processed, break this inner loop.
                    if self.device.i >= len(self.device.scripts):
                        break
                    # Retrieve a script and its associated location using modulo to handle script cycling or partitioning.
                    # Note: The modulo operation `self.device.i % 8` implies a specific load-balancing or
                    # script distribution strategy among worker threads.
                    (script, location) = self.device.scripts[self.device.i % 8]
                    self.device.i = self.device.i + 1 # Advance the script index.

                # Process the script if it is not a termination/control signal.
                if script is not None:
                    # Acquire the lock specific to the sensor data location to ensure exclusive access.
                    lock = self.device.locations_locks[location]
                    with lock:
                        
                        script_data = [] # Initialize a list to aggregate data for the script.
                        
                        # Collect data from neighboring devices for the specified location.
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        # Collect data from the device's own sensor for the specified location.
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        # If there is data to process, run the script.
                        if script_data != []:
                            
                            result = script.run(script_data) # Execute the script with collected data.

                            # Update sensor data on neighboring devices with the script's result.
                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            # Update the device's own sensor data with the script's result.
                            self.device.set_data(location, result)
                # If the script is None, it acts as a signal to break the script processing loop.
                if script is None:
                    break
            # Wait at a barrier to clear events, synchronizing with other worker threads.
            self.device.barrier_clear_events.wait()
            # Wait at a time barrier, synchronizing the progression of timepoints across devices.
            self.device.barrier_time.wait()
