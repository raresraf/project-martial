"""
@file device.py
@brief Implements components for a distributed system, likely a simulation or sensor network,
focusing on concurrent data processing, synchronization, and task management.
This module defines Device objects that manage sensor data and execute scripts
using multiple worker threads within each device, employing various synchronization
primitives like events, locks, and reusable barriers.
"""

# The 'barrier' import refers to barrier.py, which defines ReusableBarrierCond.
# This indicates that the class ReusableBarrierCond is intended to be used from an external file,
# but it's defined here in this self-contained snippet.
import barrier
from threading import Event, Thread, Lock, Condition # Condition is used by ReusableBarrierCond


class ReusableBarrierCond(object):
    """
    @brief Implements a reusable barrier for thread synchronization using a threading.Condition object.
    This barrier allows a fixed number of threads to wait for each other before
    proceeding, and can be reused across multiple synchronization points.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.

        @param num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads  # Counter for threads yet to reach the barrier.
        self.cond = Condition()  # Condition variable for signaling and waiting.


    def wait(self):
        """
        @brief Blocks the calling thread until all 'num_threads' have reached the barrier.
        When the last thread arrives, all waiting threads are notified and the barrier resets.
        """
        self.cond.acquire()  # Acquires the lock associated with the condition variable.
        self.count_threads -= 1  # Decrements the count of threads yet to reach.
        # Conditional Logic: If this is the last thread to reach the barrier.
        if self.count_threads == 0:
            self.cond.notify_all()  # Notifies all waiting threads to proceed.
            self.count_threads = self.num_threads  # Resets the counter for future use.
        else:
            self.cond.wait()  # Waits (releases lock and blocks) until notified.
        self.cond.release()  # Releases the lock.


class Device(object):
    """
    @brief Represents a single device in the distributed system.
    Each device has a unique ID, manages its sensor data, and interacts with a supervisor.
    It processes assigned scripts using multiple internal threads (DeviceThread and WorkerThread),
    employing various synchronization primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id (int): A unique identifier for the device.
        @param sensor_data (dict): A dictionary holding sensor readings for different locations.
        @param supervisor (Supervisor): A reference to the central supervisor managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Events for script and timepoint management.
        self.script_received = Event()  # Signals when scripts have been assigned.
        self.script_taken = Event()     # Signals when a script has been taken by a worker (unused).
        self.assign_script_none = Event() # Signals that `assign_script` was called with None (unused).
        self.script_de_orice_fel = Event() # General script event (unused).
        self.assign_script_not_none = Event() # Signals that `assign_script` was called with a script (unused).
        
        # Barrier instances for synchronization across multiple levels.
        self.bariera = None         # Global barrier for all devices.
        self.bariera_join = None    # Barrier for joining worker threads across devices.
        self.barrier_time = None    # Barrier for timepoint synchronization.
        self.barrier_clear_events = None # Barrier for clearing events.
        
        self.flag_terminate = False # Flag to signal threads to terminate.
        
        # Locks for critical sections.
        self.script_sent = Lock()         # Protects access to script assignment.
        self.script_sent_thread = Lock()  # Another script-related lock (unused).
        self.barrier_lock = Lock()        # Protects internal barrier state (unused).
        self.get_neighbours_lock = Lock() # Protects access to fetching neighbors (unused).
        self.index_lock = Lock()          # Protects the shared script index `i`.
        self.put_take_data = Lock()       # Protects `sensor_data` access.
        
        self.counter = 0            # General purpose counter (unused).
        self.flag_received = Event() # Signals something received (unused).
        self.got_neighbours = Event() # Signals that neighbors have been fetched.
        
        self.flag_script_received = False  # Boolean flag (redundant with Event).
        self.flag_script_taken = False     # Boolean flag (redundant with Event).
        self.flag_assign_script = 2        # State for script assignment (unused).
        self.flag_get_neigbours = False    # Boolean flag (redundant with Event).
        
        self.i = 0  # Shared index for scripts list, accessed by worker threads.
        self.scripts = [] # List of (script, location) tuples.
        self.neighbours = None # List of neighboring devices.
        self.devices = [] # List of all devices in the system.
        self.count_threads = [] # Counter for threads (used by some barriers).
        self.locations_locks = [] # List of locks, one per data location.
        self.timepoint_done = Event() # Signals that script assignment for a timepoint is done.
        self.initialize = Event() # Signals that device initialization is complete.
        
        self.thread = DeviceThread(self) # The main DeviceThread instance.
        self.threads = [] # List to hold additional WorkerThread instances.

    def __str__(self):
        """
        @brief Provides a string representation of the Device.

        @return str: A formatted string indicating the device ID.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs global setup for devices, including initializing shared barriers and locks.
        Device 0 coordinates this setup and propagates instances to other devices.

        @param devices (list): A list of all Device instances in the system.
        """
        self.devices = devices # Stores a reference to all devices in the system.
        self.count_threads = [len(self.devices)] # Initializes a thread counter for barriers.

        # Conditional Logic: Only Device 0 performs the global setup.
        if self.device_id == 0:
            # Block Logic: Identifies all unique data locations across all devices.
            locations = []
            for device in self.devices:
                l = []
                for key in device.sensor_data.keys():
                    l.append(key)
                for locatie in l:
                    locations.append(locatie)
            
            # Determines the maximum location ID to size the `locations_locks` array.
            maxim = max(locations) if locations else -1 # Handles case where no locations exist.
            self.locations_locks = [None] * (maxim + 1)
            
            # Block Logic: Creates a Lock for each unique data location.
            for locatie in locations:
                if self.locations_locks[locatie] is None:
                    lock = Lock()
                    self.locations_locks[locatie] = lock

            # Initializes various reusable barriers.
            self.bariera = ReusableBarrierCond(len(self.devices)) # Main global device barrier.
            num_threads = len(self.devices) * 8 # Total number of worker threads across all devices.
            self.bariera_join = ReusableBarrierCond(num_threads) # Barrier for joining workers (unused).
            self.barrier_time = ReusableBarrierCond(num_threads) # Barrier for timepoint synchronization.
            self.barrier_clear_events = ReusableBarrierCond(num_threads) # Barrier for event clearing.

            # Block Logic: Propagates the initialized barriers and locks to all other devices.
            for device in self.devices:
                device.i = 0  # Resets shared script index for each device.
                device.bariera = self.bariera
                device.counter = len(self.devices) # (Unused, seems to be intended for a barrier count)
                device.barrier_time = self.barrier_time
                device.barrier_clear_events = self.barrier_clear_events
                device.locations_locks = self.locations_locks

        self.thread.start() # Starts the main DeviceThread.
        # Block Logic: Creates and starts 7 additional WorkerThread instances for this Device.
        i = 0
        while i < 7:
            dev = WorkerThread(self)
            dev.start()
            self.threads.append(dev)
            i = i + 1
        
        self.initialize.set() # Signals that device setup is complete.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location or signals timepoint completion.

        @param script (callable): The script (function or object with a run method) to execute.
                                  If None, it signals that script assignment for the timepoint is done.
        @param location (int): The identifier of the data location the script operates on.
        """
        with self.script_sent: # Acquires a lock to protect script assignment.
            # Conditional Logic: Appends the script (or None signal) to the scripts list.
            if script is not None:
                self.scripts.append((script, location))
            else:
                self.scripts.append((script, location)) # Appends None, None as a termination signal.
                self.script_received.set()  # Signals that scripts have been received.
                self.timepoint_done.set()   # Signals timepoint completion.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location (int): The identifier of the data location.
        @return any: The sensor data at the specified location, or None if not found.
        """
        with self.put_take_data: # Acquires a lock to protect sensor data access.
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location (int): The identifier of the data location to update.
        @param data (any): The new data value.
        """
        with self.put_take_data: # Acquires a lock to protect sensor data access.
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down all threads associated with this device.
        """
        # Block Logic: Joins all worker and main device threads to ensure their completion.
        for thread in self.threads:
            thread.join()
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief The main thread of execution for a Device.
    It is responsible for fetching neighbor information and coordinating the processing
    of scripts by cooperatively sharing script access with other worker threads.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device (Device): The Device object this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop of the DeviceThread.
        It continuously fetches neighbor information, waits for scripts,
        and then processes them cooperatively with other worker threads.
        It also manages various synchronization barriers.
        """
        while True:
            # Block Logic: Fetches neighbor information.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            # Conditional Logic: If no neighbors are returned (supervisor signals shutdown), terminates.
            if self.device.neighbours is None:
                self.device.flag_terminate = True # Sets termination flag.
                self.device.got_neighbours.set() # Signals termination to other workers.
                break # Terminates the thread.
            
            self.device.got_neighbours.set() # Signals that neighbors have been fetched.
            
            self.device.script_received.wait() # Waits until scripts have been assigned.
            
            # Block Logic: Cooperatively processes scripts from the device's shared list.
            while True:
                with self.device.index_lock: # Acquires lock to safely get and increment script index.
                    if self.device.i >= len(self.device.scripts): # Conditional: All scripts processed.
                        break
                    # Retrieves script and location. Modulo 8 is suspicious and might be a bug,
                    # as it re-processes scripts if len(scripts) > 8 in a specific way.
                    (script, location) = self.device.scripts[self.device.i % 8]
                    self.device.i = self.device.i + 1 # Increments shared script index.
                
                # Conditional Logic: If a valid script is retrieved.
                if script is not None:
                    lock = self.device.locations_locks[location] # Gets the lock for the specific location.
                    with lock: # Acquires the location-specific lock.
                        
                        script_data = [] # List to collect data for the script.
                        
                        # Block Logic: Collects data from neighboring devices.
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        # Collects data from its own device.
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        # Conditional Logic: If any data was collected, executes the script and propagates results.
                        if script_data != []:
                            result = script.run(script_data) # Executes the script.

                            # Block Logic: Propagates the new data to all neighboring devices.
                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            self.device.set_data(location, result) # Updates data on its own device.
                # Conditional Logic: If a None script is encountered, it's a termination signal.
                if script is None:
                    break
            
            self.device.barrier_clear_events.wait() # Waits at a barrier for events to be cleared.
            # Clears events for the next timepoint.
            self.device.script_received.clear()
            self.device.got_neighbours.clear()
            self.device.barrier_time.wait() # Waits at a barrier for timepoint synchronization.

class WorkerThread(Thread):
    """
    @brief An auxiliary worker thread for a Device.
    These threads cooperate with the main `DeviceThread` to process scripts.
    """

    def __init__(self, device):
        """
        @brief Initializes a new WorkerThread instance.

        @param device (Device): The Device object this worker thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop of a WorkerThread.
        It continuously waits for neighbor information, then processes scripts
        cooperatively with other threads, and synchronizes using various barriers.
        """
        while True:
            self.device.got_neighbours.wait() # Waits until neighbors have been fetched.
            # Conditional Logic: If termination flag is set, breaks the loop.
            if self.device.flag_terminate == True:
                break
            
            self.device.script_received.wait() # Waits until scripts have been assigned.
            
            # Block Logic: Cooperatively processes scripts from the device's shared list.
            while True:
                with self.device.index_lock: # Acquires lock to safely get and increment script index.
                    if self.device.i >= len(self.device.scripts): # Conditional: All scripts processed.
                        break
                    # Retrieves script and location. Modulo 8 is suspicious and might be a bug.
                    (script, location) = self.device.scripts[self.device.i % 8]
                    self.device.i = self.device.i + 1 # Increments shared script index.

                # Conditional Logic: If a valid script is retrieved.
                if script is not None:
                    lock = self.device.locations_locks[location] # Gets the lock for the specific location.
                    with lock: # Acquires the location-specific lock.
                        
                        script_data = [] # List to collect data for the script.
                        
                        # Block Logic: Collects data from neighboring devices.
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        # Collects data from its own device.
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        # Conditional Logic: If any data was collected, executes the script and propagates results.
                        if script_data != []:
                            result = script.run(script_data) # Executes the script.

                            # Block Logic: Propagates the new data to all neighboring devices.
                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            self.device.set_data(location, result) # Updates data on its own device.
                # Conditional Logic: If a None script is encountered, it's a termination signal.
                if script is None:
                    break
            
            self.device.barrier_clear_events.wait() # Waits at a barrier for events to be cleared.
            self.device.barrier_time.wait() # Waits at a barrier for timepoint synchronization.
