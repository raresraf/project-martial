


"""
@file device.py
@brief Implements a simulated distributed device with concurrent script execution and synchronization.

This module defines the `Device` class, representing a node in a simulated sensor network or distributed system.
Each device manages its sensor data, interacts with a supervisor, and executes assigned scripts
in parallel using a pool of `ScriptRunner` worker threads. The module utilizes various threading
primitives, including `Event`, `Thread`, `Lock`, `Condition`, and a custom `ReusableBarrierCond`,
to facilitate complex synchronization and coordination between devices and within a device's
script processing pipeline.

Domain: Distributed Systems, Concurrency, Multithreading, Simulation, Sensor Networks, Synchronization.
"""

from threading import Event, Thread, Lock, Condition

class ReusableBarrierCond(object):
    """
    @brief Implements a reusable barrier synchronization primitive using a `threading.Condition`.

    This barrier allows a specified number of threads to wait until all have arrived
    at the barrier point. Once all threads have arrived, they are released, and the
    barrier automatically resets, making it available for subsequent synchronization.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes a ReusableBarrierCond instance.

        Parameters:
          - num_threads: The total number of threads that must reach the barrier
                         before all are released.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Current count of threads waiting at the barrier.
        self.cond = Condition()               # Condition variable for synchronization.

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all threads arrive.

        When a thread calls `wait()`, it decrements the internal counter.
        If it's the last thread to arrive, it notifies all other waiting threads
        and resets the barrier for future use. Otherwise, it waits on the condition
        variable until notified by the last arriving thread.
        """
        self.cond.acquire() # Acquire the lock associated with the condition variable.
        try:
            self.count_threads -= 1
            # Block Logic: Checks if this is the last thread to arrive at the barrier.
            # Invariant: If `count_threads` becomes 0, all threads have arrived.
            if self.count_threads == 0:
                self.cond.notify_all() # Release all waiting threads.
                self.count_threads = self.num_threads # Reset the barrier for reuse.
            else:
                self.cond.wait() # Wait until all other threads arrive.
        finally:
            self.cond.release() # Release the lock.

class Device(object):
    """
    @brief Represents a single device (node) in a simulated distributed system.

    Each Device instance manages its own sensor data, interacts with a supervisor,
    and can execute scripts in a parallel fashion. It also handles synchronization
    with other devices for coordinated actions using various threading primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.

        Parameters:
          - device_id: A unique identifier for this device.
          - sensor_data: A dictionary containing initial sensor data for various locations.
          - supervisor: A reference to the central supervisor responsible for coordinating devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()      # Event to signal when a script has been assigned for processing.
        self.scripts = []                   # List to store assigned scripts (script objects).
        self.timepoint_done = Event()       # Event to signal completion of a timepoint's processing.

        self.new_script = Event()           # Event for internal signaling of a new script to worker threads.
        self.new_script_received = None     # Stores the (script, location) tuple for new assignments.
        self.barrier = None                 # Reusable barrier for synchronizing device timepoints.
        self.script_lock = Lock()           # Lock to protect access to script assignment variables.
        self.lock_dict = {}                 # Dictionary of locks, one for each sensor data location, for granular access control.

        # Initialize and start the dedicated thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives across all devices.

        This method is intended to be called by a coordinating entity (e.g., the supervisor
        or device_id 0 itself) to initialize a global reusable barrier and
        per-location locks that will be shared among all devices.

        Parameters:
          - devices: A list of all Device instances in the system.
        """
        
        # Block Logic: Only device with ID 0 is responsible for initializing shared barriers and locks.
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices)) # Initialize a reusable barrier for all devices.
            # Assign the newly created barrier to all devices in the system.
            for device in devices:
                device.barrier = self.barrier
                # Block Logic: Populate the shared lock_dict with a Lock for each unique sensor data location.
                # Invariant: Each unique sensor data location will have an associated Lock for thread-safe access.
                for loc in device.sensor_data:
                    if loc not in self.lock_dict:
                        self.lock_dict[loc] = Lock()
            
            # Assign the newly created shared lock_dict to all devices.
            for device in devices:
                device.lock_dict = self.lock_dict

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific sensor data location.

        This method is used by the supervisor to assign scripts. It stores the script
        and location, and signals internal worker threads that a new script is available.

        Parameters:
          - script: The script object to execute.
          - location: The sensor data location relevant to the script.
        """
        # Acquire a lock to protect concurrent script assignments.
        self.script_lock.acquire()
        # Store the new script and its associated location.
        self.new_script_received = (script, location)
        # Signal to the DeviceThread that a new script has been received.
        self.new_script.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.

        Parameters:
          - location: The identifier for the sensor data location.

        Returns:
          - The sensor data at the given location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets (updates) sensor data for a specified location.

        Parameters:
          - location: The identifier for the sensor data location.
          - data: The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device thread.

        Joins the device's dedicated thread, ensuring all its operations are
        completed before the main program exits.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main thread for a `Device` instance, managing script execution and synchronization.

    This thread orchestrates the processing of assigned scripts, coordinating with
    a pool of `ScriptRunner` worker threads and synchronizing its actions with
    the supervisor and other devices in the system. It handles two distinct phases
    of script execution per timepoint: initial script processing from the `device.scripts` list,
    and subsequent dynamic script assignments via `device.new_script_received`.
    """
    

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.

        Parameters:
          - device: A reference to the parent `Device` instance this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script_runners = []    # List to hold instances of ScriptRunner worker threads.
        self.neighbours = []        # Stores a list of neighboring devices.
        self.script = None          # Currently assigned script object for worker threads.
        self.location = None        # Currently assigned location for worker threads.

        self.new_script = Event()           # Event to signal `ScriptRunner` that a new script is ready.
        self.script_lock = Lock()           # Lock for protecting access to `script` and `location`.
        self.wait_for_data = Event()        # Event for `DeviceThread` to wait for `ScriptRunner` to process a script.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        This method continuously:
        1. Fetches information about neighboring devices from the supervisor.
        2. Initializes a pool of `ScriptRunner` worker threads.
        3. Processes an initial batch of scripts stored in `device.scripts`.
        4. Enters a loop to handle dynamically assigned scripts (via `device.new_script`).
        5. Coordinates with `ScriptRunner` threads and other `DeviceThread` instances
           using events and a shared barrier to ensure synchronized execution.
        6. Shuts down its worker pool when a termination signal (`neighbours is None` or `script is None`) is received.
        """
        while True:
            
            # Block Logic: Fetches the list of neighboring devices from the supervisor.
            # Invariant: `neighbours` will contain devices connected to the current device.
            self.neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If `neighbours` is None, it signals the shutdown of the simulation.
            if self.neighbours is None:
                break # Exit the main loop, terminating the DeviceThread.

            # Block Logic: Initializes and starts a pool of `ScriptRunner` worker threads.
            # These threads are responsible for executing individual script tasks.
            self.script_runners = []
            # Cache local references to events and locks to pass to worker threads.
            new_scr = self.new_script
            scr_lock = self.script_lock
            wait_data = self.wait_for_data
            for _ in range(8): # Creates 8 `ScriptRunner` worker threads.
                script_runner = ScriptRunner(self, new_scr, scr_lock, wait_data)
                self.script_runners.append(script_runner)
                script_runner.start()

            # Block Logic: First phase of script processing – initial scripts assigned to the device.
            # Invariant: All scripts in `self.device.scripts` are processed by worker threads.
            for (script, location) in self.device.scripts:
                # Assign the script and location to be processed by a worker.
                self.script = script
                self.location = location
                self.new_script.set()       # Signal to a `ScriptRunner` that a new task is available.
                self.wait_for_data.wait()   # Wait for the `ScriptRunner` to finish processing the script.
                self.wait_for_data.clear()  # Reset the event for the next script.

            # Block Logic: Second phase of script processing – dynamically assigned scripts.
            # This loop continuously monitors for new scripts assigned by the supervisor
            # until a termination signal (`script is None`) is received.
            while True:
                # Wait for `device.new_script` event, which is set by `Device.assign_script`.
                self.device.new_script.wait()
                self.device.new_script.clear()
                # Release the `device.script_lock` held by `Device.assign_script` method.
                self.device.script_lock.release()

                # Transfer the newly received script and location to `DeviceThread`'s attributes.
                self.script = self.device.new_script_received[0]
                self.location = self.device.new_script_received[1]

                # Termination condition: If `script` is None, it's a signal to shut down.
                if self.script is None:
                    break

                self.new_script.set()       # Signal to `ScriptRunner` that a new script is available.
                self.wait_for_data.wait()   # Wait for the `ScriptRunner` to finish processing.
                self.wait_for_data.clear()  # Reset the event.
                # Append the processed script to the device's history or processed list.
                self.device.scripts.append((self.script, self.location)) # This seems to add it back to the list.

            # Block Logic: Signal all `ScriptRunner` worker threads to terminate.
            # Invariant: Each worker receives a `None` task (implicitly due to `self.script` being None).
            self.script = None # Set script to None as a termination signal for workers.
            self.location = None # Clear location.
            self.neighbours = None # Clear neighbours.
            for script_runner in self.script_runners:
                self.new_script.set()       # Signal to each `ScriptRunner` to check for termination.
                self.wait_for_data.wait()   # Wait for worker to acknowledge termination.
                self.wait_for_data.clear()  # Reset event.

            # Wait for all `ScriptRunner` worker threads to finish their execution.
            for script_runner in self.script_runners:
                script_runner.join()

            # Wait at the global barrier, synchronizing with all other devices before starting the next timepoint.
            self.device.barrier.wait()

class ScriptRunner(Thread):
    

class ScriptRunner(Thread):
    """
    @brief A worker thread responsible for executing individual scripts within a `DeviceThread`.

    Instances of this class are part of a pool of threads managed by a `DeviceThread`.
    They wait for new script tasks, acquire necessary locks for data access,
    execute the script, and update sensor data across the current device and its neighbors.
    """
    

    def __init__(self, device_thread, new_script, script_lock, wait_for_data):
        """
        @brief Initializes a ScriptRunner worker thread.

        Parameters:
          - device_thread: A reference to the parent `DeviceThread` instance.
          - new_script: An `Event` used by `DeviceThread` to signal a new script is ready.
          - script_lock: A `Lock` used to protect access to script and location variables.
          - wait_for_data: An `Event` used to signal `DeviceThread` when the script has been processed.
        """
        Thread.__init__(self)
        self.device_thread = device_thread
        self.new_script = new_script
        self.script_lock = script_lock
        self.wait_for_data = wait_for_data

    def run(self):
        """
        @brief The main execution loop for the ScriptRunner worker.

        This method continuously:
        1. Waits for a signal from its parent `DeviceThread` that a new script is available.
        2. Acquires the `script_lock` to access the script and location information.
        3. Collects sensor data from the current device and its neighbors for the specified location.
        4. Executes the assigned script with the collected data.
        5. Updates the sensor data on the current device and its neighbors with the script's result.
        6. Releases the locks and signals back to `DeviceThread` that processing is complete.
        7. Terminates when the `DeviceThread` signals a shutdown by providing `None` for the script.
        """
        while True:
            self.script_lock.acquire() # Acquire lock to access shared script/location variables.
            self.new_script.wait()     # Wait for `DeviceThread` to signal a new script.
            self.new_script.clear()    # Clear the event for the next signal.
            
            # Retrieve the script, location, and neighbor information from `DeviceThread`.
            script = self.device_thread.script
            location = self.device_thread.location
            neighbours = self.device_thread.neighbours
            
            self.wait_for_data.set()   # Signal `DeviceThread` that script parameters have been retrieved.
            self.script_lock.release() # Release the lock.

            # Pre-condition: If any of these are None, it indicates a shutdown signal from DeviceThread.
            if neighbours is None or location is None or script is None:
                break # Terminate the worker thread.

            # Pre-condition: If no neighbors are specified, skip script execution for this cycle.
            if neighbours == []:
                continue

            
            # Acquire the location-specific lock to ensure thread-safe access to sensor data at this location.
            self.device_thread.device.lock_dict[location].acquire()

            script_data = [] # Stores data collected for the script's execution.
            
            # Block Logic: Collects sensor data from neighboring devices for the given location.
            # Invariant: `script_data` will contain relevant sensor readings from neighbors.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Collect sensor data from the current device itself for the given location.
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Pre-condition: Only execute the script if there is data to process.
            if script_data != []:
                
                # Execute the script with the collected data.
                result = script.run(script_data)

                # Block Logic: Update sensor data on neighboring devices with the script's result.
                # Invariant: Neighbors' sensor data at `location` is updated.
                for device in neighbours:
                    device.set_data(location, result)
                
                # Update sensor data on the current device with the script's result.
                self.device_thread.device.set_data(location, result)

            # Post-condition: Release the lock for the sensor data location.
            self.device_thread.device.lock_dict[location].release()
