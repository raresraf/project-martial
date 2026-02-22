


from threading import Event, Thread, Lock, Semaphore
from ReusableBarrier import ReusableBarrier


"""
@brief This module defines Device, ScriptThread, and DeviceThread classes for simulating a distributed system.
@details It features an external `ReusableBarrier` for global synchronization, dedicated `ScriptThread`s for
parallel script execution, and multiple granular locks (for setting data, getting data, assigning scripts,
and per-location data access) along with a semaphore to manage concurrent script execution. This setup
enables fine-grained control over concurrency and resource access within a distributed sensor network.
"""

from threading import Event, Thread, Lock, Semaphore
from ReusableBarrier import ReusableBarrier


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It handles the reception and execution of scripts, which can modify its own sensor data and
    communicate with neighboring devices. Synchronization across devices is managed by a shared
    `ReusableBarrier`, and access to various shared resources (sensor data, script assignments)
    is protected by multiple granular locks. A `Semaphore` is used to limit the number of
    concurrently executing `ScriptThread` instances.
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local
    data processing and communication with peers, with fine-grained control over concurrency
    and resource access to ensure data integrity and efficient parallel execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id (int): A unique identifier for the device.
        @param sensor_data (dict): A dictionary containing initial sensor data,
                                   where keys are locations and values are data readings.
        @param supervisor (object): A reference to the supervisor object that manages
                                    the overall distributed system and provides access
                                    to network information (e.g., neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are ready for execution by `DeviceThread`.
        self.scripts = []            # List to store assigned scripts and their locations.

        # Dedicated locks for specific operations to ensure thread safety.
        self.lock_setter = Lock()    # Lock for `set_data` method.
        self.lock_getter = Lock()    # Lock for `get_data` method.
        self.lock_assign = Lock()    # Lock for `assign_script` method.

        # Shared synchronization primitives.
        self.barrier = None          # Reference to the shared ReusableBarrier for inter-device synchronization.
        self.location_lock = {}      # Dictionary to hold locks for each sensor data location (shared across devices).

        # Semaphore to limit the number of concurrently running ScriptThread instances.
        self.semaphore = Semaphore(8) # Limits concurrent script execution to 8 threads.

        self.thread = DeviceThread(self) # The main worker thread for this device.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives (barrier and location-specific locks) for all devices.
        @details This method ensures that a single `ReusableBarrier` and a shared dictionary of `location_lock`s
        are created by device 0 and then distributed among all other devices in the simulation.
        It also initiates the `DeviceThread` for each device.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of shared synchronization
                     and mutual exclusion primitives, and starting device threads.
        @pre_condition `devices` is a list of `Device` instances. This method is called for each device during setup.
        @invariant `self.barrier` and `self.location_lock` refer to globally shared instances after setup,
                   ensuring consistent synchronization across all devices. All `DeviceThread`s are started.
        """
        # Block Logic: Initialize shared barrier and location locks only once by the first device (device 0).
        # Invariant: `self.barrier` and `self.location_lock` become initialized and shared.
        if self.device_id == 0: # Only device 0 is responsible for initializing and distributing.
            self.barrier = ReusableBarrier(len(devices)) # Create a new reusable barrier.

            # Block Logic: Populate `location_lock` with a lock for each unique sensor data location across all devices.
            # Invariant: `self.location_lock` contains a Lock object for every unique sensor location.
            for device in devices[:]: # Iterate through a copy of the devices list.
                for loc in device.sensor_data.keys():
                    if loc not in self.location_lock:
                        self.location_lock[loc] = Lock() # Create a new lock for each unique location.

            # Block Logic: Distribute the shared barrier and location locks to all devices.
            # Invariant: All devices have references to the shared `self.barrier` and `self.location_lock`.
            for device in devices[:]: # Iterate through a copy of the devices list.
                device.barrier = self.barrier
                device.location_lock = self.location_lock
                
                device.thread.start() # Start the DeviceThread for each device.


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific location.
        @details This method uses `lock_assign` to ensure thread-safe appending of scripts
        to the device's queue. If a script is provided, it's appended; otherwise (script is None),
        it signals the completion of script assignments for the current timepoint by setting
        the `script_received` event.
        @param script (object): The script object to be executed, or None to signal end of assignments.
        @param location (str): The location associated with the script or data.
        @block_logic Manages the receipt of scripts and signals readiness for script execution.
        @pre_condition `self.scripts` is a list, `self.script_received` is an Event object,
                       `self.lock_assign` is an initialized Lock.
        @invariant Either a script is added to `self.scripts` (under lock), or `self.script_received` is set.
        """
        with self.lock_assign: # Acquire lock to safely modify the scripts list.
            if script is not None:
                self.scripts.append((script, location)) # Add the script and its location to the queue.
            else:
                self.script_received.set() # Signal that all scripts for the current timepoint have been assigned.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        @details This method uses `lock_getter` to ensure thread-safe reading of sensor data.
        @param location (str): The location for which to retrieve data.
        @return object: The sensor data at the specified location, or None if the location is not found.
        @block_logic Provides thread-safe access to retrieve sensor data.
        @pre_condition `self.sensor_data` is a dictionary, `self.lock_getter` is an initialized Lock.
        @invariant Returns data associated with `location` if present, otherwise None.
        """
        with self.lock_getter: # Acquire lock to safely read sensor data.
            if location in self.sensor_data:
                return self.sensor_data[location]
            else:
                return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.
        @details This method uses `lock_setter` to ensure thread-safe modification of sensor data.
        @param location (str): The location whose data is to be updated.
        @param data (object): The new data value for the specified location.
        @block_logic Provides thread-safe access to update sensor data.
        @pre_condition `self.sensor_data` is a dictionary, `self.lock_setter` is an initialized Lock.
        @invariant If `location` is a key in `self.sensor_data`, its value is updated under lock protection.
        """
        with self.lock_setter: # Acquire lock to safely modify sensor data.
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its associated thread.
        @details This ensures that the device's worker thread completes its execution before the program exits.
        """
        self.thread.join()


class ScriptThread(Thread):


    """


    @brief A worker thread dedicated to executing a single assigned script for a Device instance.


    @details This thread processes a specific script for a given location, collects data from the


    parent device and its neighbors, executes the script's logic, and updates sensor data


    in a thread-safe manner using a location-specific lock and a global semaphore.


    @architectural_intent Enhances parallelism by allowing multiple scripts to run concurrently,


    with controlled resource access to prevent race conditions during data manipulation and


    to limit overall concurrency.


    """


    


    def __init__(self, device_thread, script, location, neighbours):


        """


        @brief Initializes a new ScriptThread instance.


        @param device_thread (DeviceThread): The parent DeviceThread that spawned this script thread.


        @param script (object): The script object to be executed.


        @param location (str): The location associated with the script for which data is processed.


        @param neighbours (list): A list of neighboring Device objects from which to collect sensor data.


        """


        Thread.__init__(self) # Initialize the base Thread class.


        self.script = script # The script to execute.


        self.device_thread = device_thread # Reference to the parent DeviceThread.


        self.location = location # The sensor data location this script pertains to.


        self.neighbours = neighbours # List of neighboring devices.





    def run(self):


        """


        @brief The main execution logic for the ScriptThread.


        @details This method acquires a location-specific lock and then a global semaphore


        to control access to shared data and limit concurrency. It collects data from


        the parent device and its neighbors for the specified location, executes the assigned


        script, and then updates the relevant sensor data for the device and its neighbors.


        Finally, it releases the semaphore and the location-specific lock.


        @block_logic Processes a single script for a specific location, ensuring thread-safe


        data access and controlled concurrency.


        @pre_condition `self.script` is an object with a `run` method, `self.device_thread.device.location_lock`


                       contains a Lock for `self.location`, and `self.device_thread.device.semaphore` is available.


        @invariant The script is executed, and data is updated under the protection of both a


                   location lock and a global semaphore.


        """


        # Block Logic: Acquire a lock for the specific location to ensure exclusive access to its data.


        # Invariant: Only one ScriptThread can modify or read data for `self.location` at a time.


        self.device_thread.device.location_lock[self.location].acquire()





        # Block Logic: Acquire a global semaphore to limit the number of concurrently executing ScriptThread instances.


        # Invariant: The total number of active ScriptThread instances is limited by the semaphore's initial value.


        self.device_thread.device.semaphore.acquire()





        script_data = [] # List to accumulate data for the current script's execution.


        


        # Block Logic: Collect data from neighboring devices for the current location.


        # Invariant: `script_data` will contain data from all available neighbors for the given location.


        for device in self.neighbours:


            data = device.get_data(self.location)


            if data is not None:


                script_data.append(data)


        


        # Block Logic: Collect data from the current device itself for the current location.


        # Invariant: If available, the device's own data for the location is added to `script_data`.


        data = self.device_thread.device.get_data(self.location)


        if data is not None:


            script_data.append(data)





        # Block Logic: Execute the script if there is any data to process.


        # Pre-condition: `self.script` is an object with a `run` method, and `script_data` is a list of data.


        # Invariant: `result` holds the output of the script's execution.


        if script_data != []:


            result = self.script.run(script_data) # Execute the script with the collected data.





            # Block Logic: Propagate the script's result to neighboring devices.


            # Invariant: All neighbors receive the updated data for the given location.


            for device in self.neighbours:


                device.set_data(self.location, result)


            


            # Functional Utility: Update the current device's own data with the script's result.


            self.device_thread.device.set_data(self.location, result)





        self.device_thread.device.semaphore.release() # Release the global semaphore.


        self.device_thread.device.location_lock[self.location].release() # Release the lock for the current location.




class DeviceThread(Thread):
    """
    @brief The main worker thread for a Device instance.
    @details This thread orchestrates the device's operational cycle, including
    fetching neighbor information, waiting for script assignments, and then dispatching
    these scripts to individual `ScriptThread` instances for concurrent execution.
    It manages synchronization through a shared barrier and ensures that script processing
    is completed before proceeding to the next timepoint.
    @architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
    abstracting multi-threaded script execution and coordinating with the global barrier
    to ensure proper progression of the distributed system.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.
        @param device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initialize the base Thread with a descriptive name.
        self.device = device # Reference to the parent Device object.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        @details This method continuously monitors the simulation state. For each timepoint,
        it retrieves neighbor information. If neighbors are available, it waits for script
        assignments, creates `ScriptThread`s for each assigned script, starts and joins
        these script threads, clears the `script_received` event, and finally synchronizes
        with other `DeviceThread` instances via the shared `ReusableBarrier`.
        The loop terminates when the supervisor signals the end of the simulation
        (by returning None for neighbors).
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        script parallelization, and inter-device synchronization.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `script_received` event, and `barrier`.
        @invariant The thread progresses through timepoints, processes scripts concurrently,
                   and ensures global synchronization.
        """
        while True:
            # Functional Utility: Get information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            # Block Logic: Check if the simulation should terminate.
            # Pre-condition: `neighbours` list indicates the current state of the network.
            # Invariant: The loop terminates if no neighbors are returned by the supervisor.
            if neighbours is None:
                break

            # Block Logic: Wait until script assignments for the current timepoint are complete.
            # Pre-condition: `self.device.script_received` is an Event object.
            # Invariant: The thread proceeds only after the supervisor signals completion of script assignment.
            self.device.script_received.wait()
            script_threads = [] # List to hold ScriptThread instances for parallel script execution.

            # Block Logic: Create and start ScriptThread instances for each assigned script.
            # Pre-condition: `self.device.scripts` contains tuples of (script, location).
            # Invariant: A `ScriptThread` is created and started for each script, allowing concurrent execution.
            for (script, location) in self.device.scripts:
                # Functional Utility: Create a new ScriptThread for each script.
                thread = ScriptThread(self, script, location, neighbours)
                script_threads.append(thread) # Add the thread to the list for joining later.
                thread.start() # Start the ScriptThread to execute the script concurrently.

            # Block Logic: Wait for all spawned ScriptThread instances to complete their execution.
            # Invariant: The DeviceThread will not proceed until all its ScriptThread children have finished.
            for thread in script_threads:
                thread.join()

            # Functional Utility: Clear the `script_received` event for the next timepoint.
            self.device.script_received.clear()
            
            # Functional Utility: Clear the scripts list for the next timepoint.
            self.device.scripts = [] # Reset scripts list for the next timepoint.

            # Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
            # Invariant: All DeviceThread instances will reach this barrier before any proceeds to the next timepoint.
            self.device.barrier.wait()
