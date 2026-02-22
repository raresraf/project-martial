



"""
@brief This module defines `Device`, `DeviceThread`, `MiniT` (a script worker thread), and `ReusableBarrier` classes
for simulating a distributed system.
@details It features a condition-variable-based reusable barrier for inter-device synchronization, where each `Device`
executes scripts concurrently by spawning temporary `MiniT` threads within its dedicated `DeviceThread`. Access
to sensor data is managed with a device-specific lock (`lacat_date`), ensuring thread-safe operations during
concurrent script execution.
"""

from threading import Event, Thread, Lock, Condition


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It can receive and queue scripts for execution, which are then processed concurrently by its
    dedicated `DeviceThread` by spawning `MiniT` instances. Synchronization across devices is managed
    by a shared `ReusableBarrier`, and thread-safe access to its `sensor_data` is ensured by a
    device-specific `Lock` (`lacat_date`).
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
    processing and communication with peers, implementing explicit synchronization and mutual exclusion
    for coordinated and consistent execution across timepoints.
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
        self.script_received = Event() # Event to signal when new scripts are ready for execution.
        self.scripts = []            # List to store assigned scripts and their locations.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's script assignment.
        self.thread = DeviceThread(self) # The main worker thread for this device.
        self.thread.start()          # Start the device's execution thread.
        self.bariera = ReusableBarrier(0) # Placeholder barrier, will be updated during setup.
        self.lacat_date = Lock()     # Device-specific lock for protecting its sensor data.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared `ReusableBarrier` for all devices.
        @details This method initializes a single `ReusableBarrier` instance by device 0
        and then distributes it to all devices in the simulation. It stores a reference
        to all devices and assigns the shared barrier to each of them.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of the shared barrier.
        @pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
        @invariant `self.bariera` refers to a globally shared `ReusableBarrier` instance after setup.
        """
        if self.device_id == 0: # Only device 0 is responsible for initializing and distributing.
            barria = ReusableBarrier(len(devices)) # Create a new reusable barrier, sized for all devices.
            # Block Logic: Distribute the newly created barrier to all devices.
            # Invariant: All devices in `devices` receive a reference to the shared `barria`.
            for device in devices:
                device.bariera = barria

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific location.
        @details If a script is provided, it's appended to the device's script queue, and the
        `script_received` event is set. If no script is provided (i.e., `script` is None),
        it signifies that the current timepoint's script assignment is complete, and the
        `timepoint_done` event is set to unblock the `DeviceThread`.
        @param script (object): The script object to be executed, or None to signal end of assignments.
        @param location (str): The location associated with the script or data.
        @block_logic Handles the assignment of new scripts or signals the completion of script assignment for a timepoint.
        @pre_condition `self.scripts` is a list, `self.script_received` and `self.timepoint_done` are Event objects.
        @invariant Either a script is added and `script_received` is set, or `timepoint_done` is set.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the queue.
            self.script_received.set() # Signal that a new script has been received.
        else:
            self.timepoint_done.set() # Signal that script assignments for the current timepoint are complete.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        @param location (str): The location for which to retrieve data.
        @return object: The sensor data at the specified location, or None if the location is not found.
        @pre_condition `location` is a key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        # Functional Utility: Return None if location not in sensor_data.
        return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.
        @details This method updates the internal sensor data if the location exists.
        It's assumed that external synchronization (e.g., through `MiniT`'s `lacat_date` lock)
        protects this operation during concurrent modifications.
        @param location (str): The location whose data is to be updated.
        @param data (object): The new data value for the specified location.
        @block_logic Updates the internal sensor data.
        @pre_condition `self.sensor_data` is a dictionary.
        @invariant If `location` is a key in `self.sensor_data`, its value is updated.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its associated thread.
        @details This ensures that the device's worker thread completes its execution before the program exits.
        """
        self.thread.join()



class DeviceThread(Thread):
    """
    @brief The main worker thread for a Device instance.
    @details This thread orchestrates the device's operational cycle for each timepoint.
    It fetches neighbor information, waits for script assignments, and then processes
    these scripts concurrently by spawning temporary `MiniT` worker threads. It ensures
    inter-device synchronization through a shared `ReusableBarrier`.
    @architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
    abstracting multi-threaded script execution through temporary worker threads and
    coordinating with the global barrier to ensure proper progression of the distributed system.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.
        @param device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initialize the base Thread with a descriptive name.
        self.device = device # Reference to the parent Device object.
        self.thread_list = list() # List to hold MiniT worker instances for concurrent execution.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        @details This method continuously monitors the simulation state. For each timepoint,
        it retrieves neighbor information from the supervisor. If neighbors are available,
        it waits until `timepoint_done` is set (signaling that script assignments are complete),
        then processes the assigned scripts by creating and starting `MiniT` instances.
        After all `MiniT` instances complete, it clears `timepoint_done` and synchronizes
        with other `DeviceThread` instances via the global `ReusableBarrier`. The loop
        terminates when the supervisor signals the end of the simulation.
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        concurrent script execution via temporary threads, and inter-device synchronization.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `scripts` list, `timepoint_done` event, and `bariera` (ReusableBarrier).
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
            # Pre-condition: `self.device.timepoint_done` is an Event object.
            # Invariant: The thread proceeds only after the supervisor signals completion of script assignment.
            self.device.timepoint_done.wait()

            self.thread_list = list() # Reset list of worker threads for the new timepoint.
            # Block Logic: Create `MiniT` instances for each assigned script and add to `thread_list`.
            # Invariant: Each script from `self.device.scripts` is assigned to a `MiniT` object.
            for (script, location) in self.device.scripts:
                minithrd = MiniT(neighbours, self.device, location, script)
                self.thread_list.append(minithrd)

            # Block Logic: Start all `MiniT` instances concurrently.
            # Invariant: All `MiniT` instances begin their `run` method concurrently.
            for i in range(len(self.thread_list)):
                self.thread_list[i].start()

            # Block Logic: Wait for all `MiniT` instances to complete their execution.
            # Invariant: The DeviceThread will not proceed until all its `MiniT` children have finished.
            for i in range(len(self.thread_list)):
                self.thread_list[i].join()

            # Functional Utility: Clear the `timepoint_done` event for the next timepoint.
            self.device.timepoint_done.clear()
            # Functional Utility: Clear the scripts list for the next timepoint.
            self.device.scripts = [] # Reset scripts list for the next timepoint.
            
            # Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
            # Invariant: All DeviceThread instances will reach this barrier before any proceeds to the next timepoint.
            self.device.bariera.wait()


class MiniT(Thread):
    """
    @brief A worker thread dedicated to executing a single assigned script for a Device instance.
    @details This thread processes a specific script for a given location, collects data from the
    parent device and its neighbors, executes the script's logic, and updates sensor data
    in a thread-safe manner using the parent device's `lacat_date` lock.
    @architectural_intent Enhances parallelism by allowing multiple scripts to run concurrently,
    with controlled resource access through the device's main data lock to prevent race conditions
    during data manipulation.
    """
    
    def __init__(self, neighbours, device, location, script):
        """
        @brief Initializes a new MiniT instance.
        @param neighbours (list): A list of neighboring Device objects from which to collect sensor data.
        @param device (Device): The parent Device object that this script thread serves.
        @param location (str): The location associated with the script for which data is processed.
        @param script (object): The script object to be executed.
        """
        Thread.__init__(self) # Initialize the base Thread class.
        self.device = device # Reference to the parent Device object.
        self.location = location # The sensor data location this script pertains to.
        self.script = script # The script to execute.
        self.neighbours = neighbours # List of neighboring devices.

    def run(self):
        """
        @brief The main execution logic for MiniT.
        @details This method collects data from neighboring devices and the parent device for
        the specified location, executes the assigned script, and then updates the relevant
        sensor data for the device and its neighbors. All data modifications are protected
        by acquiring and releasing `self.device.lacat_date` lock.
        @block_logic Collects data, executes a script, and updates data across devices for a specific location.
        @pre_condition `self.script` is an object with a `run` method, `self.device.lacat_date` is an initialized Lock.
        @invariant `script_data` is populated, `script` is run, and relevant data is updated under lock protection.
        """
        script_data = [] # List to accumulate data for the current script's execution.
        # Block Logic: Collect data from neighboring devices for the current location.
        # Invariant: `script_data` will contain data from all available neighbors for the given location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Collect data from the current device itself for the current location.
        # Invariant: If available, the device's own data for the location is added to `script_data`.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Execute the script if there is any data to process.
        # Pre-condition: `self.script` is an object with a `run` method, and `script_data` is a list of data.
        # Invariant: `result` holds the output of the script's execution.
        if script_data != []:
            result = self.script.run(script_data) # Execute the script with the collected data.

            # Block Logic: Propagate the script's result to neighboring devices, protected by `lacat_date`.
            # Invariant: All neighbors receive the updated data for the given location under mutual exclusion.
            for device in self.neighbours:
                device.lacat_date.acquire() # Acquire lock for neighbor device's data.
                device.set_data(self.location, result)
                device.lacat_date.release() # Release lock for neighbor device's data.

            # Block Logic: Update the current device's own data with the script's result, protected by `lacat_date`.
            # Invariant: The device's own data is updated under mutual exclusion.
            self.device.lacat_date.acquire() # Acquire lock for own device's data.
            self.device.set_data(self.location, result)
            self.device.lacat_date.release() # Release lock for own device's data.


class MiniT(Thread):
    
    def __init__(self, neighbours, device, location, script):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
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
                device.lacat_date.acquire()
                device.set_data(self.location, result)
                device.lacat_date.release()

            self.device.lacat_date.acquire()
            self.device.set_data(self.location, result)
            self.device.lacat_date.release()

class ReusableBarrier(object):
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        
        self.count_threads = self.num_threads
        
        self.cond = Condition()
        

    def wait(self):
        
        
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            
            self.cond.notify_all()
            
            self.count_threads = self.num_threads
        else:
            
            self.cond.wait()

        
        self.cond.release()
