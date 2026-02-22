

"""
@brief This module defines `ReusableBarrier`, `Device`, and `DeviceThread` classes for simulating a distributed system.
@details It features a condition-variable-based reusable barrier for inter-device synchronization and employs
a main device thread that executes scripts sequentially, protected by a global lock. This design ensures
coordinated execution and data consistency across multiple simulated devices.
"""

from threading import Event, Thread, Condition, Lock

class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.
    @details This barrier allows a set number of threads to wait until all participants arrive at a synchronization point,
    after which all threads are released simultaneously. The barrier can then be reused for subsequent synchronization points.
    @algorithm Condition Variable based synchronization.
    @time_complexity O(1) for `wait` operation, assuming constant time for underlying threading primitives.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes a new instance of the ReusableBarrier.
        @param num_threads (int): The total number of threads that must reach the barrier before it can be passed.
        """
        self.num_threads = num_threads # Total number of threads expected.
        self.count_threads = self.num_threads # Current count of threads waiting at the barrier.
        self.cond = Condition() # Condition variable used for thread synchronization.

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all `num_threads` threads have arrived.
        @details Once all threads have arrived, they are all released, and the barrier is reset for future use.
        @block_logic Thread synchronization mechanism.
        @pre_condition `self.cond` is an initialized Condition object, `self.count_threads` accurately reflects
                       the number of threads currently waiting or yet to arrive.
        @invariant All threads attempting to pass the barrier will eventually be released together.
        """
        self.cond.acquire() # Acquire the lock associated with the condition variable.
        self.count_threads -= 1 # Decrement the count of threads yet to arrive.
        if self.count_threads == 0: # Check if this is the last thread to arrive at the barrier.
            self.cond.notify_all() # Release all waiting threads.
            self.count_threads = self.num_threads # Reset the thread count, making the barrier reusable.
        else:
            self.cond.wait() # Wait for other threads to arrive (releases the lock implicitly).
        self.cond.release() # Release the lock after being notified or decrementing the count.


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It can receive and queue scripts for execution, which are then processed sequentially by its
    dedicated `DeviceThread`. Synchronization across all `Device` instances in the simulation
    is managed by a shared `ReusableBarrier`, and access to shared resources during script
    execution is protected by a global `Lock`.
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
    processing and communication with peers, implementing explicit barrier synchronization
    and a global mutual exclusion mechanism for coordinated and consistent execution across timepoints.
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
        self.barr = None             # Reference to the shared ReusableBarrier for inter-device synchronization.
        self.lock = None             # Reference to the shared Lock for protecting shared resources (e.g., sensor data updates).
        self.thread.start()          # Start the device's execution thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives (barrier and a global lock) for all devices.
        @details This method ensures that a single `ReusableBarrier` instance and a single `Lock` instance
        are created only once by device 0 (identified by `devices[0].device_id == self.device_id` and
        `devices[0].barr is None`), and then distributes them to all other devices in the simulation.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of shared synchronization
                     and mutual exclusion primitives.
        @pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
        @invariant `self.barr` refers to a globally shared `ReusableBarrier` instance after setup.
                   `self.lock` refers to a globally shared `Lock` instance after setup.
        """
        # Block Logic: Initialize shared barrier and lock only once by the device identified as the "first" device.
        # This double-checks both the barrier being None and the device_id matching the first device in the list.
        # Invariant: Global `barr` and `lock` become initialized and shared across devices.
        if devices[0].barr is None and devices[0].device_id == self.device_id:
            bariera = ReusableBarrier(len(devices)) # Create a new reusable barrier, sized for all devices.
            global_lock = Lock() # Create a single global lock for all devices.
            # Block Logic: Distribute the newly created barrier and lock to all devices.
            # Invariant: All devices in `devices` receive a reference to the shared `bariera` and `global_lock`.
            for i in devices:
                i.barr = bariera
            for j in devices:
                j.lock = global_lock

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
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.
        @details This method updates the internal sensor data if the location exists.
        It's assumed that external synchronization (e.g., through `DeviceThread`'s global lock)
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
    these scripts sequentially. Each script execution is protected by a global `Lock`
    (`self.device.lock`) to ensure mutual exclusion during data access and modification.
    It ensures inter-device synchronization through a shared `ReusableBarrier`.
    @architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
    ensuring proper coordination and data consistency within the distributed system
    through sequential script execution protected by a global lock and barrier synchronization.
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
        it retrieves neighbor information from the supervisor. If neighbors are available,
        it waits until `timepoint_done` is set (signaling that script assignments are complete),
        then sequentially processes each assigned script. For each script, it acquires the
        global `self.device.lock`, collects data from neighbors and its own sensors, executes
        the script, and propagates results back to neighbors and itself. After releasing the
        lock and processing all scripts for the timepoint, it clears `timepoint_done`, resets
        the script list, and finally synchronizes with other `DeviceThread` instances via the
        global `ReusableBarrier`. The loop terminates when the supervisor signals the end
        of the simulation.
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        sequential script execution with global mutual exclusion, and inter-device synchronization.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `scripts` list, `timepoint_done` event, `lock`, and `barr` (ReusableBarrier).
        @invariant The thread progresses through timepoints, processes scripts sequentially under
                   global lock protection, and ensures global synchronization.
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

            # Block Logic: Iterate through all assigned scripts for the current timepoint and execute them sequentially.
            # Pre-condition: `self.device.scripts` contains tuples of (script, location).
            # Invariant: Each script is run with collected data and results are propagated to neighbors and itself.
            for (script, location) in self.device.scripts:
                # Block Logic: Acquire the global lock to protect shared resources during script execution.
                # Invariant: Only one device's script can modify shared sensor data at a time.
                self.device.lock.acquire()
                script_data = [] # List to accumulate data for the current script's execution.
                
                # Block Logic: Collect data from neighboring devices for the current location.
                # Invariant: `script_data` will contain data from all available neighbors for the given location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collect data from the current device itself for the current location.
                # Invariant: If available, the device's own data for the location is added to `script_data`.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Execute the script if there is any data to process.
                # Pre-condition: `script` is an object with a `run` method, and `script_data` is a list of data.
                # Invariant: `result` holds the output of the script's execution.
                if script_data != []:
                    result = script.run(script_data) # Execute the script with the collected data.

                    # Block Logic: Propagate the script's result to neighboring devices.
                    # Invariant: All neighbors receive the updated data for the given location.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Functional Utility: Update the current device's own data with the script's result.
                    self.device.set_data(location, result)
                self.device.lock.release() # Release the global lock after processing the script.

            # Functional Utility: Clear the `timepoint_done` event for the next timepoint.
            self.device.timepoint_done.clear()
            # Functional Utility: Clear the scripts list for the next timepoint.
            self.device.scripts = [] # Reset scripts list for the next timepoint.
            
            # Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
            # Invariant: All DeviceThread instances will reach this barrier before any proceeds to the next timepoint.
            self.device.barr.wait()
