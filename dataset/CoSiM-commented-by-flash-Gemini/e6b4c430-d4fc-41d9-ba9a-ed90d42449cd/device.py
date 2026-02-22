

"""
@brief This module defines `ReusableBarrier`, `Device`, and `DeviceThread` classes for simulating a distributed system.
@details It features a reusable barrier for inter-device synchronization, where each `Device` executes scripts
concurrently by spawning temporary threads within its dedicated `DeviceThread`, managing access to location-specific
sensor data with a shared list of locks. This design optimizes concurrent data processing and ensures data
integrity within a distributed sensor network simulation.
"""

from threading import * # Import all threading components directly.
import Queue # Imported but not explicitly used in the provided code snippet for actual queueing logic.

class ReusableBarrier():
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
        self.count_threads -= 1; # Decrement the count of threads yet to arrive.
        if self.count_threads == 0: # Check if this is the last thread to arrive at the barrier.
            self.cond.notify_all() # Release all waiting threads.
            self.count_threads = self.num_threads # Reset the thread count, making the barrier reusable.
        else:
            self.cond.wait(); # Wait for other threads to arrive (releases the lock implicitly).
        self.cond.release(); # Release the lock after being notified or decrementing the count.

class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.
    @details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
    It can receive and queue scripts for execution, which are then processed concurrently by its
    dedicated `DeviceThread` by spawning temporary worker threads. Synchronization across devices
    is managed by a shared `ReusableBarrier`, and thread-safe access to per-location sensor data
    is ensured by a shared list of `Lock` objects (`locationLock`).
    @architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
    processing and communication with peers, implementing explicit synchronization and granular locking
    for data consistency, with concurrent script execution.
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
        self.barrier = None          # Reference to the shared ReusableBarrier for inter-device synchronization.
        self.locationLock = None     # Reference to the shared list of Locks for location-specific data access.
        self.thread.start()          # Start the device's execution thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives (barrier and location-specific locks) for all devices.
        @details This method ensures that a single `ReusableBarrier` and a shared list of `Lock` objects
        (`locationLock`) are created by device 0 and then distributed among all other devices in the simulation.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Centralized initialization and distribution of shared synchronization
                     and mutual exclusion primitives.
        @pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
        @invariant `self.barrier` refers to a globally shared `ReusableBarrier` instance after setup.
                   `self.locationLock` is a shared list of `Lock` objects, ensuring consistent access to locations.
        """
        if self.device_id == 0: # Only device 0 is responsible for initializing and distributing shared resources.
            # Functional Utility: Create a new reusable barrier, sized for all devices.
            self.barrier = ReusableBarrier(len(devices))

            # Block Logic: Initialize a shared list of locks for up to 10000 locations.
            # This pre-allocates locks for potential sensor data locations.
            self.locationLock = [] # Initialize as an empty list.
            for i in range(0, 10000): # Create 10000 Lock objects.
                loc = Lock()
                self.locationLock.append(loc)

            # Block Logic: Distribute the newly created barrier and location locks to all devices.
            # Invariant: All devices in `devices` receive a reference to the shared `self.barrier` and `self.locationLock`.
            for i in devices:
                i.barrier = self.barrier
                i.locationLock = self.locationLock

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
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.
        @details This method updates the internal sensor data if the location exists.
        It's assumed that external synchronization (e.g., through `DeviceThread`'s `run_script` method)
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
    @details This thread orchestrates the device's operational cycle, including
    fetching neighbor information, waiting for script assignments, and then processing
    these scripts concurrently by spawning temporary worker threads. It ensures
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

    def run_script(self, script, location, neighbours):
        """
        @brief Executes a single script in a separate temporary thread.
        @details This method acquires a location-specific lock from `self.device.locationLock`,
        collects data from neighboring devices and the parent device for the specified location,
        runs the given script with this collected data, and then propagates the results back
        to the neighbors and the parent device. Finally, it releases the location-specific lock.
        @param script (object): The script object to be executed.
        @param location (str): The location relevant to the script and data processing.
        @param neighbours (list): A list of neighboring Device objects.
        @block_logic Collects data, executes a script, and updates data across devices for a specific location.
        @pre_condition `script` has a `run` method, `self.device.locationLock` contains a Lock for `location`.
        @invariant `script_data` is populated, `script` is run, and relevant data is updated under lock protection.
        """
        script_data = [] # List to accumulate data for the current script's execution.
        # Block Logic: Acquire a lock for the specific location to ensure exclusive access to its data.
        # Invariant: Only one script thread can modify or read data for `location` at a time.
        with self.device.locationLock[location]: # Uses `with` statement for automatic lock acquisition and release.

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


    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        @details This method continuously monitors the simulation state. For each timepoint,
        it retrieves neighbor information from the supervisor. If neighbors are available,
        it waits until `timepoint_done` is set (signaling that script assignments are complete),
        then processes the assigned scripts by creating temporary `Thread` objects, each executing
        `run_script`. After all temporary threads complete, it clears `script_received` and
        `timepoint_done`, and finally synchronizes with other `DeviceThread` instances via the
        shared `ReusableBarrier`. The loop terminates when the supervisor signals the end
        of the simulation.
        @block_logic Orchestrates the device's operational cycle, handling timepoint progression,
        concurrent script execution via temporary threads, and inter-device synchronization.
        @pre_condition `self.device` is an initialized Device object with access to `supervisor`,
                       `scripts` list, `timepoint_done` event, `script_received` event, and `barrier`.
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

            # Block Logic: Prepare a list of scripts to be executed concurrently.
            # Invariant: `queue` contains all scripts assigned for the current timepoint, along with context.
            queue = []
            for (script, location) in self.device.scripts:
                queue.append((script, location, neighbours))

            subThList = [] # List to hold temporary worker threads.
            # Block Logic: Create and append temporary worker threads to `subThList`.
            # Invariant: Each script from `queue` is assigned to a `Thread` object.
            while len(queue) > 0:
                # Functional Utility: Create a new temporary thread to run `run_script` with script details.
                subThList.append(Thread(target = self.run_script, args = queue.pop()))

            # Block Logic: Start all temporary worker threads concurrently.
            # Invariant: All temporary threads begin their `run_script` method concurrently.
            for t in subThList:
                t.start()

            # Block Logic: Wait for all temporary worker threads to complete their execution.
            # Invariant: The DeviceThread will not proceed until all its temporary worker threads have finished.
            for t in subThList:
                t.join()

            # Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
            # Invariant: All DeviceThread instances in the simulation will reach this barrier before any proceeds to the next timepoint.
            self.device.barrier.wait()

            # Functional Utility: Clear events and scripts list for the next timepoint.
            self.device.script_received.clear() # Clear the script received event.
            self.device.scripts = [] # Reset scripts list for the next timepoint.
            self.device.timepoint_done.clear() # Clear the timepoint done event.
