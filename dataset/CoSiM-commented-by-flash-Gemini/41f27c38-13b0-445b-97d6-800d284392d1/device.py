"""
@41f27c38-13b0-445b-97d6-800d284392d1/device.py
@brief Implements a simulated device for a distributed sensor network, utilizing a thread pool for concurrent script execution.
This module defines a `Device` that processes sensor data, interacts with neighbors,
and executes scripts. It features a `DeviceThread` that manages script execution,
a `ReusableBarrier` for global synchronization, and per-location semaphores for data consistency.
"""

from threading import Event, Thread, Semaphore
from concurrent.futures import wait, ALL_COMPLETED, ThreadPoolExecutor
from barrier import ReusableBarrier

class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread, a shared barrier, and location-specific semaphores.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing the device's local sensor readings.
        @param supervisor: The supervisor object responsible for managing the overall simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal that a script has been assigned.
        self.scripts = [] # List to store assigned scripts.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        # Initial barrier with 1 thread, to be reconfigured by setup_devices.
        self.barrier = ReusableBarrier(1) 
        self.thread = DeviceThread(self)
        # Dictionary of semaphores, one for each sensor data location, to control access.
        self.location_sems = {location : Semaphore(1) for location in sensor_data}
        self.thread.start()

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared reusable barrier for synchronization among all devices.
        Only the device with `device_id == 0` initializes the barrier, which is then
        assigned to all other devices.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup, typically by device 0.
        """
        # Block Logic: Device 0 initializes the shared barrier.
        # Invariant: A single `ReusableBarrier` instance is created and shared across all devices.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            # Block Logic: Assigns the initialized barrier to all devices.
            for dev in devices:
                dev.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        If a script is provided, it's added to the queue and the `script_received` event is set.
        If no script (i.e., `None`) is provided, it signals that the timepoint is done.
        @param script: The script object to assign, or `None` to signal completion.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Block Logic: Signals completion of the timepoint if no script is assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location, acquiring its corresponding semaphore.
        The semaphore ensures exclusive access to the data when being read/modified by scripts.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        if location in self.sensor_data:
            # Block Logic: Acquires the semaphore for this location to ensure exclusive read access.
            self.location_sems[location].acquire()
            data = self.sensor_data[location]
        else:
            data = None
        return data

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location, then releases its corresponding semaphore.
        @param location: The key for the sensor data to be modified.
        @param data: The new data value to store.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            # Block Logic: Releases the semaphore for this location after data has been updated.
            self.location_sems[location].release()

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread, waiting for its graceful completion.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    executing scripts concurrently via a thread pool, and coordinating with other device threads.
    Time Complexity: O(T * S_avg * (N + D)) where T is the number of timepoints, S_avg is the average number
    of scripts per timepoint, N is the number of neighbors, and D is the data retrieval/setting operations.
    The `ThreadPoolExecutor` enables parallel script execution to mitigate total time.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a `DeviceThread` instance.
        @param device: The `Device` instance that this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Initializes a thread pool with a maximum of 8 worker threads for concurrent script execution.
        self.thread_pool = ThreadPoolExecutor(8)
        self.neighbours = [] # Stores the list of neighboring devices.

    def gather_info(self, location):
        """
        @brief Collects sensor data for a specific location from neighboring devices and the current device.
        Each data retrieval from a neighbor or self will acquire the location's semaphore.
        @param location: The data location for which information is to be gathered.
        @return A list of collected sensor data from all relevant devices.
        """
        script_data = []
        # Block Logic: Iterates through neighbors to gather data, excluding itself to prevent double-counting.
        for device in self.neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        # Block Logic: Gathers data from the current device itself.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        return script_data

    def spread_info(self, result, location):
        """
        @brief Spreads the script execution result to neighboring devices and the current device.
        Each data update on a neighbor or self will release the location's semaphore.
        @param result: The result of the script execution.
        @param location: The data location to which the result pertains.
        """
        # Block Logic: Iterates through neighbors to update their data with the script result.
        for device in self.neighbours:
            if device.device_id != self.device.device_id:
                device.set_data(location, result)
        # Block Logic: Updates the current device's own data with the script result.
        self.device.set_data(location, result)

    def update(self, script, location):
        """
        @brief Orchestrates the gathering of information, execution of a script, and spreading of results for a single script.
        @param script: The script to be executed.
        @param location: The data location relevant to the script.
        """
        script_data = self.gather_info(location)
        result = None
        if script_data != []:
            result = script.run(script_data)
            self.spread_info(result, location)

    def run(self):
        """
        @brief The main loop for the device's operational thread.
        Block Logic:
        1. Continuously fetches neighbor information from the supervisor.
           Invariant: The loop terminates if `self.neighbours` is `None`, signaling the end of the simulation.
        2. Manages script execution by submitting `update` tasks to a `ThreadPoolExecutor`.
           It waits for `script_received` to be set (new scripts available) or `timepoint_done` (all scripts processed).
           Invariant: All scripts assigned for the current timepoint are submitted to the thread pool for concurrent execution.
        3. Waits for all submitted tasks in the thread pool to complete.
        4. Synchronizes with all other device threads using a shared barrier.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        5. Shuts down the thread pool upon simulation termination.
        """
        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Terminates the loop if no neighbors are returned (signaling end of simulation).
            if self.neighbours is None:
                break
            futures = [] # List to hold future objects returned by thread pool submissions.

            # Block Logic: Manages the state transitions for script assignment and timepoint completion.
            while True:
                # Block Logic: Checks if new scripts are received or if the timepoint is done.
                if self.device.script_received.is_set() or self.device.timepoint_done.wait():
                    # Block Logic: If new scripts are received, clears the event and submits each script to the thread pool.
                    if self.device.script_received.is_set():
                        
                        self.device.script_received.clear()
                        for (script, location) in self.device.scripts:
                            future = self.thread_pool.submit(self.update, script, location)
                            futures.append(future)
                    # Block Logic: If timepoint_done is signaled, clears it and breaks the loop, signaling all scripts have been handled.
                    else:
                        
                        
                        
                        self.device.timepoint_done.clear()
                        self.device.script_received.set() # Re-sets script_received to indicate readiness for next set of scripts.
                        break
            
            # Block Logic: Waits for all submitted script execution tasks to complete.
            wait(futures, timeout=None, return_when=ALL_COMPLETED)
            
            # Block Logic: Synchronizes with other device threads using a shared barrier.
            self.device.barrier.wait()
        # Block Logic: Shuts down the thread pool when the main loop terminates (simulation ends).
        self.thread_pool.shutdown()
