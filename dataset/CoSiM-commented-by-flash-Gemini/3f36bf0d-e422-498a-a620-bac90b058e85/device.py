
"""
@3f36bf0d-e422-498a-a620-bac90b058e85/device.py
@brief Implements a simulated device for a distributed sensor network using a thread pool for concurrent script execution.
This module defines a `Device` that processes sensor data, interacts with neighbors,
and executes scripts. It features a `DeviceThread` that manages script execution,
a `ReusableBarrierSem` for global synchronization, and per-location locks for data consistency.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread, a shared barrier, and location-specific locks.
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
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

        self.timepoint_done = Event() # Event to signal completion of a timepoint's setup.
        self.script_received = Event() # Event to signal that a script has been assigned.
        self.barrier = None # Shared barrier for synchronizing all device threads.
        self.location_lock = None # Dictionary to hold locks for each sensor data location.
        self.lock = None # General lock for the device's sensor_data access.

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources like the barrier and location-specific locks among devices.
        Only the device with `device_id == 0` initializes these shared resources,
        which are then distributed to other devices.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup, typically by device 0.
        """
        # Block Logic: Device 0 initializes shared resources (global lock, barrier, location-specific locks).
        # Invariant: These shared objects are created once and then referenced by all other devices.
        if self.device_id == 0:
            self.lock = Lock() # Global lock for entire sensor_data dictionary access.
            self.barrier = ReusableBarrierSem(len(devices)) # Reusable barrier for all threads.
            self.location_lock = {} # Dictionary to store locks for individual data locations.
            # Block Logic: Distributes shared resources and initializes location locks for all devices.
            for device in devices:
                device.location_lock = self.location_lock
                for location in device.sensor_data:
                    self.location_lock[location] = Lock() # Initialize a lock for each data location.
                    if device.device_id != 0:
                        device.barrier = self.barrier
                        device.lock = self.lock


    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        Signals that a script has been received, or that a timepoint is done if no script.
        @param script: The script object to be executed, or `None` to signal completion.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Block Logic: Signals completion of timepoint setup if no script is assigned.
            self.timepoint_done.set()
            

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location, protected by the device's global lock.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        # Block Logic: Acquires the global device lock to ensure thread-safe access to sensor_data.
        with self.lock:
            res = self.sensor_data[location] if location in self.sensor_data else None
        return res

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location, protected by the device's global lock.
        @param location: The key for the sensor data to be modified.
        @param data: The new data value to store.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        # Block Logic: Acquires the global device lock to ensure thread-safe modification of sensor_data.
        with self.lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread, waiting for its graceful completion.
        """
        self.thread.join()

    def run_script(self, script, location, neighbours):
        """
        @brief Executes a single script for a given location, collecting data from neighbors and itself.
        This method acquires a lock specific to the `location` to ensure atomic updates for that data point.
        @param script: The script object to execute.
        @param location: The data location the script operates on.
        @param neighbours: A list of neighboring Device instances.
        """
        # Block Logic: Acquires a lock specific to the data location to prevent race conditions during script execution.
        self.location_lock[location].acquire()
        script_data = []
        
        # Block Logic: Collects data from neighboring devices for the specified location.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
            
        # Block Logic: Collects data from its own device for the specified location.
        data = self.get_data(location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script if data is available and propagates the result to neighbors and itself.
        if script_data != []:
            
            result = script.run(script_data)
            

            # Block Logic: Updates neighboring devices with the script's result.
            for device in neighbours:
                device.set_data(location, result)
            # Block Logic: Updates its own device's data with the script's result.
            self.set_data(location, result)

        # Block Logic: Releases the location-specific lock after script execution and data propagation.
        self.location_lock[location].release()
        

class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    executing scripts concurrently via a thread pool, and synchronizing with other device
    threads using a reusable barrier.
    Time Complexity: O(T * S_avg * (N + D)) where T is the number of timepoints, S_avg is the average number
    of scripts per timepoint, N is the number of neighbors, and D is the data retrieval/setting operations.
    The concurrency of `run_script` helps mitigate the script execution time.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a `DeviceThread` instance.
        @param device: The `Device` instance that this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        
    def run(self):
        """
        @brief The main loop for the device's operational thread.
        Block Logic:
        1. Continuously fetches neighbor information from the supervisor.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        2. Waits for the `timepoint_done` event to be set, indicating that scripts are ready to be processed.
        3. Manages a thread pool to concurrently execute `run_script` for each assigned script.
           Invariant: A maximum of 8 scripts are run in parallel at any given time.
        4. Waits for all initiated `run_script` threads to complete.
        5. Synchronizes with all other device threads using a shared barrier.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        6. Clears the `timepoint_done` event for the next timepoint.
        """
        while True:    
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).
            self.device.timepoint_done.wait()

            
            thread_list = [] # List to keep track of currently running script threads.

            # Block Logic: Iterates through assigned scripts and manages their concurrent execution using a thread pool.
            # Invariant: At most 8 `run_script` threads are active simultaneously.
            for (script, location) in self.device.scripts:
                
                # Block Logic: If the thread pool is not full, starts a new thread for script execution.
                if len(thread_list) < 8:
                    
                    t = Thread(target=self.device.run_script, args=(script, location, neighbours))
                    t.start()
                    thread_list.append(t)
                else:
                    # Block Logic: If the thread pool is full, waits for the oldest thread to complete before starting a new one.
                    out_thread = thread_list.pop(0) # Remove the oldest thread from the list.
                    out_thread.join() # Wait for it to finish.
                    
                    # Start a new thread after one has completed.
                    t = Thread(target=self.device.run_script, args=(script, location, neighbours))
                    t.start()
                    thread_list.append(t)

            # Block Logic: Waits for any remaining `run_script` threads in the pool to complete.
            for thread in thread_list:
                thread.join()
                
            # Block Logic: Synchronizes with other device threads using a shared barrier,
            # ensuring all devices complete their processing before proceeding.
            self.device.barrier.wait()
            # Block Logic: Clears the `timepoint_done` event for the next timepoint.
            self.device.timepoint_done.clear()
            
