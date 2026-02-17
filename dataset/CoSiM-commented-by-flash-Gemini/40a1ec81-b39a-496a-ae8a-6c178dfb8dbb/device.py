"""
@40a1ec81-b39a-496a-ae8a-6c178dfb8dbb/device.py
@brief Implements a simulated device in a distributed system, using a shared barrier and lock for synchronization.
This module defines a `Device` class that processes sensor data and executes scripts,
and a `DeviceThread` that manages the device's operational cycle. Synchronization
across devices is achieved through a globally shared `ReusableBarrierCond` and a `Lock`.
"""

import barrier
from threading import Event, Thread, Lock


class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its local sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread and globally shared synchronization primitives.
    """
    
    # Class-level shared attributes for synchronization across all device instances.
    barrier = None # Shared ReusableBarrierCond for global time step synchronization.
    lock = None # Shared Lock to protect access to sensor data across devices.

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

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up globally shared synchronization primitives (barrier and lock)
        and initializes the device's dedicated thread.
        Only the device with `device_id == 0` is responsible for initializing the
        shared barrier and lock.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: Initializes shared synchronization primitives if this is the first device.
        # Invariant: A single `ReusableBarrierCond` and `Lock` instance are created and shared across all devices.
        if(self.device_id == 0):
             Device.barrier = barrier.ReusableBarrierCond(len(devices)) # Initialize shared barrier.
             Device.lock = Lock() # Initialize shared lock.
        
        # Block Logic: Creates and starts the dedicated thread for this device.
        self.thread = DeviceThread(self, Device.barrier, Device.lock)
        self.thread.start()        
        

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        If a script is provided, it's added to the queue. If no script (None),
        it signals that the script assignment phase for the timepoint is complete.
        @param script: The script object to assign, or `None` to signal completion.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Block Logic: Signals that script assignments are complete and the timepoint is done.
            self.script_received.set()        
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        @param location: The key for the sensor data to be modified.
        @param data: The new data value to store.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread, waiting for its graceful completion.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    executing scripts, and coordinating with other device threads using a shared barrier and lock.
    Time Complexity: O(T * S * (N * D_access + D_script_run)) where T is the number of timepoints,
    S is the number of scripts per device, N is the number of neighbors, D_access is data access
    time, and D_script_run is script execution time.
    """
    
    def __init__(self, device, barrier, lock):
        """
        @brief Initializes a `DeviceThread` instance.
        @param device: The `Device` instance that this thread is responsible for.
        @param barrier: The shared `ReusableBarrierCond` for global synchronization.
        @param lock: The shared `Lock` for protecting access to sensor data.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.lock = lock

    def run(self):
        """
        @brief The main loop for the device's operational thread.
        Block Logic:
        1. Continuously synchronizes with all other device threads using the shared barrier.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           proceed, ensuring synchronized advancement of the simulation.
        2. Fetches neighbor information from the supervisor.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        3. Waits for `script_received` event to be set, indicating that all scripts for the timepoint are assigned.
        4. Clears the `script_received` event for the next timepoint.
        5. Executes assigned scripts: for each script, it acquires the shared lock to get data from neighbors and itself,
           runs the script, and then acquires the lock again to update data on neighbors and itself.
           Invariant: Data access and modification are protected by a global lock.
        6. Waits for the `timepoint_done` event to be set, indicating that all processing for the current timepoint is complete.
        """
        while True:
            # Block Logic: Synchronizes all device threads at the start of each timepoint.
            self.barrier.wait()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits until the device has received all scripts for the current timepoint.
            self.device.script_received.wait()
            # Block Logic: Resets the `script_received` event for the next timepoint.
            self.device.script_received.clear()

            # Block Logic: Processes each script assigned to the device for the current timepoint.
            # Invariant: Each script retrieves data from neighbors and itself, executes, and updates data.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Block Logic: Acquires global lock to get data from neighboring devices safely.
                for device in neighbours:
                    self.lock.acquire()
                    data = device.get_data(location)
                    self.lock.release()
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Acquires global lock to get data from its own device safely.
                self.lock.acquire()
                data = self.device.get_data(location)
                self.lock.release()
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if any data was collected and propagates the result.
                if script_data != []:
                    
                    result = script.run(script_data)

                    # Block Logic: Acquires global lock to update data on neighboring devices safely.
                    for device in neighbours:
                        self.lock.acquire()
                        device.set_data(location, result)
                        self.lock.release()
                    
                    # Block Logic: Acquires global lock to update data on its own device safely.
                    self.lock.acquire()
                    self.device.set_data(location, result)
                    self.lock.release()

            # Block Logic: Waits for the `timepoint_done` event, signaled after all scripts have been processed.
            self.device.timepoint_done.wait()
