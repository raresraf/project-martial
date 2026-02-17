
"""
@3c70c51e-fb9f-4b44-99e4-f3dd4e9b450b/device.py
@brief Implements a simulated device in a distributed environment, featuring multi-threaded script execution and manual time-step synchronization.
This module defines a `Device` that processes sensor data, communicates with neighbors, and executes scripts
through a `DeviceThread` and a nested `RunScripts` thread. Synchronization across all devices
is managed through a custom counting mechanism and `Event` objects.
"""

import threading
from threading import Event, Thread, Lock, Semaphore, current_thread

class Device(object):
    """
    @brief Represents a single device in the distributed simulation.
    Manages its local state, assigned scripts, and coordinates its activities
    within the overall system.
    """
        
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's sensor readings.
        @param supervisor: The supervisor object that orchestrates the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.next_iteration = True  
        self.thread = DeviceThread(self)
        self.set_data_lock = Lock() # Protects access to sensor_data and step.
        self.step = 0 # Tracks the current simulation step for this device.


        self.all_devices = [] # List of all devices in the simulation.
        self.all_devices_count = 0 # Total count of all devices.
        self.new_time = Event() # Event to signal the start of a new time step.
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the list of all devices in the simulation for global synchronization.
        @param devices: A list of all Device instances.
        Precondition: This method is called once during simulation initialization.
        """
        self.all_devices = devices
        self.all_devices_count = len(self.all_devices)

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.
        If `script` is `None`, it signals that the current timepoint is done for this device.
        @param script: The script object to assign, or `None` to signal timepoint completion.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Block Logic: Signals that the device has received all scripts for the current timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location, protected by a lock.
        @param location: The key for the sensor data to be modified.
        @param data: The new data value to store.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            # Block Logic: Ensures atomic update of sensor data using a lock.
            with self.set_data_lock:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's main thread by waiting for its completion.
        """
        self.thread.join()

    def increment_step(self):
        """
        @brief Increments the simulation step counter for this device, protected by a lock.
        This is part of the manual synchronization mechanism to track global time steps.
        """
        # Block Logic: Atomically increments the `step` counter for the device.
        with self.set_data_lock:
            self.step += 1
                        
class DeviceThread(Thread):
    """
    @brief The main execution thread for a `Device` instance.
    This thread manages the time-step progression, script execution, and global synchronization
    across all devices in the simulation.
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
        1. Continuously signals the start of a new time step for this device.
        2. Retrieves neighbor information from the supervisor.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        3. Waits for the `timepoint_done` event, indicating all scripts are assigned or timepoint complete.
        4. Initiates a `RunScripts` thread to execute assigned scripts concurrently.
        5. Waits for the `RunScripts` thread to complete.
        6. Increments its own `step` counter.
        7. Checks if all devices have completed the current step. If so, signals `new_time` for all devices.
           Otherwise, it waits for `new_time` to be signaled by another device.
           Invariant: All devices proceed to the next simulation step synchronously.
        """
        while True: 
            self.device.new_time.set() # Signal that this device is ready for a new time step.
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).
            self.device.timepoint_done.wait()
            # Block Logic: Creates and starts a `RunScripts` thread to execute the device's scripts for the current timepoint.
            run_thread = RunScripts(self.device, neighbours)
            run_thread.start()
            # Block Logic: Waits for the `RunScripts` thread to complete its execution before proceeding.
            run_thread.join()

            # Block Logic: Increments the local step counter to mark progress.
            self.device.increment_step()
            count = 0
            # Block Logic: Counts how many devices have completed the current step.
            # This is part of a manual barrier synchronization mechanism.
            for d in self.device.all_devices:
                if d.step == self.device.step:
                    count += 1
                    
            # Block Logic: If all devices have completed the current step, signals all devices to proceed to the next.
            # Otherwise, waits for another device to signal the start of the next time step.
            if count == self.device.all_devices_count:
                for d in self.device.all_devices:
                    d.new_time.set()
            else:
                self.device.new_time.wait()

class RunScripts(Thread):
    """
    @brief A helper thread responsible for executing all assigned scripts for a device
    at a specific timepoint.
    This thread collects data from the device and its neighbors, runs the script,
    and then updates the relevant sensor data on both the device and its neighbors.
    """
    def __init__(self, device, neighbours):
        """
        @brief Initializes a `RunScripts` instance.
        @param device: The parent `Device` instance for which scripts are being run.
        @param neighbours: A list of neighboring `Device` instances.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for the `RunScripts` thread.
        Block Logic:
        1. Clears the `new_time` event, signaling that this device is currently processing.
        2. Iterates through each assigned script, collecting relevant data from neighbors and itself.
        3. Executes the script if data is available.
        4. Propagates the script's result back to neighbors and its own device.
           Invariant: Sensor data on relevant devices is updated based on script execution.
        """
        self.device.new_time.clear() # Clear the event to indicate processing is ongoing.
        
        # Block Logic: Iterates through all scripts assigned to the device for the current timepoint.
        for (script, location) in self.device.scripts:
            script_data = []
            
            # Block Logic: Collects data from neighboring devices for the specified location.
            # Note: This implementation does not use explicit locks for neighbor data access,
            # which might lead to race conditions if not handled by higher-level mechanisms.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Collects data from its own device for the specified location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script if any data was collected and propagates the result.
            if script_data != []:
                
                result = script.run(script_data)

                # Block Logic: Updates neighboring devices with the script's result.
                for device in self.neighbours:
                    device.set_data(location, result)
                
                # Block Logic: Updates its own device's data with the script's result.
                self.device.set_data(location, result)
