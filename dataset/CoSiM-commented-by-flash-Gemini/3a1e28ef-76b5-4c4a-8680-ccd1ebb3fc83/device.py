

"""
@3a1e28ef-76b5-4c4a-8680-ccd1ebb3fc83/device.py
@brief Implements a device and its operational thread for a simulated distributed system.
This module defines how a device processes sensor data, interacts with neighboring devices,
and executes scripts in a synchronized manner using a reusable barrier.
"""

from threading import *
from barrier import *


class Device(object):
    """
    @brief Represents a single device in the distributed system.
    Manages its own sensor data, assigned scripts, and interacts with a supervisor.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary holding the device's sensor readings.
        @param supervisor: An object managing the overall system and device interactions.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.lock = Lock()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the reusable barrier for synchronization among all device threads.
        This method is typically called once during system initialization to set the total
        number of threads participating in the barrier.
        @param devices: A list of all Device instances in the system.
        Precondition: `DeviceThread.barr` must be an initialized `ReusableBarrierCond` instance.
        """
        DeviceThread.barr.set_th(len(devices))


    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.
        If no script is provided (i.e., script is None), it signals the timepoint as done.
        @param script: The script object to be executed.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Block Logic: Signals that a script has been received and the timepoint is done.
            # This path is typically taken when there are no more scripts for the current timepoint.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.
        @param location: The key corresponding to the desired sensor data.
        @return The sensor data if the location exists, otherwise None.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data for a specific location.
        @param location: The key for the sensor data to be updated.
        @param data: The new data value to set.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the associated device thread by waiting for its completion.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the execution lifecycle of a device within the simulated environment.
    This includes fetching neighbor information, coordinating execution steps using a barrier,
    and running assigned scripts that may involve data exchange with neighbors.
    Time Complexity: O(T * S * (N + D)) where T is the number of timepoints, S is the number of scripts per device,
    N is the number of neighbors, and D is the data retrieval/setting operations.
    """
    barr = ReusableBarrierCond() # Shared barrier for all device threads.

    def __init__(self, device):
        """
        @brief Initializes a DeviceThread instance.
        @param device: The Device instance that this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        

    def run(self):
        """
        @brief The main execution loop for the device thread.
        Block Logic:
        1. Continuously fetches neighbor devices from the supervisor.
           Invariant: If `neighbours` is None, the simulation is over, and the loop breaks.
        2. Synchronizes all device threads using a shared barrier.
           Invariant: All active `DeviceThread` instances must reach this point before any can proceed,
           ensuring timepoint alignment across the system.
        3. Waits for the `timepoint_done` event to be set, indicating readiness for script execution.
           Precondition: `self.device.timepoint_done` is set by the supervisor or `assign_script`.
        4. Iterates through assigned scripts, collecting data from neighbors and self, executing scripts,
           and propagating results.
           Invariant: Each script processes its relevant data and updates both local and neighbor device states.
        5. Clears the `timepoint_done` event, resetting for the next timepoint.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Block Logic: Ensures all device threads are synchronized before proceeding with a timepoint.
            DeviceThread.barr.wait()
            
            # Block Logic: Waits for a signal from the supervisor or `assign_script` that the timepoint is ready.
            self.device.timepoint_done.wait()
            
            # Block Logic: Processes each assigned script.
            # Invariant: Each script retrieves data, executes, and then updates relevant devices.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Block Logic: Acquires locks and collects data from neighbor devices for the current script's location.
                # Invariant: Locks are acquired before accessing neighbor data to maintain consistency.
                for device in neighbours:
                    device.lock.acquire()
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collects its own sensor data for the current script's location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if data is available and propagates the results.
                if script_data != []:
                    
                    result = script.run(script_data)
                    
                    # Block Logic: Releases locks and updates neighbor devices with the script result.
                    # Invariant: Locks are released after updating data.
                    for device in neighbours:
                        device.lock.release()
                        device.set_data(location, result)

                    # Block Logic: Updates the current device's sensor data with the script result.
                    self.device.set_data(location, result)
            
            # Block Logic: Resets the timepoint_done event for the next cycle.
            self.device.timepoint_done.clear()
