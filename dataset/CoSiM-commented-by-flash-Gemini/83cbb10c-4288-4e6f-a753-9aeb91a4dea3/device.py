


"""
@83cbb10c-4288-4e6f-a753-9aeb91a4dea3/device.py
@brief Defines the Device and DeviceThread classes for distributed simulation or data processing.

This module provides the foundational components for simulating a network of interconnected devices.
Each `Device` manages its own sensor data, interacts with a `Supervisor` for network topology,
and executes scripts in a dedicated `DeviceThread`. Synchronization across devices for
timepoint progression and script execution is managed via a `ReusableBarrier`.
"""

from threading import Event, Thread
import ReusableBarrier

class Device(object):
    """
    @brief Represents a simulated device in a distributed environment.

    Each Device instance manages its own sensor data, communicates with a supervisor
    to determine its neighbors, and processes assigned scripts within its dedicated thread.
    It utilizes a reusable barrier for synchronization across all active devices
    and events to signal script reception and timepoint completion.
    """
    

    reusable_barrier = ReusableBarrier.ReusableBarrier()

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        Registers the device with the global reusable barrier and sets up
        its unique identifier, initial sensor data, and a reference to the supervisor.
        It also initializes internal state for script management and threading.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's initial sensor readings.
        @param supervisor: A reference to the supervisor object managing the device network.
        """
        

        Device.reusable_barrier.add_thread()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Placeholder for setting up interactions with other devices.

        This method is intended to configure how this device interacts with a list
        of other devices in the simulation, though it currently performs no action.

        @param devices: A list of other Device instances in the simulation.
        """
        
        
        pass

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device.

        If a script is provided, it is added to the device's script queue along
        with the location it pertains to. If no script is provided (i.e., None),
        it signals that script reception is complete for the current timepoint,
        allowing the device thread to proceed.

        @param script: The script object to be executed, or None to signal completion.
        @param location: The data location (e.g., sensor ID) the script operates on.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()


    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The key identifying the sensor data to retrieve.
        @return: The sensor data at the specified location, or None if not found.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.

        Updates the sensor data if the location already exists in the device's
        sensor data dictionary.

        @param location: The key identifying the sensor data to update.
        @param data: The new data value to set for the specified location.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's processing thread.

        This method waits for the device's associated thread to complete
        its execution, ensuring a clean shutdown.
        """
        
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the execution lifecycle of a Device.

    This thread is responsible for continuously fetching neighbor information,
    synchronizing with other device threads, processing assigned scripts,
    and updating sensor data based on script execution. It runs until
    explicitly signaled to stop by the absence of network neighbors.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        Sets up the thread with a descriptive name and associates it with
        the Device instance it will manage.

        @param device: The Device instance this thread is responsible for.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device



    def run(self):
        """
        @brief The main execution loop for the device thread.

        This loop continuously performs the following steps:
        1. Retrieves the current set of neighboring devices from the supervisor.
        2. Terminates if no neighbors are found, indicating the end of the simulation for this device.
        3. Waits for all active device threads to reach a synchronization point (barrier).
        4. Waits for new scripts to be assigned by the supervisor.
        5. Clears the script received flag for the next timepoint.
        6. Iterates through all assigned scripts, collects relevant data from neighbors and itself,
           executes the script, and disseminates the results back to relevant devices.
        """
        while True:
            # Block Logic: Determine the current set of active neighbors for data exchange.
            # Functional Utility: Adapts the device's operational scope dynamically based on network topology.
            neighbours = self.device.supervisor.get_neighbours()
            # Invariant: If no neighbors are returned, the simulation for this device is complete.
            if neighbours is None:
                break

            # Block Logic: Synchronizes the execution of all active device threads.
            # Functional Utility: Ensures that all devices are ready to process the current timepoint's
            #                      scripts before proceeding, maintaining simulation consistency.
            Device.reusable_barrier.wait();
            # Block Logic: Pauses execution until new scripts are assigned to the device.
            # Functional Utility: Orchestrates script processing based on external directives,
            #                      typically from a supervisor or central orchestrator.
            self.device.script_received.wait()
            # Inline: Resets the script_received event, preparing it for the next script assignment cycle.
            self.device.script_received.clear();

            # Block Logic: Processes each assigned script using data from local sensors and neighbors.
            # Invariant: Each script operates on a specific 'location' and its collected data.
            for (script, location) in self.device.scripts:
                script_data = []
                # Block Logic: Aggregates sensor data from all neighboring devices for the current script's location.
                # Functional Utility: Gathers necessary context from the distributed environment for script execution.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Includes the device's own sensor data for the current script's location.
                # Functional Utility: Ensures self-awareness and self-contribution to script processing.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Block Logic: Executes the assigned script with the collected data.
                    # Architectural Intent: Decouples data processing logic from device management,
                    #                      allowing dynamic and flexible behavioral updates.
                    result = script.run(script_data)

                    # Block Logic: Disseminates the processed result to all neighboring devices.
                    # Functional Utility: Propagates local computations across the network,
                    #                      enabling distributed state updates.
                    for device in neighbours:
                        device.set_data(location, result)
                    # Block Logic: Updates the device's own sensor data with the processed result.
                    # Functional Utility: Reflects the outcome of local computations on the device's internal state.
                    self.device.set_data(location, result)

