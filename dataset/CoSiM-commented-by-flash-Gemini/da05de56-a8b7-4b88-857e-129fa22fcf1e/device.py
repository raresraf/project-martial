

"""
@brief This module defines the Barrier class for thread synchronization and the Device and DeviceThread classes,
which represent simulated devices in a distributed system.
@details The devices can execute scripts, interact with their neighbors, and share sensor data, simulating
a distributed sensor network or an IoT environment.
"""

from threading import Event, Thread, Condition


class Barrier():
    """
    @brief Implements a barrier for synchronizing multiple threads using class-level attributes.
    @details This barrier allows a set number of threads to wait until all participants arrive at a synchronization point,
    after which all threads are released simultaneously. The barrier is configured using static class variables
    `num_threads` and `count_threads`.
    @algorithm Condition Variable based synchronization with static thread counting.
    @time_complexity O(1) for `wait` operation, assuming constant time for underlying threading primitives.
    """
    
    num_threads = 0 # Static class variable: Total number of threads expected to reach the barrier.
    count_threads = 0 # Static class variable: Current count of threads waiting or yet to arrive at the barrier.

    def __init__(self):
        """
        @brief Initializes a new instance of the Barrier.
        @details Each barrier instance has its own condition variable, but shares the static thread counts.
        """
        self.cond = Condition()     # Condition variable used for thread synchronization.
        self.thread_event = Event() # Unused event object (potential for future use or remnant).

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all `num_threads` threads have arrived.
        @details Once all threads have arrived, they are all released, and the static thread count is reset
        for future reuse of the barrier.
        @block_logic Thread synchronization mechanism.
        @pre_condition `self.cond` is an initialized Condition object, `Barrier.num_threads` and `Barrier.count_threads`
                       accurately reflect the global state of threads.
        @invariant All threads attempting to pass the barrier will eventually be released together.
        """
        self.cond.acquire()         # Acquire the lock associated with the condition variable.
        Barrier.count_threads -= 1  # Decrement the count of threads yet to arrive (static).

        if Barrier.count_threads == 0:  # Check if this is the last thread to arrive at the barrier.
            self.cond.notify_all()      # Release all waiting threads.
            Barrier.count_threads = Barrier.num_threads # Reset the static thread count, making the barrier reusable.
        else:
            self.cond.wait()            # Wait for other threads to arrive (releases the lock implicitly).

        self.cond.release()         # Release the lock after being notified or decrementing the count.

    @staticmethod
    def add_thread():
        """
        @brief Increments the global count of threads participating in the barrier and resets the current count.
        @details This static method is called when a new thread is added to the system that needs to use this barrier.
        It effectively prepares the barrier for a new set of synchronization cycles.
        @block_logic Manages the static thread count for the barrier.
        @pre_condition This method should be called once for each thread that will use the barrier.
        @invariant `Barrier.num_threads` is incremented, and `Barrier.count_threads` is set equal to `Barrier.num_threads`.
        """
        Barrier.num_threads += 1    # Increment the total number of threads.
        Barrier.count_threads = Barrier.num_threads # Reset the current count to the total number of threads.


class Device(object):
    """
    @brief Represents a simulated device within a distributed sensor network.
    @details This class models an individual device that can collect sensor data,
    interact with a central supervisor, and execute scripts. It directly manages
    its own worker thread and uses a static Barrier for synchronization.
    @architectural_intent Acts as an autonomous agent in a distributed system,
                           capable of local data processing and communication with peers.
    """
    barrier = Barrier() # Class-level Barrier instance for synchronizing all Device threads.

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
        Device.barrier.add_thread() # Register this device's thread with the shared barrier.
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are ready for execution.
        self.scripts = []            # List to store assigned scripts and their locations.
        self.thread = DeviceThread(self) # Create a dedicated thread for this device.
        self.thread.start()          # Start the device's execution thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Placeholder method for setting up devices.
        @details In this implementation, the actual setup (like barrier initialization)
        is handled during the Device's __init__ method and the static Barrier class.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic No specific actions are performed here as setup is handled elsewhere.
        """
        pass

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific location.
        @details If a script is provided, it's appended to the device's script queue.
        If `script` is None, it signals that the script assignment phase for the current
        timepoint is complete, and the `script_received` event is set to unblock the device's thread.
        @param script (object): The script object to be executed, or None to signal end of assignments.
        @param location (str): The location associated with the script or data.
        @block_logic Manages the receipt of scripts and signals readiness for script execution.
        @pre_condition `self.scripts` is a list, `self.script_received` is an Event object.
        @invariant If `script` is not None, it's added to `self.scripts`. If `script` is None, `self.script_received` is set.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the queue.
        else:
            self.script_received.set() # Signal that all scripts for the current timepoint have been assigned.

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
        @param location (str): The location whose data is to be updated.
        @param data (object): The new data value for the specified location.
        @block_logic Updates the internal sensor data if the location exists.
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
    @brief Encapsulates the execution logic for a Device instance in a separate thread.
    @details This thread is responsible for synchronizing with other device threads using a shared barrier,
    fetching neighbor information, executing assigned scripts, and updating sensor data.
    @architectural_intent Enables concurrent operation of multiple Device instances,
                           allowing them to perform their tasks in parallel.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.
        @param device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the device thread.
        @details This method continuously retrieves neighbor information from the supervisor,
        synchronizes with other device threads via the shared barrier, waits for script assignments,
        and then executes those scripts. It processes incoming data from neighbors and its own sensors,
        runs scripts, and propagates results back to neighbors and itself. The loop terminates
        if the supervisor indicates that there are no more neighbors (signifying simulation end).
        @block_logic Orchestrates the device's main operational cycle within the simulation.
        @pre_condition `self.device` is an initialized Device object.
        @invariant The thread waits at the barrier, processes scripts if any, and updates data for each timepoint.
        """
        while True:
            # Functional Utility: Get information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            # Block Logic: Check if the simulation should terminate.
            # Pre-condition: `neighbours` list indicates the current state of the network.
            # Invariant: The loop terminates if no neighbors are returned by the supervisor.
            if neighbours is None:
                break
            
            # Block Logic: Synchronize all device threads before proceeding to the script execution phase.
            # Invariant: All active device threads will reach this barrier before any proceeds.
            Device.barrier.wait()
            
            # Block Logic: Wait until script assignments for the current timepoint are complete and signal that scripts are ready.
            # Pre-condition: `self.device.script_received` is an Event object.
            # Invariant: The thread proceeds only after the supervisor signals completion of script assignment.
            self.device.script_received.wait()
            
            # Functional Utility: Reset the event for the next timepoint's script assignment.
            self.device.script_received.clear()

            # Block Logic: Iterate through all assigned scripts for the current timepoint and execute them.
            # Pre-condition: `self.device.scripts` contains tuples of (script, location).
            # Invariant: Each script is run with collected data and results are propagated to neighbors and itself.
            for (script, location) in self.device.scripts:
                script_data = []
                
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
                    result = script.run(script_data)

                    # Block Logic: Propagate the script's result to neighboring devices.
                    # Invariant: All neighbors receive the updated data for the given location.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Functional Utility: Update the current device's own data with the script's result.
                    self.device.set_data(location, result)