
"""
@73c0d468-96d1-4fd2-8f85-f19036243f21/ScriptExecutor.py
@brief Implements a multi-threaded device simulation with concurrent script execution and fine-grained locking.
This module defines `Device`, `DeviceThread`, and `ScriptExecutor` classes.
The `Device` manages sensor data and scripts, while `DeviceThread` orchestrates
the execution of multiple `ScriptExecutor` threads, using a thread pool pattern
with a `Semaphore`. Synchronization is handled by a `ReusableBarrier` for global
time-step coordination, and a list of `Lock` objects (`location_locks`) provides
per-location data protection across devices.
"""
from threading import Event, Thread, Lock, Semaphore
class ScriptExecutor(Thread):
	"""
	@brief A dedicated thread for executing a single script for a specific data location.
	This thread is responsible for gathering data, running the script, and then
	propagating the results to relevant devices, ensuring thread-safe access to data
	through location-specific `Lock` objects and concurrency control with a semaphore.
	"""
	
	def __init__(self, device, neighbours, current_location, script):
		"""
		@brief Initializes a `ScriptExecutor` instance.
		@param device: The parent `Device` instance for which the script is being run.
		@param neighbours: A list of neighboring `Device` instances.
		@param current_location: The data location that the script operates on.
		@param script: The script object to execute.
		"""
		Thread.__init__(self)
		self.device = device


		self.script = script
		self.neighbours = neighbours
		self.current_location = current_location

	def run(self):
		"""
		@brief The main execution logic for `ScriptExecutor`.
		Block Logic:
		1. Acquires the location-specific lock for `current_location`.
		2. Collects data from neighboring devices and its own device for the specified `current_location`.
		3. Executes the assigned `script` if any data was collected.
		4. Propagates the script's `result` to neighboring devices and its own device.
		5. Releases the location-specific lock.
		6. Releases a permit back to the global semaphore (`threads_limit`), allowing another script thread to start.
		Invariant: All data access and modification for a given `current_location` are protected by a shared `Lock`,
		and overall script concurrency is limited by a `Semaphore`.
		"""
		# Block Logic: Acquires the location-specific lock to ensure exclusive access to data at this `current_location`.
		self.device.location_locks[self.current_location].acquire()
		script_data = []

		# Block Logic: Collects data from neighboring devices for the specified location.
		for device in self.neighbours:
			data = device.get_data(self.current_location)
			if data is not None:
				script_data.append(data)
		
		# Block Logic: Collects data from its own device for the specified location.
		data = self.device.get_data(self.current_location)
		if data is not None:
			script_data.append(data)

		# Block Logic: Executes the script if any data was collected and propagates the result.
		if script_data != []:
			
			result = self.script.run(script_data)

			# Block Logic: Updates neighboring devices with the script's result.
			for device in self.neighbours:
				device.set_data(self.current_location, result)
			
			# Block Logic: Updates its own device's data with the script's result.
			self.device.set_data(self.current_location, result)
		# Block Logic: Releases the location-specific lock after all data operations for this script are complete.
		self.device.location_locks[self.current_location].release()
		# Block Logic: Releases a permit back to the global semaphore, allowing another `ScriptExecutor` thread to start.
		self.device.threads_limit.release()


# Assumed ReusableBarrier class definition from ReusableBarrier.py
# (Not part of this file's direct code, but used by Device)
# from ReusableBarrier import ReusableBarrier 

class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its local sensor data, assigned scripts, and coordinates its operation
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
        self.script_received = Event() # Event to signal that a script has been assigned.
        self.scripts = [] # List to store assigned scripts.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.

        # Block Logic: Shared list of Locks, one for each unique data location across all devices.
        self.location_locks = []

        # Block Logic: Shared barrier for global time step synchronization, to be initialized by device 0.
        self.barrier = None

        # Block Logic: List to hold `ScriptExecutor` instances, and a Semaphore to limit their concurrent execution.
        self.threads = []
        self.threads_limit = Semaphore(8) # Limits to 8 concurrent script threads.
        self.thread = DeviceThread(self) # The dedicated thread for this device.
        self.thread.start()

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (global barrier and location-specific locks) among all devices.
        Only the device with `device_id == 0` is responsible for initializing these resources,
        which are then distributed to all other devices.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: The device with `device_id == 0` initializes the shared `location_locks` and `barrier`.
        if self.device_id == 0:
            
            my_locations = [] # Temporarily stores all unique locations across all devices.

            # Block Logic: Gathers all unique sensor data locations from all devices.
            for device in devices:
                locations = device.sensor_data.keys()
                for location in locations:
                    # Block Logic: If a new location is found, add it and create a new Lock for it.
                    if location not in my_locations:
                        my_locations.append(location)
                        self.location_locks.append(Lock())

            # Block Logic: Initializes the shared `ReusableBarrier` with the total number of devices.
            barrier = ReusableBarrier(len(devices))

            # Block Logic: Distributes the initialized shared `location_locks` and `barrier` to all devices.
            for device in devices:
                device.location_locks = self.location_locks
                device.barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        Signals that a script has been received. If no script is provided, it signals
        `timepoint_done`, indicating the completion of script assignment for the current timepoint.
        @param script: The script object to assign.
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
        @brief Retrieves sensor data for a given location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        `ScriptExecutor` will acquire the appropriate lock from `location_locks` before calling this method.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        `ScriptExecutor` will acquire the appropriate lock from `location_locks` before calling this method.
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
    executing scripts concurrently via `ScriptExecutor` instances, and coordinating with
    other device threads using a shared `ReusableBarrier`.
    Time Complexity: O(T * S_total * (N * D_access + D_script_run)) where T is the number of timepoints,
    S_total is the total number of scripts executed by the device, N is the number of neighbors,
    D_access is data access time, and D_script_run is script execution time.
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
        2. Clears the list of `ScriptExecutor` threads for the current timepoint.
        3. Waits for the `timepoint_done` event to be set, indicating that scripts are ready to be processed.
        4. Creates `ScriptExecutor` instances for each assigned script. Concurrency is limited by `threads_limit`.
        5. Starts all `ScriptExecutor` instances.
        6. Waits for all `ScriptExecutor` instances to complete.
        7. Clears the `timepoint_done` event for the next cycle.
        8. Synchronizes with all other device threads using a shared `ReusableBarrier`.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            self.device.threads = [] # Clears the list of script executor threads for the current timepoint.

            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).
            self.device.timepoint_done.wait()

            # Block Logic: Iterates through assigned scripts, creating and preparing `ScriptExecutor` instances.
            # Concurrency is controlled by `self.device.threads_limit` (a Semaphore).
            for (script, location) in self.device.scripts:
                    
                # Block Logic: Acquires a permit from the semaphore before appending a new `ScriptExecutor`.
                # This ensures the thread pool limit is respected.
                self.device.threads_limit.acquire()
                self.device.threads.append(ScriptExecutor(
                    self.device, neighbours, location, script
                    ))
            # Block Logic: Starts all created `ScriptExecutor` instances.
            for thread in self.device.threads:
                thread.start()
            # Block Logic: Waits for all initiated `ScriptExecutor` instances to complete their execution.
            for thread in self.device.threads:
                thread.join()
            # Block Logic: Clears the `timepoint_done` event for the next timepoint cycle.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes with other device threads using a shared barrier,
            # ensuring all devices complete their processing before proceeding.
            self.device.barrier.wait()
