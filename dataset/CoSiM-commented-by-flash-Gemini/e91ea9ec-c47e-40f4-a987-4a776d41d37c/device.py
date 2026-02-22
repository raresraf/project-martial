"""
@brief This module defines the Device and DeviceThread classes, representing simulated devices
in a distributed system.
@details It utilizes a condition-variable based reusable barrier (`ReusableBarrierCond`) for synchronization
and distinct locks (`lock_for_data`, `lock_for_location`) for managing access to sensor data and locations,
respectively.
"""

from barrier import ReusableBarrierCond
from threading import Lock, Event, Thread

class Device(object):
	"""
	@brief Represents a simulated device in a distributed sensor network.
	@details This class manages a device's unique identifier, sensor data, and interactions
	with a supervisor. It is capable of receiving and executing scripts, which can modify
	its own sensor data and affect neighboring devices. It employs a shared reusable barrier
	for inter-device synchronization and distinct locks for managing data consistency.
	@architectural_intent Acts as an autonomous agent in a distributed system, capable of
	local data processing and communication with peers, with explicit synchronization
	and locking mechanisms for robust concurrent operation.
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
		self.script_received = Event() # Event to signal when new scripts have been assigned.
		self.scripts = []            # List to store assigned scripts and their locations.
		self.timepoint_done = Event()  # Event to signal completion of a timepoint's processing (unused in this version).
		self.thread = DeviceThread(self) # The main worker thread for this device.
		self.thread.start()          # Start the device's execution thread.
		self.list_of_devices = []    # List to store references to all other devices in the simulation.
		self.lock_for_data = Lock()  # Lock for protecting access to `self.sensor_data`.
		self.lock_for_location = Lock() # Lock for protecting access to a specific location's data across devices.
		self.number_of_devices = 0   # Total count of devices in the simulation.
		self.reusable_bar = Event()  # Event used for initial barrier setup synchronization.
		self.bar = None              # Reference to the shared ReusableBarrierCond for inter-device synchronization.

	def __str__(self):
		"""
		@brief Returns a string representation of the Device.
		@return str: A string in the format "Device %d" % device_id.
		"""
		return "Device %d" % self.device_id

	def setup_devices(self, devices):
		"""
		@brief Sets up the list of all devices and the total number of devices in the simulation.
		@details This method is called by the supervisor to provide each device with references
		to all other devices in the system, which is essential for coordination and communication.
		@param devices (list): A list of all Device objects in the simulation.
		@block_logic Initializes the device's awareness of its peers in the network.
		@pre_condition `devices` is a list of `Device` instances.
		@invariant `self.list_of_devices` contains all device references, and `self.number_of_devices`
		           is accurately set to the total count of devices.
		"""
		self.list_of_devices = devices       # Store the list of all devices.
		self.number_of_devices = len(devices) # Store the total count of devices.

	def assign_script(self, script, location):
		"""
		@brief Assigns a script to be executed by the device at a specific location.
		@details If a script is provided, it's appended to the device's script queue.
		The `script_received` event is set to signal the `DeviceThread` that new scripts
		are available for processing.
		@param script (object): The script object to be executed.
		@param location (str): The location associated with the script or data.
		@block_logic Manages the receipt of scripts and signals readiness for script execution.
		@pre_condition `self.scripts` is a list, `self.script_received` is an Event object.
		@invariant If `script` is not None, it's added to `self.scripts`. `script_received` is always set.
		"""
		if script is not None:
			self.scripts.append((script, location)) # Add the script and its location to the queue.
		self.script_received.set() # Signal that new scripts have been received.

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
		@details This method acquires `lock_for_data` before modifying the sensor data
		to ensure thread safety during updates to the device's local sensor data.
		@param location (str): The location whose data is to be updated.
		@param data (object): The new data value for the specified location.
		@block_logic Safely updates the internal sensor data using a lock to prevent race conditions.
		@pre_condition `self.sensor_data` is a dictionary, and `self.lock_for_data` is an initialized Lock object.
		@invariant If `location` is a key in `self.sensor_data`, its value is updated under lock protection.
		"""
		if location in self.sensor_data:
			self.lock_for_data.acquire() # Acquire the lock to protect `self.sensor_data`.
			self.sensor_data[location] = data
			self.lock_for_data.release() # Release the lock after the update.

	def shutdown(self):
		"""
		@brief Shuts down the device by joining its associated thread.
		@details This ensures that the device's worker thread completes its execution before the program exits.
		"""
		self.thread.join()

	def set_barrier(self, barrier):
		"""
		@brief Sets the shared reusable barrier for this device and signals its readiness.
		@details This method is called by the initializing device (device 0) to distribute
		the created `ReusableBarrierCond` instance to all other devices. The `reusable_bar`
		event is set to unblock devices waiting for the barrier to be assigned.
		@param barrier (ReusableBarrierCond): The shared barrier instance.
		@block_logic Distributes the shared barrier instance among devices.
		@pre_condition `barrier` is an initialized `ReusableBarrierCond` object.
		@invariant `self.bar` refers to the shared barrier, and `self.reusable_bar` is set.
		"""
		self.reusable_bar.set() # Signal that the barrier has been set.
		self.bar = barrier       # Store the reference to the shared barrier.


class DeviceThread(Thread):
	"""
	@brief The main worker thread for a Device instance.
	@details This thread orchestrates the device's operational cycle, including
	synchronization via a shared barrier, script execution, and data management.
	It handles the initial setup of the shared barrier if it's the first device,
	waits for script assignments, and then processes them.
	@architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
	ensuring proper coordination and data consistency across the distributed system.
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
		it retrieves neighbor information, manages the initialization and synchronization
		with the shared `ReusableBarrierCond`, waits for script assignments, and then executes
		those scripts. It collects data from neighbors and its own sensors, runs scripts, and
		propagates results back to neighbors and itself under the protection of locks.
		The loop terminates when the supervisor signals the end of the simulation.
		@block_logic Orchestrates the device's operational cycle, handling timepoint progression,
		script execution, and inter-device synchronization.
		@pre_condition `self.device` is an initialized Device object with access to `supervisor`,
		               `script_received` event, and `list_of_devices`.
		@invariant The thread progresses through timepoints, processes scripts, and ensures global synchronization.
		"""
		ok = 0 # Flag to ensure barrier initialization happens once.

		# Block Logic: Main simulation loop for the device.
		# Invariant: The loop continues as long as the supervisor provides neighbors.
		while True:
			# Functional Utility: Get information about neighboring devices from the supervisor.
			neighbours = self.device.supervisor.get_neighbours()
			
			# Block Logic: Check if the simulation should terminate.
			# Pre-condition: `neighbours` list indicates the current state of the network.
			# Invariant: The loop terminates if no neighbors are returned by the supervisor.
			if neighbours is None:
				break

			# Block Logic: Initialize the shared barrier if this is the first device (device_id == 0).
			# This ensures the barrier is created only once and then distributed to all devices.
			# Invariant: `self.bar` is an initialized `ReusableBarrierCond` for all devices after this block.
			if ok == 0 and self.device.device_id == 0: 
				# Create the shared barrier with the total number of devices.
				self.bar = ReusableBarrierCond(self.device.number_of_devices)
				# Distribute the created barrier to all devices in the system.
				for dev in self.device.list_of_devices:
					dev.set_barrier(self.bar)
				ok += 1 # Set flag to prevent re-initialization.
			
			# Block Logic: If this device is not device 0 and hasn't received the barrier yet, wait for it.
			# Pre-condition: `self.device.reusable_bar` event is set by device 0 when the barrier is ready.
			# Invariant: `self.device.bar` will be assigned before proceeding.
			if ok == 0:
				self.device.reusable_bar.wait() # Wait until the barrier is set by device 0.

			# Block Logic: Synchronize all device threads before proceeding to the script execution phase.
			# Invariant: All active device threads will reach this barrier before any proceeds.
			self.device.bar.wait()

			# Block Logic: Wait until script assignments for the current timepoint are complete.
			# Pre-condition: `self.device.script_received` is an Event object.
			# Invariant: The thread proceeds only after the supervisor signals completion of script assignment.
			self.device.script_received.wait()
            # Functional Utility: Reset the event for the next timepoint's script assignment.
			self.device.script_received.clear() # Clear the event for the next timepoint.


			# Block Logic: Iterate through all assigned scripts for the current timepoint and execute them.
			# Pre-condition: `self.device.scripts` contains tuples of (script, location).
			# Invariant: Each script is run with collected data and results are propagated to neighbors and itself.
			for (script, location) in self.device.scripts:
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

					# Block Logic: Acquire `lock_for_location` to ensure atomic updates to shared location data.
					# Invariant: Shared location data is updated safely across devices.
					self.device.lock_for_location.acquire()
					# Block Logic: Propagate the script's result to neighboring devices.
					# Invariant: All neighbors receive the updated data for the given location.
					for device in neighbours:
						device.set_data(location, result)
					
					# Functional Utility: Update the current device's own data with the script's result.
					self.device.set_data(location, result)
					self.device.lock_for_location.release() # Release the lock after updates.

            # Functional Utility: Clear the scripts list for the next timepoint.
			self.device.scripts = [] # Reset scripts list for the next timepoint.
            
			# Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
			# Invariant: All DeviceThread instances will reach this barrier before any proceeds to the next timepoint.
			self.device.bar.wait()
