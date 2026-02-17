"""
@4d83f913-7c7a-4473-beac-f0ea3b74b56f/device.py
@brief Implements a simulated device for a distributed sensor network, with dynamic lock management and barrier synchronization.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` for operational logic and uses a `ReusableBarrierSem`
for global time-step synchronization. A unique aspect is the dynamic creation and
sharing of `Lock` objects (`location_lock`) for each data `location` across all devices,
managed by the device with `device_id == 0`.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
import time


class Device(object):
	"""
	@brief Represents a single device in the distributed system simulation.
	Manages its sensor data, assigned scripts, and coordinates its operation
	through a dedicated thread, a shared barrier, and dynamically managed
	location-specific locks.
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
		self.cb = Event() # Event used to signal that the barrier and location_lock have been set.
		self.scripts = [] # List to store assigned scripts.


		self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
		self.thread = DeviceThread(self)
		self.thread.start()
		self.data_lock = Lock() # Lock to protect access to the device's own sensor_data.
		self.dev_l = [] # Will store a reference to all devices in the simulation.
		self.no_dev = -1 # Stores the total number of devices.
		self.max = 0 # Stores the maximum location index found across all devices.
		self.location_lock = [] # Will store a list of locks, one for each unique data location.


	def __str__(self):
		"""
		@brief Provides a string representation of the device.
		@return A string in the format "Device <device_id>".
		"""
		return "Device %d" % self.device_id


	def setup_devices(self, devices):
		"""
		@brief Sets up the list of all devices and determines the maximum location index.
		This method is typically called once during system setup.
		@param devices: A list of all Device instances in the simulation.
		Precondition: This method is called early in the simulation setup.
		"""
		# Block Logic: Iterates through all devices to find the maximum location index.
		# This is used to pre-allocate the `location_lock` list.
		for x in devices:
			for val in x.sensor_data:
				if val > self.max:
					self.max = val
		
		self.dev_l = devices # Stores a reference to all devices.
		self.no_dev = len(devices) # Stores the total count of devices.
	
	def get_max_dev(self):
		"""
		@brief Returns the maximum sensor data location index observed across all devices.
		@return The maximum location index (integer).
		"""
		return self.max
	
	def get_no_dev(self):
		"""
		@brief Returns the total number of devices in the simulation.
		@return The count of devices (integer).
		"""
		return self.no_dev

	def assign_script(self, script, location):
		"""
		@brief Assigns a script to the device for execution at a specific data `location`.
		Signals that a script has been received.
		@param script: The script object to assign.
		@param location: The data location relevant to the script.
		"""
		if script is not None:
			self.scripts.append((script, location))
			self.script_received.set()
		else:
			# Block Logic: Signals that a script was processed even if it was None (e.g., end of scripts for timepoint).
			self.script_received.set()

	def get_data(self, location):
		"""
		@brief Retrieves sensor data for a given location.
		@param location: The key identifying the sensor data.
		@return The data associated with the location, or `None` if the location is not found.
		"""
		return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		"""
		@brief Sets or updates sensor data for a specified location, protected by a data lock.
		@param location: The key for the sensor data to be modified.
		@param data: The new data value to store.
		Precondition: `location` must be a valid key in `self.sensor_data`.
		"""
		if location in self.sensor_data:
			# Block Logic: Acquires the device's data lock to ensure thread-safe modification of sensor_data.
			self.data_lock.acquire()
			self.sensor_data[location] = data
			self.data_lock.release()

	def set_b(self, barr, locat):
		"""
		@brief Sets the shared barrier and location-specific locks for this device.
		Signals its `cb` event once these shared resources are set.
		@param barr: The shared `ReusableBarrierSem` instance.
		@param locat: The shared list of `Lock` objects for location-specific data access.
		"""
		self.bar = barr
		self.cb.set() # Signal that shared barrier and location locks are ready.
		self.location_lock = locat

	def shutdown(self):
		"""
		@brief Shuts down the device's operational thread, waiting for its graceful completion.
		"""
		self.thread.join()


class DeviceThread(Thread):
	"""
	@brief The dedicated thread of execution for a `Device` instance.
	This thread manages the device's operational cycle, including fetching neighbor data,
	executing scripts, and coordinating with other device threads using a shared barrier
	and dynamically managed location-specific locks.
	Time Complexity: O(T * S * (N * D_access + D_script_run)) where T is the number of timepoints,
	S is the number of scripts per device, N is the number of neighbors, D_access is data access
	time, and D_script_run is script execution time.
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
		1. Performs initial setup of global barrier and location locks if this is device 0.
		2. Continuously fetches neighbor information from the supervisor.
		   Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
		3. Synchronizes all device threads using the shared barrier.
		4. Waits for `script_received` event to be set, indicating all scripts for the timepoint are assigned.
		5. Executes assigned scripts: for each script, it acquires the location-specific lock to get data
		   from neighbors and itself, runs the script, and then releases the lock after updating data.
		   Invariant: Data access and modification for a given location are protected by its corresponding lock.
		6. Synchronizes all device threads again using the shared barrier.
		7. Sets and immediately waits on `timepoint_done` event for specific signaling.
		"""

		i = 0 # Counter used for initial setup condition for device 0.
		
		# Block Logic: Main simulation loop, continues until supervisor signals termination.
		while True:
		
			# Block Logic: Fetches neighbor devices from the supervisor.
			neighbours = self.device.supervisor.get_neighbours()
			if neighbours is None:
				break
		
			# Block Logic: Handles the initial setup of the shared barrier and location locks.
			# Only device with `device_id == 0` performs this initialization.
			if self.device.device_id == 0 and i == 0:
				max_location_idx = 0
				location_lock_list = []
				# Block Logic: Determines the maximum location index across all devices to size the lock list.
				for dev in self.device.dev_l:
					if dev.get_max_dev() > max_location_idx:
						max_location_idx = dev.get_max_dev()
				max_location_idx = max_location_idx + 1 # Adjust for 0-based indexing.
				# Block Logic: Initializes a list of locks, one for each potential location.
				for _ in range(max_location_idx): # Using range instead of xrange for Python 3 compatibility.
					location_lock_list.append(Lock())

				# Block Logic: Initializes the shared reusable barrier.
				self.bar = ReusableBarrierSem(self.device.get_no_dev())
				i=2 # Prevents re-initialization on subsequent loops for device 0.
				# Block Logic: Distributes the initialized barrier and location locks to all devices.
				for dev in self.device.dev_l:
					dev.set_b(self.bar,location_lock_list)
		
			else:
				# Block Logic: Other devices wait for device 0 to signal that the shared resources are set.
				self.device.cb.wait()
				

			# Block Logic: Synchronizes all device threads at the start of each timepoint.
			self.device.bar.wait()
		
			# Block Logic: Waits until the device has received all scripts for the current timepoint.
			self.device.script_received.wait()
			
			# Block Logic: Processes each script assigned to the device for the current timepoint.
			# Invariant: Each script retrieves data from neighbors and itself, executes, and updates data,
			# all while holding the appropriate location-specific lock.
			for (script, location) in self.device.scripts:
				# Block Logic: Acquires the lock specific to the data location to ensure exclusive access.
				self.device.location_lock[location].acquire()
			
				script_data = []
				
				# Block Logic: Collects data from neighboring devices for the specified location.
				for device in neighbours:
					data = device.get_data(location)
					if data is not None:
						script_data.append(data)
				
				# Block Logic: Collects data from its own device for the specified location.
				data = self.device.get_data(location)
				if data is not None:
					script_data.append(data)
			
				# Block Logic: Executes the script if any data was collected.
				if script_data != []:
					
					result = script.run(script_data)
			
					# Block Logic: Updates neighboring devices with the script's result.
					for device in neighbours:
						device.set_data(location, result)
					
					# Block Logic: Updates its own device's data with the script's result.
					self.device.set_data(location, result)
					
				# Block Logic: Releases the location-specific lock after all data operations for this script are complete.
				self.device.location_lock[location].release()

			# Block Logic: Synchronizes all device threads again using the shared barrier after script execution.
			self.device.bar.wait()
			
			# Block Logic: Sets the `timepoint_done` event to signal completion of this device's timepoint processing.
			self.device.timepoint_done.set()
		
			# Block Logic: Waits on the `timepoint_done` event. This pattern (set then wait) suggests
			# a signaling mechanism where the event is cleared elsewhere, potentially by the supervisor,
			# allowing all devices to move forward simultaneously to the next timepoint.
			self.device.timepoint_done.wait()
