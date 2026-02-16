"""
@file device.py
@brief Implements a simulated device for distributed computation.

This file defines the `Device` class, representing a computational node
in a distributed system, and `DeviceThread`, which manages the execution
logic for each device. Devices can hold sensor data, process scripts
collaboratively with neighbors, and synchronize using barriers.
It's designed to simulate a distributed sensing and processing network.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
import time


class Device(object):
	"""
	@class Device
	@brief Represents a simulated computational device in a distributed system.

	Each `Device` instance can hold sensor data, execute assigned scripts,
	and interact with a `Supervisor` to coordinate with other devices.
	It manages its own thread of execution and uses synchronization primitives
	for coordinated data access and script execution.

	@attribute device_id (int): A unique identifier for the device.
	@attribute sensor_data (dict): A dictionary holding sensor data, keyed by location.
	@attribute supervisor (Supervisor): A reference to the central supervisor managing all devices.
	@attribute script_received (Event): Synchronization event, set when a new script is assigned.
	@attribute cb (Event): Synchronization event, set when the barrier and location locks are set up by device 0.
	@attribute scripts (list): A list of (script, location) tuples to be executed.
	@attribute timepoint_done (Event): Synchronization event, set when a timepoint's script execution is complete.
	@attribute thread (DeviceThread): The dedicated thread of execution for this device.
	@attribute data_lock (Lock): A lock to protect `sensor_data` during concurrent access.
	@attribute dev_l (list): A list of all `Device` objects in the system (only populated in device 0).
	@attribute no_dev (int): The total number of devices in the system (only populated in device 0).
	@attribute max (int): The maximum sensor data location across all devices (only populated in device 0).
	@attribute location_lock (list): A list of locks, one for each possible sensor data location, to protect access to data at that location.
	"""

	def __init__(self, device_id, sensor_data, supervisor):
		"""
		@brief Initializes a new Device instance.

		Sets up the device's unique ID, sensor data, supervisor reference,
		and various synchronization primitives and data structures required
		for its operation in the distributed system. It also starts the
		device's dedicated thread.

		@param device_id (int): A unique identifier for this device.
		@param sensor_data (dict): Initial sensor data for this device, typically a dictionary mapping locations to data values.
		@param supervisor (Supervisor): The supervisor object responsible for managing this device.
		"""
		self.device_id = device_id
		self.sensor_data = sensor_data
		self.supervisor = supervisor
		self.script_received = Event()
		self.cb = Event()
		self.scripts = []
		self.timepoint_done = Event()
		self.thread = DeviceThread(self)
		self.thread.start()
		self.data_lock = Lock()
		self.dev_l = []
		self.no_dev = -1
		self.max = 0
		self.location_lock = []


	def __str__(self):
		"""
		@brief Returns a string representation of the Device.

		This method provides a human-readable string that identifies the device
		by its `device_id`.

		@return (str): A string in the format "Device [device_id]".
		"""
		return "Device %d" % self.device_id


	def setup_devices(self, devices):
		"""
		@brief Sets up device-specific and global device information.

		This method is primarily called by the supervisor or device 0 to
		initialize global device parameters suchs as the maximum sensor data
		location (`self.max`) and the total number of devices (`self.no_dev`).
		It also stores a reference to all devices in `self.dev_l`.

		@param devices (list): A list of all `Device` objects in the simulated system.
		"""
		
		self.max = 0
		
		
		for x in devices:
			for val in x.sensor_data:
				
				if val > self.max:
					self.max = val
		
		self.dev_l = devices
		self.no_dev = len(devices)
	
	def get_max_dev(self):
		"""
		@brief Returns the maximum sensor data location index observed across all devices.

		This value is typically initialized by device 0 and represents the upper
		bound for sensor data locations.

		@return (int): The maximum sensor data location index.
		"""
		return self.max
	
	def get_no_dev(self):
		"""
		@brief Returns the total number of devices in the simulated system.

		This value is typically initialized by device 0 and reflects the
		total count of `Device` objects managed by the supervisor.

		@return (int): The total number of devices.
		"""
		return self.no_dev

	def assign_script(self, script, location):
		"""
		@brief Assigns a script to be executed at a specific data location.

		This method adds a new (script, location) tuple to the device's
		`scripts` queue and sets the `script_received` event to signal
		the device's thread that there are new scripts to process.
		If `script` is None, it still sets the event, effectively waking
		the thread without adding a script.

		@param script (object): The script object to be executed.
		@param location (int): The sensor data location that the script pertains to.
		"""
		
		if script is not None:
			self.scripts.append((script, location))
			self.script_received.set()
		else:
			self.script_received.set()

	def get_data(self, location):
		"""
		@brief Retrieves sensor data for a specific location.

		This method provides access to the sensor data stored on the device
		for a given `location`.

		@param location (int): The specific location for which to retrieve data.
		@return (any): The data at the specified location, or `None` if the location is not found.
		"""
		return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		"""
		@brief Sets or updates sensor data for a specific location.

		This method updates the `sensor_data` at a given `location` with the
		new `data`. It uses a lock to ensure thread-safe access to the `sensor_data`
		dictionary.

		@param location (int): The specific location for which to set data.
		@param data (any): The new data value to be stored at the location.
		"""
		if location in self.sensor_data:
			self.data_lock.acquire()
			self.sensor_data[location] = data
			self.data_lock.release()

	def set_b(self, barr,locat):
		"""
		@brief Sets the barrier and location-specific locks for the device.

		This method is used to assign the `ReusableBarrierSem` instance and
		the list of `location_lock`s to the device. It then sets the `cb`
		event to signal that these resources have been initialized.

		@param barr (ReusableBarrierSem): The barrier synchronization object.
		@param locat (list): A list of `threading.Lock` objects for each data location.
		"""
		self.bar = barr
		self.cb.set()
		self.location_lock = locat

	def shutdown(self):
		"""
		@brief Shuts down the device's thread.

		This method waits for the device's associated `DeviceThread` to
		complete its execution, ensuring a clean shutdown of the device.
		"""
		self.thread.join()


class DeviceThread(Thread):
	"""
	@class DeviceThread
	@brief Manages the asynchronous execution of scripts on a `Device`.

	This class extends `threading.Thread` to provide a dedicated thread
	for each `Device` instance. It handles the continuous loop of waiting
	for scripts, executing them, and synchronizing with other devices
	using barriers and locks.

	@attribute device (Device): A reference to the `Device` object this thread is managing.
	"""

	def __init__(self, device):
		"""
		@brief Initializes a new `DeviceThread` instance.

		Sets up the thread with a descriptive name and stores a reference
		to the `Device` object it will manage.

		@param device (Device): The `Device` instance that this thread will operate on.
		"""
		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device


	def run(self):
		"""
		@brief The main execution loop for the device thread.

		This method continuously executes in a loop, simulating the device's
		operation over timepoints. It handles:
		- Initializing barriers and locks (by device 0).
		- Waiting for synchronization signals from the supervisor or other devices.
		- Processing assigned scripts at specific data locations.
		- Collecting data from neighbors and its own device for script execution.
		- Updating sensor data with script results.
		- Using barriers to synchronize with other devices at each timepoint.
		"""
		i = 0
		
	


		
		while True:
		
		
			
			neighbours = self.device.supervisor.get_neighbours()
			if neighbours is None:
				break
		




		
		
			if self.device.device_id == 0 and i == 0:
				max = 0
				location_lock = []
				for dev in self.device.dev_l:
					if dev.get_max_dev() > max:
						max = dev.get_max_dev()
				max = max + 1
				for i in range(max):
					location_lock.append(Lock())

				self.bar = ReusableBarrierSem(self.device.get_no_dev())
				i=2
				for dev in self.device.dev_l:
					dev.set_b(self.bar,location_lock)
		
			else:


				self.device.cb.wait()
				

			self.device.bar.wait()
		
		
			self.device.script_received.wait()
			
		
		
		
		
		

			
			for (script, location) in self.device.scripts:
				self.device.location_lock[location].acquire()
			
				script_data = []
				
				for device in neighbours:
					data = device.get_data(location)
					if data is not None:
						script_data.append(data)
				
				data = self.device.get_data(location)
				if data is not None:
					script_data.append(data)
			
			
			
			
			
			
				if script_data != []:
					
					result = script.run(script_data)
			
			
					
					for device in neighbours:
						device.set_data(location, result)
					
					self.device.set_data(location, result)
					
					self.device.location_lock[location].release()

			
		
			self.device.bar.wait()
			self.device.timepoint_done.set()
		
			
			self.device.timepoint_done.wait()