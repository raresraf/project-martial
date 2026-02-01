"""
This module implements a distributed device simulation framework using Python threading
and a custom barrier synchronization mechanism.

Algorithm:
- Device: Represents a simulated entity managing sensor data and script execution.
- DeviceThread: Dedicated thread for each device, responsible for orchestrating script
  execution and synchronizing with other devices using a shared barrier.
- Barrier: A synchronization primitive (imported from `Barrier`) that ensures all
  participating threads reach a specific point before any can proceed.
"""

from threading import Event, Thread, Lock
from Barrier import *


class Device(object):
	"""
	Represents a simulated device in a distributed environment.

	Each device has a unique ID, manages its sensor data, and interacts
	with a supervisor. It is capable of receiving and executing scripts
	on its data, coordinating with other devices using a shared barrier.

	Attributes:
		device_id (int): A unique identifier for the device.
		sensor_data (dict): A dictionary storing sensor readings,
							where keys represent locations.
		supervisor (Supervisor): A reference to the central supervisor managing devices.
		script_received (threading.Event): Event to signal when script assignments are complete.
		scripts (list): A list to store assigned scripts (script, location) tuples.
		num_devices (int): The total number of devices in the simulation, used for barrier initialization.
		thread (DeviceThread): The dedicated thread for this device's operations.
	"""

	def __init__(self, device_id, sensor_data, supervisor):
		"""
		Initializes a new Device instance.

		Args:
			device_id (int): The unique identifier for this device.
			sensor_data (dict): Initial sensor data for the device.
			supervisor (Supervisor): The supervisor object responsible for managing devices.
		"""
		self.device_id = device_id
		self.sensor_data = sensor_data
		self.supervisor = supervisor

		# Event for signaling when script assignments are complete.
		self.script_received = Event()
		# List to hold (script, location) tuples assigned to this device.
		self.scripts = []
		# Will be set during setup_devices for barrier initialization.
		self.num_devices = 0
		# Create and start a dedicated thread for this device's operations.
		self.thread = DeviceThread(self)
		self.thread.start()

	def __str__(self):
		"""
		Returns a string representation of the Device.

		Returns:
			str: A string in the format "Device <device_id>".
		"""
		return "Device %d" % self.device_id

	def setup_devices(self, devices):
		"""
		Initializes the shared synchronization barrier across all devices.
		This method is typically called once by a designated device (e.g., devices[0])
		to create the barrier, which is then shared among all participating devices.

		Pre-condition: This method should be called by a central entity (e.g., supervisor)
					   after all devices are instantiated.
		Invariant: The `num_devices` attribute is set, and a shared `Barrier` object
				   is created (if not already existing) and assigned to the `thread.barrier`
				   attribute of all devices.

		Args:
			devices (list): A list of all Device objects in the simulation.
		"""
		self.num_devices = len(devices)
		# Block Logic: Only the first device in the list creates the shared barrier.
		if devices[0].thread.barrier is None:
			devices[0].thread.barrier = Barrier(self.num_devices, str(devices[0].device_id))
			# Propagate the created barrier to all other device threads.
			for i in range (1, len(devices)):
				devices[i].thread.barrier = devices[0].thread.barrier

	def assign_script(self, script, location):
		"""
		Assigns a script to be executed on data at a specific location for this device.
		If `script` is None, it signals that script assignments for the current timepoint are complete.

		Args:
			script (Script or None): The script object to execute, or None to signal completion.
			location (int): The data location where the script should be applied.
		"""
		if script is not None:
			# Acquire a lock to ensure thread-safe modification of the scripts list.
			with self.thread.lock:
				self.scripts.append((script, location))
		else:
			# Signal that script assignment for the current timepoint is complete.
			self.script_received.set()

	def get_data(self, location):
		"""
		Retrieves sensor data for a specific location.
		Acquires a lock to ensure thread-safe access to sensor data.

		Args:
			location (int): The location from which to retrieve data.

		Returns:
			any: The sensor data at the specified location, or None if the location
				 does not exist in the device's sensor_data.
		"""
		# Acquire a lock to ensure thread-safe access to sensor data.
		with self.thread.lock:
			return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		"""
		Sets sensor data for a specific location.
		Acquires a lock to ensure thread-safe modification of sensor data.
		This implementation updates the data only if the new data is greater than the current.

		Args:
			location (int): The location at which to set data.
			data (any): The new data value to set.
		"""
		# Acquire a lock to ensure thread-safe modification of sensor data.
		with self.thread.lock:
			if location in self.sensor_data:
				# Functional Utility: Update data only if the new value is greater.
				# This could be a specific domain logic for sensor updates (e.g., maximum reading).
				if self.sensor_data[location] < data:
					self.sensor_data[location] = data

	def shutdown(self):
		"""
		Joins the device's dedicated thread, effectively waiting for it to complete
		its execution before the program exits.
		"""
		self.thread.join()


class DeviceThread(Thread):
	"""
	The dedicated thread for a `Device` object, responsible for orchestrating
	script execution and synchronizing with other device threads.

	Attributes:
		device (Device): The Device object associated with this thread.
		barrier (Barrier): A reference to the shared synchronization barrier.
		lock (threading.Lock): A lock to protect access to the device's shared resources
							   (like `sensor_data` and `scripts`).
	"""

	def __init__(self, device):
		"""
		Initializes a new DeviceThread instance.

		Args:
			device (Device): The Device object that this thread will manage.
		"""
		# Initialize the base Thread class with a descriptive name.
		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device
		# The barrier is set by the `Device.setup_devices` method.
		self.barrier = None
		# A lock to protect the device's data and script list.
		self.lock = Lock()

	def run(self):
		"""
		The main execution loop for the DeviceThread.

		Invariant: The loop continues until the supervisor signals termination
				   by returning None for neighbors. Within each iteration, the
				   thread retrieves neighbor information, waits for new script
				   assignments, executes the scripts, and then synchronizes
				   with other devices via the shared barrier.
		"""
		while True:
			# Retrieve information about neighboring devices from the supervisor.
			neighbours = self.device.supervisor.get_neighbours()
			# Block Logic: If no neighbors are returned (e.g., simulation termination signal), break the loop.
			if neighbours is None:
				break

			# Wait for the `script_received` event, which signals that script assignments
			# for the current timepoint are complete and ready for processing.
			self.device.script_received.wait()
			# Clear the `script_received` event, resetting it for the next timepoint.
			self.device.script_received.clear()

			# Block Logic: Iterate through each assigned script and execute it.
			for (script, location) in self.device.scripts:
				script_data = []
				# Block Logic: Collect data from neighboring devices for the current location.
				for device in neighbours:
					data = device.get_data(location)
					if data is not None:
						script_data.append(data)
				# Collect data from the current device for the current location.
				data = self.device.get_data(location)
				if data is not None:
					script_data.append(data)

				# Block Logic: If there is data collected, execute the script and update results.
				if script_data != []:
					# Execute the script with the collected data.
					result = script.run(script_data)

					# Update the sensor data in neighboring devices.
					for device in neighbours:
						device.set_data(location, result)
					# Update the sensor data in the current device.
					self.device.set_data(location, result)

			# Functional Utility: Clear the scripts list after processing for the current timepoint.
			self.device.scripts = []

			# Synchronize with other DeviceThreads using the shared barrier, ensuring
			# all devices complete their timepoint processing before proceeding.
			self.barrier.wait()
