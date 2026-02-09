

"""
This module defines a distributed device simulation framework with a two-phase
barrier synchronization and locking for data consistency.

It features a `Device` class representing a network node and a `DeviceThread` that
manages the device's execution lifecycle. The framework relies on an external
`ReusableBarrierCond` for synchronization and uses separate locks for data and
location access.
"""

from barrier import ReusableBarrierCond
from threading import Lock, Event, Thread

class Device(object):
	"""
	Represents a device in a distributed network that can process sensor data.

	Each device has a main thread that executes scripts and synchronizes with
	other devices using a shared barrier. It uses locks to manage concurrent
	access to sensor data and locations.
	"""

	def __init__(self, device_id, sensor_data, supervisor):
		"""
		Initializes a Device instance.

		Args:
			device_id (int): A unique identifier for the device.
			sensor_data (dict): A dictionary of sensor data, keyed by location.
			supervisor (Supervisor): A supervisor object that manages the network.
		"""
		
		self.device_id = device_id
		self.sensor_data = sensor_data
		self.supervisor = supervisor
		self.script_received = Event()
		self.scripts = []
		self.timepoint_done = Event()
		self.thread = DeviceThread(self)
		self.thread.start()
		self.list_of_devices = []
		self.lock_for_data = Lock()
		self.lock_for_location = Lock()
		self.number_of_devices = 0
		self.reusable_bar = Event()

	def __str__(self):
		
		return "Device %d" % self.device_id

	def setup_devices(self, devices):
		"""
		Initializes the list of devices in the network.

		Args:
			devices (list): A list of all devices in the network.
		"""

		
		self.list_of_devices = devices
		self.number_of_devices = len(devices)

	def assign_script(self, script, location):
		"""
		Assigns a script to the device.

		Args:
			script (Script): The script object to execute.
			location (int): The location identifier associated with the script's data.
		"""
		
		if script is not None:
			self.scripts.append((script, location))
		self.script_received.set()

	def get_data(self, location):
		"""
		Retrieves sensor data for a given location.

		Args:
			location (int): The location to retrieve data for.

		Returns:
			The sensor data, or None if the location is not found.
		"""
		
		return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		"""
		Updates sensor data for a given location, protected by a lock.

		Args:
			location (int): The location to update data for.
			data: The new data value.
		"""
		
		
		if location in self.sensor_data:
			self.lock_for_data.acquire()
			self.sensor_data[location] = data
			self.lock_for_data.release()

	def shutdown(self):
		"""Shuts down the device's thread."""
		
		self.thread.join()

	def set_barrier(self, barrier):
		"""
		Sets the shared barrier for this device.

		Args:
			barrier (ReusableBarrierCond): The shared barrier instance.
		"""
		self.reusable_bar.set()
		self.bar = barrier


class DeviceThread(Thread):
	"""
	The main execution thread for a Device instance.
	"""

	def __init__(self, device):
		"""
		Initializes the device thread.

		Args:
			device (Device): The device that this thread will run.
		"""
		
		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device

	def run(self):
		"""
		The main execution loop for the device.

		This loop initializes and distributes a shared barrier, then enters a
		synchronization and script execution cycle. It waits at the barrier,
		waits for scripts, executes them with location-based locking, and then
		waits at the barrier again to complete the timepoint.
		"""
		ok = 0

		
		while True:
			
			neighbours = self.device.supervisor.get_neighbours()
			if neighbours is None:
				break

			
			if ok == 0 and self.device.device_id == 0: 
				self.bar = ReusableBarrierCond(self.device.number_of_devices)
				for dev in self.device.list_of_devices:
					dev.set_barrier(self.bar)
				ok += 1
			
			if ok == 0:
				


				self.device.reusable_bar.wait() 

			
			self.device.bar.wait()

			
			self.device.script_received.wait()

			
			for (script, location) in self.device.scripts:
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

					
					self.device.lock_for_location.acquire()
					for device in neighbours:
						device.set_data(location, result)
					
					self.device.set_data(location, result)
					self.device.lock_for_location.release()

			
			self.device.bar.wait()
