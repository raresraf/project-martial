"""
@file device.py
@brief Implements a device model for a distributed simulation.

This file defines a `Device` class and its associated `DeviceThread`. The
simulation employs a complex synchronization scheme where the root device (ID 0)
is responsible for lazily initializing and distributing a shared barrier and
a set of location-based locks to all other devices from within its own
execution thread.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
import time


class Device(object):
	"""
	Represents a device in a simulated network, managing sensor data,
	scripts, and complex multi-device synchronization.
	"""

	def __init__(self, device_id, sensor_data, supervisor):
		"""
		Initializes a Device instance.

		Args:
			device_id (int): A unique identifier for the device.
			sensor_data (dict): The device's sensor data, mapping locations to values.
			supervisor: The central supervisor object.
		"""
		self.device_id = device_id
		self.sensor_data = sensor_data
		self.supervisor = supervisor
		self.script_received = Event()
		self.cb = Event() # An event to signal that the barrier is configured.
		self.scripts = []


		self.timepoint_done = Event()
		self.thread = DeviceThread(self)
		self.thread.start()
		self.data_lock = Lock()
		self.dev_l = [] # List of all devices in the simulation.
		self.no_dev = -1 # Total number of devices.
		self.max = 0 # Maximum location value seen.
		self.location_lock = []


	def __str__(self):
		"""Returns a string representation of the device."""
		return "Device %d" % self.device_id


	def setup_devices(self, devices):
		"""
		Performs initial setup by recording all devices and calculating the
		maximum location value across the network.

		Args:
			devices (list): A list of all Device objects in the simulation.
		"""
		# Block Logic: Iterates through all devices to find the highest
		# location value, which determines the number of location locks needed.
		for x in devices:
			for val in x.sensor_data:
				if val > self.max:
					self.max = val
		
		self.dev_l = devices
		self.no_dev = len(devices)
	
	def get_max_dev(self):
		"""Returns the maximum location value found."""
		return self.max
	
	def get_no_dev(self):
		"""Returns the total number of devices."""
		return self.no_dev

	def assign_script(self, script, location):
		"""
		Assigns a script to be executed by the device.

		Args:
			script: The script object to execute.
			location: The location context for the script.
		"""
		if script is not None:
			self.scripts.append((script, location))
			self.script_received.set()
		else:
			# A None script can also trigger the event, signaling the device
			# to proceed with the scripts it has.
			self.script_received.set()

	def get_data(self, location):
		"""Retrieves sensor data for a given location."""
		return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		"""
		Updates sensor data for a location with thread safety.

		Args:
			location: The location to update.
			data: The new data value.
		"""
		if location in self.sensor_data:
			self.data_lock.acquire()
			self.sensor_data[location] = data
			self.data_lock.release()

	def set_b(self, barr, locat):
		"""
		Callback method used by the root device to set the shared barrier
		and location locks on other devices.

		Args:
			barr: The shared ReusableBarrierSem instance.
			locat (list): The list of shared location Lock objects.
		"""
		self.bar = barr
		self.cb.set() # Signal that the barrier is now configured.
		self.location_lock = locat

	def shutdown(self):
		"""Shuts down the device by joining its execution thread."""
		self.thread.join()


class DeviceThread(Thread):
	"""

	The main execution thread for a Device, containing complex setup and
	synchronization logic.
	"""

	def __init__(self, device):
		"""
		Initializes the DeviceThread.

		Args:
			device (Device): The parent device object.
		"""
		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device

	def run(self):
		"""
		The main simulation loop.

		The root device's thread (ID 0) performs a one-time setup of the shared
		barrier and locks, then distributes them to other devices. All devices then
		enter a synchronized loop, processing scripts at each timepoint.
		"""
		i = 0
		while True:
			neighbours = self.device.supervisor.get_neighbours()
			if neighbours is None:
				break

			# Block Logic: One-time setup performed by the root device's thread.
			if self.device.device_id == 0 and i == 0:
				max = 0
				location_lock = []
				# Determine the total number of location locks needed.
				for dev in self.device.dev_l:
					if dev.get_max_dev() > max:
						max = dev.get_max_dev()
				max = max + 1
				for i in range(max):
					location_lock.append(Lock())

				# Create the shared barrier.
				self.bar = ReusableBarrierSem(self.device.get_no_dev())
				i=2 # Prevents re-entry into this setup block.
				# Distribute the barrier and locks to all other devices.
				for dev in self.device.dev_l:
					dev.set_b(self.bar,location_lock)
			else:
				# Non-root devices wait here until the root has finished setup.
				self.device.cb.wait()

			# --- Start of synchronized timepoint ---
			# First barrier: ensures all devices are ready for the timepoint.
			self.device.bar.wait()
		
			self.device.script_received.wait()

			# Block Logic: Process all assigned scripts.
			for (script, location) in self.device.scripts:
				# Pre-condition: Acquire lock for the specific location.
				self.device.location_lock[location].acquire()
			
				script_data = []
				# Gather data from neighbors and self.
				for device in neighbours:
					data = device.get_data(location)
					if data is not None:
						script_data.append(data)
				
				data = self.device.get_data(location)
				if data is not None:
					script_data.append(data)
			
				# Invariant: Data is gathered, and script is ready to run.
				if script_data != []:
					result = script.run(script_data)
			
					# Propagate result to all relevant devices.
					for device in neighbours:
						device.set_data(location, result)
					
					self.device.set_data(location, result)
					
					self.device.location_lock[location].release()

			# Second barrier: ensures all script processing is complete across
			# all devices before signaling the timepoint is done.
			self.device.bar.wait()
			self.device.timepoint_done.set()
		
			# This wait seems redundant but may be for synchronization with the supervisor.
			self.device.timepoint_done.wait()