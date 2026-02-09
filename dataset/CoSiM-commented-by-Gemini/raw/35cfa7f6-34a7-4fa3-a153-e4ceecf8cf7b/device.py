"""
A device simulation framework with a master-worker setup and flawed synchronization.

This module implements a device simulation where one device (ID 0) is responsible
for creating and distributing shared resources like locks and a barrier. The main
thread logic contains several synchronization flaws, including inconsistent locking
on data access and an incorrect use of threading Events that causes the system
to deadlock after the first time step.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
import time


class Device(object):
	"""
	Represents a single device, participating in a master-coordinated setup.
	"""

	def __init__(self, device_id, sensor_data, supervisor):
		"""
		Initializes a Device instance.

		Args:
			device_id (int): A unique identifier for the device.
			sensor_data (dict): The device's internal sensor data.
			supervisor: The central supervisor managing the device network.
		"""
		self.device_id = device_id
		self.sensor_data = sensor_data
		self.supervisor = supervisor
		self.script_received = Event()
		self.cb = Event() # A callback event to signal setup completion.
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
		"""Returns the string representation of the device."""
		return "Device %d" % self.device_id


	def setup_devices(self, devices):
		"""
		Performs initial data gathering required by the master device (ID 0)
		to configure the simulation's shared resources.
		"""
		# Gathers data about the maximum location value across all devices.
		for x in devices:
			for val in x.sensor_data:
				if val > self.max:
					self.max = val
		
		self.dev_l = devices
		self.no_dev = len(devices)
	
	def get_max_dev(self):
		"""Returns the maximum location value observed during setup."""
		return self.max
	
	def get_no_dev(self):
		"""Returns the total number of devices observed during setup."""
		return self.no_dev

	def assign_script(self, script, location):
		"""
		Receives a script and signals the device thread that work is available.
		
		Note: This sets the `script_received` event on every assignment, not just
		at the end of a timepoint's assignments.
		"""
		if script is not None:
			self.scripts.append((script, location))
			self.script_received.set()
		else:
			self.script_received.set()

	def get_data(self, location):
		"""
		Retrieves sensor data. This read operation is not thread-safe against
		concurrent writes in `set_data`.
		"""
		return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		"""
		Updates sensor data. This write operation is protected by a lock.
		"""
		if location in self.sensor_data:
			self.data_lock.acquire()
			self.sensor_data[location] = data
			self.data_lock.release()

	def set_b(self, barr,locat):
		"""
		Callback method used by the master device to provide this device with
		shared resources and signal that setup is complete.
		"""
		self.bar = barr
		self.cb.set()
		self.location_lock = locat

	def shutdown(self):
		"""Waits for the main device thread to complete."""
		self.thread.join()


class DeviceThread(Thread):
	"""
	The main control thread for a device, containing complex setup and
	synchronization logic that results in a deadlock.
	"""

	def __init__(self, device):
		"""Initializes the DeviceThread."""
		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device

	def run(self):
		"""The main execution loop."""
		i = 0
		
		while True:
			neighbours = self.device.supervisor.get_neighbours()
			if neighbours is None:
				break

			# Block Logic: A one-time setup routine performed by the master device (ID 0).
			# It creates location-specific locks and a shared barrier, then distributes
			# them to all other devices.
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
				i=2 # Prevents this block from running again.
				for dev in self.device.dev_l:
					dev.set_b(self.bar,location_lock)
			else:
				# All non-master devices wait here until the master calls `set_b`.
				self.device.cb.wait()
				
			# All threads synchronize after setup.
			self.device.bar.wait()
		
			# Wait for a script assignment to arrive.
			self.device.script_received.wait()
			
			# Block Logic: Process assigned scripts using location-specific locks.
			for (script, location) in self.device.scripts:
				self.device.location_lock[location].acquire()
			
				script_data = []
				
				# Aggregate data from neighbours and self.
				for device in neighbours:
					data = device.get_data(location)
					if data is not None:
						script_data.append(data)
				
				data = self.device.get_data(location)
				if data is not None:
					script_data.append(data)
			
				# Invariant: Script runs only if there is data to process.
				if script_data != []:
					result = script.run(script_data)
			
					# Broadcast the result to all participants.
					for device in neighbours:
						device.set_data(location, result)
					
					self.device.set_data(location, result)
					
					self.device.location_lock[location].release()

			# All threads synchronize after processing scripts.
			self.device.bar.wait()
			
			# CRITICAL: The thread sets and immediately waits on the same event,
			# causing a permanent self-deadlock after the first timepoint.
			self.device.timepoint_done.set()
			self.device.timepoint_done.wait()