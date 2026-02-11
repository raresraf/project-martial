
"""
This module implements a device simulation for a concurrent system that
contains a convoluted and critically flawed synchronization mechanism.

The setup of the shared barrier is unusually placed inside the main execution
loop of the device threads. More importantly, the locking strategy is incorrect,
as threads acquire a local lock that offers no protection when modifying data
on neighboring devices, leading to race conditions.
"""

from barrier import ReusableBarrierCond
from threading import Lock, Event, Thread

class Device(object):
	"""
	Represents a single device in the simulation.

	Each device holds its own local locks and an event to signal barrier
	initialization, but the locking strategy is not sufficient to ensure
	thread safety in a multi-device context.
	"""

	def __init__(self, device_id, sensor_data, supervisor):
		
		self.device_id = device_id
		self.sensor_data = sensor_data
		self.supervisor = supervisor
		self.script_received = Event()
		self.scripts = []
		self.timepoint_done = Event()
		self.thread = DeviceThread(self)
		self.thread.start()
		self.list_of_devices = []
		# Each device has its own independent locks. They are not shared.
		self.lock_for_data = Lock()
		self.lock_for_location = Lock()
		self.number_of_devices = 0
		# An event used to signal that the shared barrier has been set.
		self.reusable_bar = Event()

	def __str__(self):
		"""Returns a string representation of the device."""
		return "Device %d" % self.device_id

	def setup_devices(self, devices):
		"""Stores the list of all devices for later use by the leader thread."""

		
		self.list_of_devices = devices
		self.number_of_devices = len(devices)

	def assign_script(self, script, location):
		"""Assigns a script to be executed by the device."""
		if script is not None:
			self.scripts.append((script, location))
		self.script_received.set()

	def get_data(self, location):
		"""Retrieves sensor data from a specific location."""
		return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		"""
		Sets sensor data at a specific location, using a device-local lock.
		
		This lock only prevents concurrent writes to this device's own data from
		multiple threads that have a reference to this specific device instance.
		It does not protect against writes from other devices' threads.
		"""
		
		if location in self.sensor_data:
			self.lock_for_data.acquire()
			self.sensor_data[location] = data
			self.lock_for_data.release()

	def shutdown(self):
		"""Waits for the device's main thread to terminate."""
		self.thread.join()

	def set_barrier(self, barrier):
		"""Callback for the leader thread to set the shared barrier."""
		self.reusable_bar.set()
		self.bar = barrier


class DeviceThread(Thread):
	"""The main control thread for a device's lifecycle."""

	def __init__(self, device):
		


		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device

	def run(self):
		"""
		The main execution loop, containing flawed synchronization logic.
		"""
		ok = 0

		
		while True:
			
			neighbours = self.device.supervisor.get_neighbours()
			if neighbours is None:
				break

			
			# The leader thread (device 0) creates and distributes the barrier
			# on the first loop iteration.
			if ok == 0 and self.device.device_id == 0: 
				self.bar = ReusableBarrierCond(self.device.number_of_devices)
				for dev in self.device.list_of_devices:
					dev.set_barrier(self.bar)
				ok += 1
			
			# Follower threads wait here until the leader has set the barrier.
			if ok == 0:
				
				self.device.reusable_bar.wait() 

			
			# First synchronization point: all threads wait at the barrier.
			self.device.bar.wait()

			
			# Wait for scripts to be assigned for the current timepoint.
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

					
					# --- CRITICAL FLAW ---
					# This thread acquires its OWN local lock. This lock provides NO
					# protection for the data on the 'neighbours' devices. When this
					# thread calls `device.set_data()`, the neighbor device will
					# acquire its own separate lock. As a result, multiple threads
					# from different devices can concurrently modify the same data,
					# leading to a race condition.
					self.device.lock_for_location.acquire()
					for device in neighbours:
						device.set_data(location, result)
					
					self.device.set_data(location, result)
					self.device.lock_for_location.release()

			
			# Second synchronization point: all threads wait after script execution.
			self.device.bar.wait()
