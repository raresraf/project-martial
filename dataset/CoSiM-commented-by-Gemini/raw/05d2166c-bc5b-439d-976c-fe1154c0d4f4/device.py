"""
Models a network of devices that collaboratively process sensor data in a synchronized,
time-stepped simulation. This implementation uses a coarse-grained locking mechanism and
a specific data update rule where values are only changed if the new value is greater,
suggesting a distributed maximum-finding or state-converging algorithm.
"""

from threading import Event, Thread, Lock
from Barrier import *


class Device(object):
	"""
	Represents a single device in the network. Each device manages its own sensor data
	and executes scripts in a dedicated thread, synchronizing with other devices at the
	end of each time step using a shared barrier.
	"""

	def __init__(self, device_id, sensor_data, supervisor):
		"""
		Initializes a device.

		Args:
			device_id (int): The unique identifier for this device.
			sensor_data (dict): A dictionary holding the initial sensor values for the device,
			                    keyed by location.
			supervisor (Supervisor): An object responsible for providing network topology
			                         (neighbor information).
		"""
		self.device_id = device_id
		self.sensor_data = sensor_data
		self.supervisor = supervisor

		# Event used to signal that all scripts for a timepoint have been assigned.
		self.script_received = Event()
		self.scripts = []
		self.num_devices = 0
		
		# The main execution thread for this device.
		self.thread = DeviceThread(self)
		self.thread.start()

	def __str__(self):
		"""Returns a string representation of the device."""
		return "Device %d" % self.device_id

	def setup_devices(self, devices):
		"""
		Initializes and distributes a shared barrier for synchronization across all device threads.
		This method relies on device 0 to create the barrier and then shares it with all others.

		Args:
			devices (list): A list of all Device objects in the simulation.
		"""
		self.num_devices = len(devices)
		# Pre-condition: Barrier should be created only once by the first device in the list.
		if devices[0].thread.barrier is None:
			devices[0].thread.barrier = Barrier(self.num_devices, str(self.device_id))
			# Invariant: All device threads must share the same barrier instance.
			for i in range (1, len(devices)):
				devices[i].thread.barrier = devices[0].thread.barrier

	def assign_script(self, script, location):
		"""
		Assigns a script to the device. A `None` script signals that all scripts
		for the current timepoint have been assigned and processing can begin.

		Args:
			script (Script): The script object to execute.
			location (str): The location associated with the script.
		"""
		if script is not None:
			# Use the thread's lock to ensure thread-safe modification of the scripts list.
			with self.thread.lock:
				self.scripts.append((script, location))
		else:
			# A 'None' script acts as a trigger to start the computation for the timepoint.
			self.script_received.set()

	def get_data(self, location):
		"""
		Thread-safely retrieves sensor data for a given location.

		Args:
			location (str): The location from which to retrieve data.

		Returns:
			The sensor data if the location exists, otherwise None.
		"""
		# A single lock protects all sensor data access for this device.
		with self.thread.lock:
			return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		"""
		Thread-safely updates sensor data. The update is conditional: the new data is only
		written if it is greater than the current value. This suggests a convergence
		algorithm, such as finding a maximum value across devices.

		Args:
			location (str): The location to update.
			data: The new data value.
		"""
		with self.thread.lock:
			if location in self.sensor_data:
				# This condition implies a specific algorithmic purpose, e.g., distributed max.
				if self.sensor_data[location] < data:
					self.sensor_data[location] = data

	def shutdown(self):
		"""Waits for the device's main execution thread to terminate."""
		self.thread.join()


class DeviceThread(Thread):
	"""
	The main execution thread for a device. It waits for scripts, executes them,
	and synchronizes with other devices at a barrier.
	"""

	def __init__(self, device):
		"""
		Initializes the device thread.

		Args:
			device (Device): The parent device this thread belongs to.
		"""
		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device
		# The shared barrier instance for synchronizing all device threads.
		self.barrier = None
		# A coarse-grained lock protecting this device's shared data (scripts and sensor_data).
		self.lock = Lock()

	def run(self):
		"""
		The main control loop for the device. It processes scripts for each timepoint
		in a synchronized manner.
		"""
		# The outer loop represents the entire simulation lifetime.
		while True:
			# Get the current set of neighbors for this timepoint.
			neighbours = self.device.supervisor.get_neighbours()
			if neighbours is None:
				# A 'None' neighbor list signals the end of the simulation.
				break

			# Block until the supervisor signals that all scripts for this timepoint are assigned.
			self.device.script_received.wait()
			self.device.script_received.clear()

			# Process all assigned scripts for the current timepoint.
			for (script, location) in self.device.scripts:
				script_data = []
				
				# Gather data from all neighbors for the specified location.
				for device in neighbours:
					data = device.get_data(location)
					if data is not None:
						script_data.append(data)
				
				# Also include the local device's data.
				data = self.device.get_data(location)
				if data is not None:
					script_data.append(data)

				# Pre-condition: Only execute the script if there is data to process.
				if script_data:
					result = script.run(script_data)

					# Distribute the result to all neighbors, applying the update rule.
					for device in neighbours:
						device.set_data(location, result)
					# Update the local device's data as well.
					self.device.set_data(location, result)

			# Invariant: All threads must wait at the barrier. This ensures that no device
			# begins the next timepoint until all devices have completed the current one.
			self.barrier.wait()