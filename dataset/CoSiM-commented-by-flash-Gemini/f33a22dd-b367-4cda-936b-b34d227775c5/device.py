


from threading import *
from barrier import ReusableBarrierCond
import Queue


class Device(object):
	"""
	Represents a device in a simulated distributed system. Each device
	manages sensor data, processes scripts, and coordinates with a supervisor
	and other devices using a thread pool and synchronization primitives.
	"""
	def __init__(self, device_id, sensor_data, supervisor):
		"""
		Initializes a new Device instance.

		Args:
			device_id (int): A unique identifier for the device.
			sensor_data (dict): A dictionary holding the device's sensor data,
								keyed by location.
			supervisor (Supervisor): The supervisor object responsible for
									 managing devices and providing neighborhood information.
		"""
		self.device_id = device_id
		self.sensor_data = sensor_data
		self.supervisor = supervisor
		# Event to signal when a new script has been assigned to the device.
		self.script_received = Event()
		# List to store (script, location) tuples for execution.
		self.scripts = []
		# Event to signal that the device has completed processing for the current timepoint.
		self.timepoint_done = Event()
		# List of all devices in the simulation (set by setup_devices).
		self.devices = []
		# List of Lock objects, providing fine-grained locking for specific data locations.
		self.location_lock=[]
		# Queue for scripts to be processed by worker threads.
		self.queue=Queue.Queue()
		# Event to signal the start of worker threads after setup is complete.
		self.event_start = Event()
		# Global barrier for synchronizing all devices. Set by setup_devices.
		self.barrier = None
		# The dedicated thread for this device's execution logic (manages worker pool).
		self.thread = DeviceThread(self)
		self.thread.start()
		# Semaphore to control the flow of tasks to worker threads.
		self.script_semaphore = Semaphore(value=0)
		# Flag to signal worker threads to stop.
		self.stop_workers = False

	def __str__(self):
		"""
		Returns a string representation of the Device.
		"""
		return "Device %d" % self.device_id

	def setup_devices(self, devices):
		"""
		Configures shared synchronization primitives (barrier and per-location locks)
		among a group of devices. The device with device_id 0 acts as the initializer
		for these shared resources and starts the DeviceThread for all devices.

		Args:
			devices (list): A list of all Device objects participating in the simulation.
		"""
		# Only the device with device_id == 0 initializes the shared resources.
		if self.device_id == 0:
			# Create a reusable barrier for all participating devices.
			self.barrier = ReusableBarrierCond(len(devices))
			# Initialize a list of 100 Lock objects for location-specific locking.
			for i in range(100):
				self.location_lock.append(Lock())
			# Distribute the shared location locks and barrier to all devices,
			# and signal their DeviceThreads to start.
			for dev in devices:
				dev.location_lock = self.location_lock
				dev.barrier = self.barrier
				dev.event_start.set()
		
		self.devices=devices

	def assign_script(self, script, location):
		"""
		Assigns a script to be executed by the device at a specific data location.
		If `script` is None, it signals the completion of script assignments for the timepoint.

		Args:
			script (Script): The script object to be executed.
			location: The data location pertinent to the script.
		"""
		if script is not None:
			self.scripts.append((script, location))
		else:
			# Signal that all scripts for this timepoint have been assigned.
			self.timepoint_done.set()

	def get_data(self, location):
		"""
		Retrieves sensor data for a given location.

		Args:
			location: The key for the sensor data.

		Returns:
			The sensor data for the specified location, or None if not found.
		"""
		return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		"""
		Sets or updates sensor data for a given location.

		Args:
			location: The key for the sensor data.
			data: The new data to set.
		"""
		if location in self.sensor_data:
			self.sensor_data[location] = data

	def shutdown(self):
		"""
		Shuts down the device by joining its associated DeviceThread.
		"""
		self.thread.join()

class Worker(Thread):
	"""
	A worker thread within a device's thread pool, responsible for
	processing script execution tasks from a shared queue.
	"""
	def __init__(self,device,worker_id):
		"""
		Initializes a Worker thread.

		Args:
			device (Device): The Device object to which this worker belongs.
			worker_id (int): A unique identifier for this worker thread.
		"""
		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device
		self.worker_id = worker_id

	def run(self):
		"""
		The main execution loop for the worker thread. It continuously
		acquires tasks from the device's queue, executes scripts under
		a location-specific lock, and propagates results.
		"""
		while True:
			# Acquire semaphore, blocking until a script is available or shutdown is signaled.
			self.device.script_semaphore.acquire()
			# Check for shutdown signal.
			if self.device.stop_workers is True:
				break

			# Retrieve a script execution task from the queue.
			tuplu=self.device.queue.get()
			script = tuplu[0]
			location = tuplu[1]
			neighbours = tuplu[2]
			lock = tuplu[3] # Per-location lock for the current script's data.

			# Acquire the location-specific lock to ensure exclusive access to data.
			with lock:
				script_data = []
				
				# Iterate through neighbors to collect data.
				for device in neighbours:
					data = device.get_data(location)
					if data is not None:
						script_data.append(data)
						
						# BUG: The following two lines appear to be incorrectly indented.
						# 'data = self.device.get_data(location)' should likely be outside
						# the 'for device in neighbours' loop to avoid multiple appends
						# of the local device's data.
						data = self.device.get_data(location)
						if data is not None:
							script_data.append(data)
				
				# Execute the script if data was collected.
				if script_data != []:
						
					result = script.run(script_data)
						
					# Propagate result to neighbors.
					for device in neighbours:
						device.set_data(location, result)
						
					# Update local device data.
					self.device.set_data(location, result)
		# After loop, clear script_data (though it's a local variable, so this line is redundant).
		# self.script_data = []


class DeviceThread(Thread):
	"""
	The main execution thread for a Device object.
	It acts as a producer, queueing script execution tasks for Worker threads,
	and manages the overall time-stepped simulation, including global synchronization.
	"""
	def __init__(self, device):
		"""
		Initializes the DeviceThread.

		Args:
			device (Device): The Device object associated with this thread.
		"""
		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device

	def run(self):
		"""
		The main loop for the device thread. It initializes worker threads,
		manages the queuing of script tasks, handles timepoint synchronization,
		and orchestrates the graceful shutdown of worker threads.
		"""
		# Wait for the event_start signal, indicating that global setup is complete.
		self.device.event_start.wait()
		t=[] # List to hold worker thread instances.
		# Spawn 8 worker threads.
		for i in range(8):
			thread=Worker(self.device,i)
			t.append(thread)

		# Start all worker threads.
		for i in range(8):
			t[i].start()

		while True:
			# Get current neighbors from supervisor.
			neighbours = self.device.supervisor.get_neighbours()
			# If no neighbors, the simulation has ended.
			if neighbours is None:
				break

			# Wait until scripts for the current timepoint are ready.
			self.device.timepoint_done.wait()

			# Enqueue all scripts for worker threads to process.
			for (script, location) in self.device.scripts:
				# Add script, location, neighbors, and the corresponding lock to the queue.
				self.device.queue.put((script , location , neighbours, self.device.location_lock[location]))
				# Release the semaphore to signal a worker thread that a new task is available.
				self.device.script_semaphore.release()

			# Reset timepoint_done for the next cycle.
			self.device.timepoint_done.clear()
			# Clear the list of scripts as they have all been enqueued.
			self.device.scripts = []

			# Wait for all devices to reach this global barrier before proceeding to the next timepoint.
			self.device.barrier.wait()

		# --- Shutdown phase ---
		self.device.stop_workers = True # Signal workers to stop.

		# Release semaphore for each worker to unblock them so they can exit gracefully.
		for i in range(8):
			self.device.script_semaphore.release()

		# Wait for all worker threads to finish.
		for i in range(8):
			t[i].join()
