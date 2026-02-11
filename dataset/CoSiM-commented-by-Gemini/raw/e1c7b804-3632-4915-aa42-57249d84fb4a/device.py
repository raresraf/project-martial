"""
This module implements a device simulation using a producer-consumer pattern
with a dedicated pool of worker threads for each device.

The main `DeviceThread` acts as a producer, placing tasks on a queue, while a
pool of `Worker` threads act as consumers, processing these tasks in parallel.
Synchronization is managed by a queue, a semaphore, and a reusable barrier.
"""

from threading import *
from barrier import ReusableBarrierCond
import Queue


class Device(object):
	"""
	Represents a device that manages a task queue and a pool of worker threads.

	Each device encapsulates the logic for distributing scripts as tasks to its
	internal workers and synchronizing with other devices in the network.
	"""
	def __init__(self, device_id, sensor_data, supervisor):
		
		self.device_id = device_id
		self.sensor_data = sensor_data
		self.supervisor = supervisor
		self.script_received = Event()
		self.scripts = []
		self.timepoint_done = Event()
		self.devices = []
		# A list of locks for different data locations, shared among devices.
		self.location_lock=[]
		# A queue for holding tasks (scripts) for the worker threads.
		self.queue=Queue.Queue()
		self.event_start = Event()
		self.barrier = None
		self.thread = DeviceThread(self)
		self.thread.start()
		# A semaphore to signal to worker threads that a new task is available.
		self.script_semaphore = Semaphore(value=0)
		# A flag to signal workers to terminate.
		self.stop_workers = False

		



	def __str__(self):
		"""Returns a string representation of the device."""
		return "Device %d" % self.device_id

	def setup_devices(self, devices):
		"""
		Initializes shared synchronization objects for all devices.

		The leader device (ID 0) creates a shared barrier and pre-allocates a
		fixed-size list of 100 locks for data locations, which are then
		distributed to all devices.
		"""

		if self.device_id == 0:
			self.barrier = ReusableBarrierCond(len(devices))
			for i in range(100):
				self.location_lock.append(Lock())
			for dev in devices:
				dev.location_lock = self.location_lock
				dev.barrier = self.barrier
				dev.event_start.set()
		
		self.devices=devices

		

		

	def assign_script(self, script, location):
		"""Assigns a script to the device."""
		if script is not None:
			self.scripts.append((script, location))
		else:
			self.timepoint_done.set()

	def get_data(self, location):
		"""Retrieves sensor data for a given location."""


		return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		"""Sets sensor data at a specific location."""
		if location in self.sensor_data:
			self.sensor_data[location] = data

	def shutdown(self):
		"""Waits for the device's main thread to terminate."""
		self.thread.join()

class Worker(Thread):
	"""
	A worker thread that consumes and executes tasks from a queue.
	
	Each device has a pool of these workers running concurrently.
	"""
	def __init__(self,device,worker_id):


		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device
		self.worker_id = worker_id

	def run(self):
		"""The main loop for the worker thread."""
		while True:
			# Wait for the semaphore to be released, indicating a task is available.
			self.device.script_semaphore.acquire()


			if self.device.stop_workers is True:
				break

			# Retrieve a task from the queue.
			tuplu=self.device.queue.get()
			script = tuplu[0]
			location = tuplu[1]
			neighbours = tuplu[2]
			lock = tuplu[3]

			# Process the task with the appropriate lock.
			with lock:
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






class DeviceThread(Thread):
	"""
	The main control thread for a device, acting as a task producer.

	This thread is responsible for creating the worker pool and then, in each
	timepoint, placing script execution tasks onto a queue for the workers
	to consume.
	"""

	def __init__(self, device):
		


		Thread.__init__(self, name="Device Thread %d" % device.device_id)
		self.device = device

	def run(self):
		"""The main producer loop."""

		self.device.event_start.wait()
		# Create and start the pool of worker threads.
		t=[]
		for i in range(8):
			thread=Worker(self.device,i)
			t.append(thread)

		for i in range(8):
			t[i].start()

		while True:
			

			neighbours = self.device.supervisor.get_neighbours()
			if neighbours is None:
				break

			# Wait for the signal to start queueing tasks for the timepoint.
			self.device.timepoint_done.wait()

			

			# Act as a producer: put tasks on the queue and signal the semaphore.
			for (script, location) in self.device.scripts:
				self.device.queue.put((script , location , neighbours, self.device.location_lock[location]))
				self.device.script_semaphore.release()

			
			self.device.timepoint_done.clear()

			# Wait for all devices to finish their timepoint.
			self.device.barrier.wait()

		# --- Shutdown sequence ---
		self.device.stop_workers = True 

		# Release the semaphore enough times to unblock all workers.
		for i in range(8):
			self.device.script_semaphore.release()

		# Wait for all worker threads to terminate.
		for i in range(8):
			t[i].join()

		
