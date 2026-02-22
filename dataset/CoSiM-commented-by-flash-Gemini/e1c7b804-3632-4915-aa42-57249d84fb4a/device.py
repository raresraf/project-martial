


from threading import *
from barrier import ReusableBarrierCond
import Queue


"""
@brief This module defines `Device`, `Worker` (a persistent script worker thread), and `DeviceThread` classes
for simulating a distributed system.
@details It uses a `ReusableBarrierCond` for inter-device synchronization and employs a main device thread
that manages a pool of `Worker` threads. These `Worker` threads process scripts from a shared queue,
using a semaphore for task distribution and a pre-allocated list of locks for thread-safe sensor data access,
optimizing concurrent data processing and ensuring data integrity within the simulation.
"""

from threading import * # Import all threading components directly.
import Queue # Note: In Python 3, 'Queue' is typically 'queue'. Assuming Python 2 or specific setup.


class Device(object):
	"""
	@brief Represents a simulated device in a distributed sensor network.
	@details This class manages a device's unique ID, sensor data, and interactions with a supervisor.
	It can receive and queue scripts for execution, which are then processed by a pool of dedicated
	`Worker` threads. Synchronization across devices is managed by a shared `ReusableBarrierCond`,
	and thread-safe access to per-location sensor data is ensured by a pre-allocated list of `Lock` objects
	(`location_lock`). A `Queue` and `Semaphore` are used to manage the distribution of scripts to workers.
	@architectural_intent Acts as an autonomous agent in a distributed system, capable of local data
	processing and communication with peers, utilizing a persistent pool of worker threads for parallel
	script execution, granular locking for data consistency, and event-driven task distribution.
	"""
	
	def __init__(self, device_id, sensor_data, supervisor):
		"""
		@brief Initializes a new Device instance.
		@param device_id (int): A unique identifier for the device.
		@param sensor_data (dict): A dictionary containing initial sensor data,
		                           where keys are locations and values are data readings.
		@param supervisor (object): A reference to the supervisor object that manages
		                            the overall distributed system and provides access
		                            to network information (e.g., neighbors).
		"""
		self.device_id = device_id
		self.sensor_data = sensor_data
		self.supervisor = supervisor
		self.script_received = Event() # Event to signal when new scripts are ready for execution (unused in this version).
		self.scripts = []            # List to store assigned scripts (temporarily).
		self.timepoint_done = Event()  # Event to signal completion of a timepoint's script assignment.
		self.devices = []            # Reference to the list of all Device objects in the simulation.
		self.location_lock=[]        # Shared list of Locks for location-specific data access.
		self.queue=Queue.Queue()     # Shared queue for distributing scripts to worker threads.
		self.event_start = Event()   # Event to signal that initial setup is complete and workers can start.
		self.barrier = None          # Reference to the shared ReusableBarrierCond for inter-device synchronization.
		self.thread = DeviceThread(self) # The main controlling thread for this device.
		self.thread.start()          # Start the device's main controlling thread.
		self.script_semaphore = Semaphore(value=0) # Semaphore to signal availability of scripts in the queue.
		self.stop_workers = False    # Flag to signal worker threads to terminate.

		



	def __str__(self):
		"""
		@brief Returns a string representation of the Device.
		@return str: A string in the format "Device %d" % device_id.
		"""
		return "Device %d" % self.device_id

	def setup_devices(self, devices):
		"""
		@brief Sets up shared synchronization primitives (barrier and location-specific locks) and distributes them.
		@details This method initializes a single `ReusableBarrierCond` and a pre-allocated list of `Lock` objects
		(`location_lock`) by device 0. These resources are then distributed among all other devices in the simulation.
		It also sets the `event_start` to unblock worker threads.
		@param devices (list): A list of all Device objects in the simulation.
		@block_logic Centralized initialization and distribution of shared synchronization
		             and mutual exclusion primitives, and signaling worker readiness.
		@pre_condition `devices` is a list of `Device` instances, and this method is called for each device.
		@invariant `self.barrier` refers to a globally shared `ReusableBarrierCond` instance after setup.
		           `self.location_lock` is a shared list of `Lock` objects, ensuring consistent access to locations.
		           `self.event_start` is set, allowing worker threads to proceed.
		"""
		if self.device_id == 0: # Only device 0 is responsible for initializing and distributing shared resources.
			self.barrier = ReusableBarrierCond(len(devices)) # Create a new reusable barrier.
			# Block Logic: Pre-allocate locks for a fixed number of locations.
			# Invariant: `self.location_lock` is populated with 100 Lock objects.
			for i in range(100): # Create 100 Lock objects.
				self.location_lock.append(Lock())
			# Block Logic: Distribute the newly created barrier and location locks to all devices.
			# Invariant: All devices in `devices` receive a reference to the shared `self.barrier` and `self.location_lock`.
			for dev in devices:
				dev.location_lock = self.location_lock
				dev.barrier = self.barrier
				dev.event_start.set() # Signal worker threads to start.
		
		self.devices=devices # Store the list of all devices.

	def assign_script(self, script, location):
		"""
		@brief Assigns a script to be executed by the device at a specific location.
		@details If a script is provided, it's appended to the device's temporary script list.
		If no script is provided (i.e., `script` is None), it signifies that the current
		timepoint's script assignment is complete, and the `timepoint_done` event is set to
		unblock the `DeviceThread`.
		@param script (object): The script object to be executed, or None to signal end of assignments.
		@param location (str): The location associated with the script or data.
		@block_logic Handles the assignment of new scripts or signals the completion of script assignment for a timepoint.
		@pre_condition `self.scripts` is a list, `self.timepoint_done` is an Event object.
		@invariant Either a script is added to `self.scripts`, or `timepoint_done` is set.
		"""
		if script is not None:
			self.scripts.append((script, location)) # Add the script and its location to the queue.
		else:
			self.timepoint_done.set() # Signal that script assignments for the current timepoint are complete.

	def get_data(self, location):
		"""
		@brief Retrieves sensor data for a specific location.
		@param location (str): The location for which to retrieve data.
		@return object: The sensor data at the specified location, or None if the location is not found.
		"""
		return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		"""
		@brief Sets or updates sensor data for a specific location.
		@details This method updates the internal sensor data if the location exists.
		It's assumed that external synchronization (e.g., through `Worker` threads' locks)
		protects this operation during concurrent modifications.
		@param location (str): The location whose data is to be updated.
		@param data (object): The new data value for the specified location.
		@block_logic Updates the internal sensor data.
		@pre_condition `self.sensor_data` is a dictionary.
		@invariant If `location` is a key in `self.sensor_data`, its value is updated.
		"""
		if location in self.sensor_data:
			self.sensor_data[location] = data

	def shutdown(self):
		"""
		@brief Shuts down the device by joining its associated controlling thread.
		@details This ensures that the device's main `DeviceThread` completes its execution before the program exits.
		"""
		self.thread.join()

class Worker(Thread):
	"""
	@brief A persistent worker thread that continuously retrieves and executes scripts from its parent Device's shared queue.
	@details Each `Worker` thread is spawned once and remains active, acquiring scripts from a `Queue`
	signaled by a `Semaphore`. It executes assigned scripts, handles data collection from neighbors
	and itself, and updates sensor data in a thread-safe manner using location-specific locks.
	@architectural_intent Enhances parallelism by distributing script execution among a fixed pool
	of threads, ensuring controlled resource access through location-specific locks and
	semaphore-driven task consumption.
	"""

	def __init__(self,device,worker_id):
		"""
		@brief Initializes a new Worker thread.
		@param device (Device): The parent Device object that this worker thread serves.
		@param worker_id (int): A unique identifier for this specific worker thread within its parent Device.
		"""
		Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initialize the base Thread with a descriptive name.
		self.device = device # Reference to the parent Device object.
		self.worker_id = worker_id # Unique ID of this worker thread.

	def run(self):
		"""
		@brief The main execution loop for the Worker thread.
		@details This method continuously acquires permits from `self.device.script_semaphore`
		(signaling available scripts in the queue). If a script is available and the `stop_workers`
		flag is not set, it retrieves a script from `self.device.queue`, acquires a location-specific
		lock, collects data from neighbors and itself, executes the script, and propagates results.
		The loop terminates when the `stop_workers` flag is set and all scripts are processed.
		@block_logic Continuously processes scripts from a shared queue, executing them with thread-safe data access.
		@pre_condition `self.device` is an initialized Device object with access to `queue`,
		               `script_semaphore`, `location_lock`, `stop_workers`, and `supervisor`.
		@invariant The worker thread either processes a script or waits for one, respecting termination signals.
		"""
		while True:
			# Block Logic: Acquire a permit from the semaphore, waiting if no scripts are available.
			# Invariant: A permit is acquired only when a script is in the queue or a termination signal is sent.
			self.device.script_semaphore.acquire()


			if self.device.stop_workers is True:
				break

			# Functional Utility: Retrieve a script task from the shared queue.
			tuplu=self.device.queue.get() # Dequeue the task tuple (script, location, neighbours, lock).
			script = tuplu[0]
			location = tuplu[1]
			neighbours = tuplu[2]
			lock = tuplu[3]

			# Block Logic: Acquire a lock for the specific location to ensure exclusive access to its data.
			# Invariant: Only one worker thread can modify or read data for `location` at a time.
			with lock: # Uses `with` statement for automatic lock acquisition and release.
				script_data = [] # List to accumulate data for the current script's execution.
				
				# Block Logic: Collect data from neighboring devices for the current location.
				# Invariant: `script_data` will contain data from all available neighbors for the given location.
				for device in neighbours:
					data = device.get_data(location)
					if data is not None:
						script_data.append(data)
						
						# This nested data collection seems to be an error in the original code,
						# as it's collecting data from `self.device` inside the loop for `neighbours`.
						# It should likely be outside this loop. However, respecting "Zero Code Mutation".
						data = self.device.get_data(location)
						if data is not None:
							script_data.append(data)
				# Corrected logic would typically collect data from self.device once outside the neighbours loop.
				# Assuming the intent is to collect data from self.device AFTER collecting from neighbours for the script.
				# The original code structure is kept as per "Zero Code Mutation" rule.

				# Block Logic: Execute the script if there is any data to process.
				# Pre-condition: `script` is an object with a `run` method, and `script_data` is a list of data.
				# Invariant: `result` holds the output of the script's execution.
				if script_data != []:
						
					result = script.run(script_data) # Execute the script with the collected data.
						
					# Block Logic: Propagate the script's result to neighboring devices.
					# Invariant: All neighbors receive the updated data for the given location.
					for device in neighbours:
						device.set_data(location, result)
						
					# Functional Utility: Update the current device's own data with the script's result.
					self.device.set_data(location, result)
class DeviceThread(Thread):
	"""
	@brief The main controlling thread for a Device instance.
	@details This thread orchestrates the device's operational cycle for each timepoint.
	It spawns a fixed pool of `Worker` threads once at the beginning, then continuously
	retrieves neighbor information, waits for script assignments, adds scripts to a
	shared `Queue`, releases `script_semaphore` permits for workers, and finally
	synchronizes with other `DeviceThread` instances via the global `ReusableBarrierCond`.
	It also handles the graceful shutdown of worker threads.
	@architectural_intent Manages the lifecycle and execution flow of a Device in the simulation,
	abstracting multi-threaded script execution through a persistent worker pool and ensuring proper
	coordination and data consistency within the distributed system.
	"""
	
	def __init__(self, device):
		"""
		@brief Initializes a new DeviceThread instance.
		@param device (Device): The Device object that this thread will manage.
		"""
		Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initialize the base Thread with a descriptive name.
		self.device = device # Reference to the parent Device object.

	def run(self):
		"""
		@brief The main execution loop for the DeviceThread.
		@details This method first waits for the initial setup to complete (`event_start`),
		then spawns 8 persistent `Worker` threads. It then enters a continuous loop where it
		retrieves neighbor information from the supervisor. If neighbors are available,
		it waits until `timepoint_done` is set (signaling script assignments are complete),
		adds each assigned script to a shared `Queue`, and releases a permit from
		`script_semaphore` for each script. After all scripts are queued, it clears
		`timepoint_done` and synchronizes with other `DeviceThread` instances via the
		global `ReusableBarrierCond`. The loop terminates when the supervisor signals
		the end of the simulation. Finally, it signals worker threads to stop and joins them.
		@block_logic Orchestrates the device's operational cycle, handling timepoint progression,
		script queuing for parallel execution, and inter-device synchronization and termination.
		@pre_condition `self.device` is an initialized Device object with access to `supervisor`,
		               `scripts` list, `timepoint_done` event, `queue`, `script_semaphore`, `barrier`,
		               and `event_start`.
		@invariant The thread progresses through timepoints, processes scripts by queuing them for workers,
		           and ensures global synchronization and graceful termination.
		"""
		# Block Logic: Wait for initial setup to complete before spawning worker threads.
		# Invariant: `self.device.event_start` is set by device 0 once global resources are distributed.
		self.device.event_start.wait()
		t=[] # List to hold the spawned Worker threads.
		# Block Logic: Spawn 8 persistent Worker threads.
		# Invariant: `t` contains 8 active `Worker` threads.
		for i in range(8): # Creates 8 Worker thread instances.
			thread=Worker(self.device,i)
			t.append(thread)

		# Block Logic: Start all spawned Worker threads.
		# Invariant: All `Worker` threads begin their `run` method concurrently.
		for i in range(8):
			t[i].start()

		while True:
			# Functional Utility: Get information about neighboring devices from the supervisor.
			neighbours = self.device.supervisor.get_neighbours()
			
			# Block Logic: Check if the simulation should terminate.
			# Pre-condition: `neighbours` list indicates the current state of the network.
			# Invariant: The loop terminates if no neighbors are returned by the supervisor.
			if neighbours is None:
				break

			# Block Logic: Wait until script assignments for the current timepoint are complete.
			# Pre-condition: `self.device.timepoint_done` is an Event object.
			# Invariant: The thread proceeds only after the supervisor signals completion of script assignment.
			self.device.timepoint_done.wait()

			# Block Logic: Queue all assigned scripts and release semaphore permits for worker threads.
			# Invariant: Each script is added to `self.device.queue`, and a permit is released.
			for (script, location) in self.device.scripts:
				# Functional Utility: Add the script task (with its location-specific lock) to the shared queue.
				self.device.queue.put((script , location , neighbours, self.device.location_lock[location]))
				self.device.script_semaphore.release() # Release a permit to signal a worker thread.

			# Functional Utility: Clear the `timepoint_done` event for the next timepoint.
			self.device.timepoint_done.clear()
            # Functional Utility: Clear the scripts list for the next timepoint.
			self.device.scripts = [] # Reset scripts list for the next timepoint.


			# Block Logic: Synchronize with other DeviceThread instances via the shared barrier.
			# Invariant: All DeviceThread instances in the simulation will reach this barrier before any proceeds to the next timepoint.
			self.device.barrier.wait()

		# Block Logic: Signal worker threads to stop and wait for their termination.
		self.device.stop_workers = True # Set the flag to terminate workers.

		# Functional Utility: Release remaining semaphore permits to unblock all workers for termination.
		for i in range(8):
			self.device.script_semaphore.release()

		# Block Logic: Wait for all worker threads to gracefully terminate.
		# Invariant: The DeviceThread will not exit until all its Worker children have finished.
		for i in range(8):
			t[i].join()
