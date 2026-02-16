


"""
@file device.py
@brief Implements a simulated distributed device in a sensor network.

This module defines the `Device` and `DeviceThread` classes, which represent
individual processing units in a distributed system. Each device manages
its own sensor data, interacts with a central supervisor, and communicates
with neighboring devices to process shared scripts and synchronize operations
using threading primitives and custom barriers.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
import time


class Device(object):
	"""
    @brief Represents a single device within the simulated distributed system.
    Functional Utility: Manages sensor data, assigned scripts, and coordinates
                        its thread of execution. It holds state relevant to its
                        identity, data, and its interaction with other devices
                        and a supervisor.
    """

	def __init__(self, device_id, sensor_data, supervisor):
		"""
        @brief Initializes a new Device instance.
        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary or similar structure holding sensor data for different locations.
        @param supervisor The supervisor object responsible for coordinating devices.
        """
		self.device_id = device_id # @brief Unique identifier for this device.
		self.sensor_data = sensor_data # @brief Dictionary containing sensor data, keyed by location.
		self.supervisor = supervisor # @brief Reference to the Supervisor instance managing this device.
		self.script_received = Event() # @brief Event flag indicating if a new script has been assigned.
		self.cb = Event() # @brief Control barrier event, used for synchronization with the supervisor.
		self.scripts = [] # @brief List to store assigned scripts, each with a location.
		self.timepoint_done = Event() # @brief Event flag indicating completion of a timepoint's processing.
		self.thread = DeviceThread(self) # @brief The dedicated thread of execution for this device.
		self.thread.start() # Functional Utility: Starts the `DeviceThread` upon initialization.
		self.data_lock = Lock() # @brief A threading.Lock to ensure thread-safe access to `sensor_data`.
		self.dev_l = [] # @brief List of all devices in the system (set by `setup_devices`).
		self.no_dev = -1 # @brief Total number of devices in the system (set by `setup_devices`).
		self.max = 0 # @brief Maximum sensor data value across all devices (set by `setup_devices`).
		self.location_lock = [] # @brief List of Locks, one for each possible sensor data location, for fine-grained locking.


	def __str__(self):
		"""
        @brief Returns a string representation of the Device.
        Functional Utility: Provides a human-readable string that identifies
                            the device by its `device_id`, useful for logging
                            and debugging purposes.
        @return A string in the format "Device <device_id>".
        """
		return "Device %d" % self.device_id


	def setup_devices(self, devices):
		"""
        @brief Sets up the list of all devices in the system and calculates the maximum sensor data value.
        Functional Utility: This method is called to initialize the device's knowledge
                            about all other devices in the distributed system. It populates
                            `dev_l` with the list of devices and determines the global
                            maximum sensor data value (`max`) across all devices.
        @param devices A list of all `Device` objects in the simulation.
        """
		# Block Logic: Determine the maximum sensor data value across all devices.
		# Invariant: After this loop, `self.max` holds the highest sensor value found
		#            among all devices' `sensor_data`.
		for x in devices:
			for val in x.sensor_data:
				
				if val > self.max:
					self.max = val
		
		self.dev_l = devices # @brief Store the list of all devices.
		self.no_dev = len(devices) # @brief Store the total number of devices.
	
	def get_max_dev(self):
		"""
        @brief Retrieves the maximum sensor data value observed across all devices.
        Functional Utility: Provides access to the globally determined maximum sensor
                            data value, which is typically computed during the
                            `setup_devices` phase.
        @return The maximum sensor data value (`self.max`).
        """
		return self.max
	
	def get_no_dev(self):
		"""
        @brief Retrieves the total number of devices in the simulated system.
        Functional Utility: Provides access to the count of all participating
                            devices, which is typically set during the
                            `setup_devices` phase.
        @return The total number of devices (`self.no_dev`).
        """
		return self.no_dev

	def assign_script(self, script, location):
		"""
        @brief Assigns a script to be executed at a specific data location on the device.
        Functional Utility: This method adds a new `script` and its associated `location`
                            to the device's list of pending scripts. It then sets the
                            `script_received` event to signal the device's thread that
                            new work is available. If no script is provided (i.e., `None`),
                            it still signals the thread, potentially indicating a no-op or
                            a termination signal.
        @param script The script object to be executed. If `None`, it signals completion without new script.
        @param location The data location (e.g., index) where the script should be applied.
        """
		if script is not None: # Block Logic: If a valid script is provided, append it to the list.
			self.scripts.append((script, location))
			self.script_received.set() # Functional Utility: Signal that a script has been received.
		else: # Block Logic: If no script is provided (None), still signal script received.
			self.script_received.set() # Functional Utility: Signal that a script has been received, possibly for a no-op or shutdown.

	def get_data(self, location):
		"""
        @brief Retrieves sensor data for a specific location.
        Functional Utility: Accesses the `sensor_data` dictionary to retrieve the
                            value associated with the given `location`. It provides
                            a safe way to read data without direct dictionary access.
        @param location The key representing the sensor data's location.
        @return The sensor data value if `location` exists in `sensor_data`, otherwise `None`.
        """
		return self.sensor_data[location] if location in self.sensor_data else None

	def set_data(self, location, data):
		"""
        @brief Sets sensor data for a specific location in a thread-safe manner.
        Functional Utility: Updates the sensor data value associated with the given
                            `location`. It uses a `data_lock` to ensure that
                            concurrent writes to `sensor_data` are synchronized,
                            preventing race conditions and data corruption.
        @param location The key representing the sensor data's location.
        @param data The new value to set for the sensor data at the specified `location`.
        """
		if location in self.sensor_data: # Block Logic: Check if the location exists in sensor data before attempting to update.
			self.data_lock.acquire() # Functional Utility: Acquire lock for thread-safe data modification.
			self.sensor_data[location] = data # Functional Utility: Update the sensor data.
			self.data_lock.release() # Functional Utility: Release lock after data modification.

	def set_b(self, barr,locat):
		"""
        @brief Sets the barrier and location-specific locks for the device.
        Functional Utility: This method is used to inject the `ReusableBarrierSem`
                            (`barr`) and a list of `location_lock`s into the device.
                            It also sets an event (`cb`) to signal that these
                            synchronization primitives have been configured,
                            allowing the `DeviceThread` to proceed.
        @param barr The `ReusableBarrierSem` instance for inter-device synchronization.
        @param locat A list of `Lock` objects, one for each sensor data location.
        """
		self.bar = barr # @brief Reusable barrier for synchronizing device threads.
		self.cb.set() # Functional Utility: Signal that barrier and location locks have been set.
		self.location_lock = locat # @brief List of Locks for individual data locations.

	def shutdown(self):
		"""
        @brief Shuts down the device's thread.
        Functional Utility: This method blocks until the `DeviceThread` associated
                            with this device completes its execution. It ensures
                            a clean termination of the device's operations.
        """
		self.thread.join()


class DeviceThread(Thread):
	"""
    @brief Represents the dedicated thread of execution for a `Device` instance.
    Functional Utility: This thread manages the device's main operational loop,
                        including synchronizing with other devices, executing
                        assigned scripts, processing sensor data, and propagating
                        results within the simulated distributed system.
    """

	def __init__(self, device):
		"""
        @brief Initializes a new DeviceThread instance.
        @param device The `Device` instance that this thread will manage.
        """
		Thread.__init__(self, name="Device Thread %d" % device.device_id) # Functional Utility: Initialize the base Thread class with a descriptive name.
		self.device = device # @brief Reference to the `Device` object this thread is associated with.




	def run(self):
		"""
        @brief The main execution loop for the device thread.
        Functional Utility: This method orchestrates the device's behavior in the simulated
                            distributed system. It continuously checks for assigned scripts,
                            executes them on relevant sensor data, propagates results to
                            neighboring devices, and synchronizes its operations with other
                            device threads using barriers and events. It also handles initial
                            setup for global synchronization mechanisms if it is the primary device.
        """
		i = 0 # @brief Counter or flag used for initial setup logic (specific to device 0).
		
		# Block Logic: Main operational loop for the device thread.
		# Invariant: The loop continues indefinitely until a shutdown condition is met
		#            (e.g., supervisor indicates no neighbors, implying termination).
		while True:
		
			# Block Logic: Get neighbors from the supervisor. If none, break the loop.
			# Invariant: `neighbours` will contain the list of devices this device can interact with.
			neighbours = self.device.supervisor.get_neighbours()
			if neighbours is None:
				break
		
			# Block Logic: Initial setup for global synchronization (only for device 0 at start).
			# Pre-condition: Executed once by device with `device_id == 0` during its first run.
			# Invariant: Global `location_lock`s and `ReusableBarrierSem` (`bar`) are initialized
			#            and distributed to all devices.
			if self.device.device_id == 0 and i == 0:
				max = 0 # @brief Local variable to track maximum sensor value for `location_lock` initialization.
				location_lock = [] # @brief Local list to hold location-specific locks.
				# Block Logic: Determine the overall maximum sensor value among all devices.
				for dev in self.device.dev_l:
					if dev.get_max_dev() > max:
						max = dev.get_max_dev()
				max = max + 1 # Functional Utility: Max value plus one to define lock array size.
				# Block Logic: Initialize location-specific locks.
				for j in range(max):
					location_lock.append(Lock())

				self.bar = ReusableBarrierSem(self.device.get_no_dev()) # Functional Utility: Initialize reusable barrier for all devices.
				i=2 # Functional Utility: Set `i` to non-zero to prevent re-initialization.
				# Block Logic: Distribute the initialized barrier and location locks to all devices.
				for dev in self.device.dev_l:
					dev.set_b(self.bar,location_lock)
		
			else:
				# Block Logic: Wait for the control barrier event to be set by the supervisor.
				# Invariant: Ensures that global synchronization primitives are set up before proceeding.
				self.device.cb.wait()
				
			# Functional Utility: Synchronize with all other device threads.
			# Ensures all devices are ready to proceed with the current timepoint's operations.
			self.device.bar.wait()
		
			# Block Logic: Wait for scripts to be assigned.
			# Invariant: The thread proceeds only when new scripts are available for execution.
			self.device.script_received.wait()
			
			# Block Logic: Execute assigned scripts and propagate results.
			# Pre-condition: `self.device.scripts` contains scripts to execute.
			# Invariant: Scripts are executed for their respective locations, and results are
			#            updated on the current device and its neighbors.
			for (script, location) in self.device.scripts:
				self.device.location_lock[location].acquire() # Functional Utility: Acquire lock for the specific location to ensure exclusive access to data.
			
				script_data = [] # @brief List to collect relevant sensor data for script execution.
				
				# Block Logic: Collect sensor data from neighboring devices for the current location.
				for device in neighbours:
					data = device.get_data(location)
					if data is not None:
						script_data.append(data)
				
				# Block Logic: Collect sensor data from the current device for the current location.
				data = self.device.get_data(location)
				if data is not None:
					script_data.append(data)
			
				# Block Logic: Execute the script if there is data.
				if script_data != []:
					
					result = script.run(script_data) # Functional Utility: Execute the script with collected data.
			
					# Block Logic: Propagate the script's result to neighboring devices.
					for device in neighbours:
						device.set_data(location, result) # Functional Utility: Update neighbor's sensor data.
					
					self.device.set_data(location, result) # Functional Utility: Update current device's sensor data.
					
					self.device.location_lock[location].release() # Functional Utility: Release lock for the specific location after data updates.

			# Functional Utility: Synchronize all devices again after script execution.
			# Ensures all devices have completed their script execution and data propagation
			# before moving to the next timepoint.
			self.device.bar.wait()
			self.device.timepoint_done.set() # Functional Utility: Signal that this device has completed its timepoint processing.
		
			# Functional Utility: Wait for the supervisor to signal the end of the timepoint.
			self.device.timepoint_done.wait()
            # Functional Utility: Clear the script_received event for the next cycle.
            self.device.script_received.clear()

