"""
@file ScriptExecutor.py
@brief Defines a device model using worker threads with semaphore-based concurrency control.

This file implements a `Device` simulation where script execution is handled by
`ScriptExecutor` worker threads. The number of concurrent workers is limited by a
`Semaphore`. A root device is responsible for discovering all unique data locations
and creating a shared set of location-specific locks and a shared barrier for
synchronization.
"""

from threading import Event, Thread, Lock, Semaphore
from ReusableBarrier import ReusableBarrier

class Device(object):
    """
    Represents a device in the simulation, which uses a semaphore to manage
    a pool of worker threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        # A list of locks, one for each unique location, shared among all devices.
        self.location_locks = []

        self.barrier = None

        self.threads = []
        # Limits the number of concurrent ScriptExecutor threads to 8.
        self.threads_limit = Semaphore(8)
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared locks and a barrier.

        Executed by the root device (ID 0). It scans all devices to find every
        unique location, creates a corresponding lock for each, and then shares
        the list of locks and a new barrier with all other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            my_locations = []

            # Block Logic: Discover all unique locations across all devices.
            for device in devices:
                locations = device.sensor_data.keys()
                for location in locations:
                    if location not in my_locations:
                        my_locations.append(location)
                        self.location_locks.append(Lock())
            
            barrier = ReusableBarrier(len(devices))

            # Invariant: Distribute the shared locks and barrier to all devices.
            for device in devices:
                device.location_locks = self.location_locks
                device.barrier = barrier

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of script assignment for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. This access is not synchronized by this method."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data. This access is not synchronized by this method."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a device, which manages the lifecycle of
    `ScriptExecutor` worker threads.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            self.device.threads = []

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()

            # Block Logic: For each script, acquire a semaphore slot, then create a worker.
            for (script, location) in self.device.scripts:
                self.device.threads_limit.acquire()
                self.device.threads.append(ScriptExecutor(
                    self.device, neighbours, location, script
                    ))
            
            # Start and join all created worker threads.
            for thread in self.device.threads:
                thread.start()
            for thread in self.device.threads:
                thread.join()

            self.device.timepoint_done.clear()
            # Wait at the barrier for all devices to finish the timepoint.
            self.device.barrier.wait()


class ScriptExecutor(Thread):
	"""
	A worker thread that executes a single script, respecting a location-specific
	lock and a concurrency-limiting semaphore.
	"""
	def __init__(self, device, neighbours, current_location, script):
		Thread.__init__(self)
		self.device = device
		self.script = script
		self.neighbours = neighbours
		self.current_location = current_location

	def run(self):
		"""Executes the script and manages synchronization."""
		# Pre-condition: Acquire the lock for this script's specific location.
		self.device.location_locks[self.current_location].acquire()
		script_data = []

		# Block Logic: Gather data from neighbors and self.
		for device in self.neighbours:
			data = device.get_data(self.current_location)
			if data is not None:
				script_data.append(data)
		
		data = self.device.get_data(self.current_location)
		if data is not None:
			script_data.append(data)
		
		# Invariant: Data is gathered and ready for execution.
		if script_data != []:
			result = self.script.run(script_data)

			# Propagate the result back to all devices.
			for device in self.neighbours:
				device.set_data(self.current_location, result)
			
			self.device.set_data(self.current_location, result)
		
		# Release the location-specific lock.
		self.device.location_locks[self.current_location].release()
		# Release the semaphore to allow another worker thread to start.
		self.device.threads_limit.release()