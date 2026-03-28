"""
Defines a device simulation model using a producer-consumer pattern with a
thread pool and a work queue.

This module contains a `Device` class that uses a `DeviceThread` to manage a
pool of worker threads. The `DeviceThread` acts as a producer, placing script-execution
jobs onto a `Queue`, and the worker threads act as consumers, processing jobs
from the queue. Synchronization is managed via a global barrier and locks for
specific data locations.
"""

from threading import Event, Thread, Lock
from Queue import Queue
# Note: The file imports 'reentrantbarrier' but also defines a 'ReusableBarrier'
# class below, which seems to be what is actually used.
from reentrantbarrier import Barrier

class Device(object):
    """
    Represents a device node which manages a pool of worker threads via a queue.
    
    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): The device's local sensor data.
        supervisor: A reference to the simulation supervisor.
        scripts (list): A list of scripts to be executed in the current timepoint.
        timepoint_done (Event): An event signaling the end of script assignment.
        location_locks (dict): A dictionary of locks, one per data location.
        barrier (Barrier): A global barrier to synchronize all devices.
        thread (DeviceThread): The manager thread for this device.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its manager thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.location_locks = {}
        self.barrier = None
        self.ready_to_get_script = False
        self.all_devices = None

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """Sets the global device barrier."""
        self.barrier = barrier

    def broadcast_barrier(self, devices):
        """Broadcasts the created barrier to all other devices."""
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(self.barrier)

    def setup_devices(self, devices):
        """
        Sets up shared resources. Device 0 creates and broadcasts the global barrier.
        """
        self.all_devices = devices
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            self.broadcast_barrier(devices)

    def assign_script(self, script, location):
        """
        Assigns a script for processing. If script is None, signals end of timepoint.
        """
        if script is None:
            self.timepoint_done.set()
            return
        else:
            # Lazily initialize and broadcast a lock for a newly seen location.
            if self.location_locks.setdefault(location, None) is None:
                self.location_locks[location] = Lock()
                self.ready_to_get_script = True
            self.broadcast_lock_for_location(location)

            self.scripts.append((script, location))
            self.script_received.set()

    def broadcast_lock_for_location(self, location):
        """Ensures all devices have a reference to the same lock for a given location."""
        for device_no in xrange(len(self.all_devices)):
            self.all_devices[device_no].location_locks[location] = self.location_locks[location]

    def set_data(self, location, data):
        """Sets the data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def get_data(self, location):
        """Gets the data for a given location."""
        return self.sensor_data.get(location)

    def shutdown(self):
        """Shuts down the device by joining its manager thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    Manages a thread pool and a work queue for its parent device.
    Acts as the "producer" in a producer-consumer model.
    """
    def __init__(self, device):
        """Initializes the thread pool and work queue."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_count = 8
        self.pool = Queue(self.thread_count)
        self.threads = []
        self.create_workers()
        self.start_workers()

    def create_workers(self):
        """Creates worker threads that will consume from the queue."""
        for _ in xrange(self.thread_count):
            self.threads.append(Thread(target=self.execute_script))

    def start_workers(self):
        """Starts all worker threads."""
        for thread in self.threads:
            thread.start()

    def collect_data_from_neighbours(self, neighbours, location):
        """Helper to gather data for a location from all neighbors."""
        result = []
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    result.append(data)
        return result

    def execute_script(self):
        """
        The target function for worker threads (the "consumer").
        
        Pulls jobs from the queue, executes the script, and signals completion.
        A (None, None, None) tuple is the sentinel for shutdown.
        """
        neighbours, script, location = self.pool.get()
        while True:
            if neighbours is None and script is None and location is None:
                self.pool.task_done()
                break

            script_data = []
            # Acquire lock for the location before processing.
            self.device.location_locks[location].acquire()
            
            # Collect data from neighbors and self.
            collected_data = self.collect_data_from_neighbours(neighbours, location)
            if collected_data:
                script_data.extend(collected_data)
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Run script and update data on all relevant devices.
            if script_data:
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
            
            self.device.location_locks[location].release()
            
            self.pool.task_done() # Signal that this job is complete.
            neighbours, script, location = self.pool.get() # Get next job.

    def run(self):
        """
        The main loop for the manager thread (the "producer").
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Shutdown signal
                break

            # The logic here is complex, using an event and a flag to manage
            # the state of script processing for the timepoint.
            while True:
                # Waits for the supervisor to signal that script assignment is done.
                self.device.timepoint_done.wait()

                if not self.device.ready_to_get_script:
                    self.device.timepoint_done.clear()
                    self.device.ready_to_get_script = True
                    break
                else:
                    # Place all scripts for this timepoint onto the work queue.
                    for (script, location) in self.device.scripts:
                        self.pool.put((neighbours, script, location))
                    self.device.ready_to_get_script = False

            # Wait for all workers to finish processing all items in the queue.
            self.pool.join()
            # Wait at the global barrier for all other devices to finish.
            self.device.barrier.wait()

        # Clean shutdown sequence for the thread pool.
        self.pool.join() # Ensure queue is empty before shutdown.
        # Add a sentinel value for each worker to signal termination.
        for _ in xrange(self.thread_count):
            self.pool.put((None, None, None))
        for thread in self.threads:
            thread.join()
        self.device.location_locks.clear()
