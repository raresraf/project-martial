"""
This module simulates a network of devices processing sensor data in parallel.

This implementation uses a classic producer-consumer pattern, where each Device's
main thread acts as a producer, putting tasks on a shared queue, and a persistent
pool of Worker threads act as consumers. It features fine-grained locking based on
data location and a shared barrier for step synchronization.

NOTE: The module depends on a `ReusableBarrier` class from a `barrier` module,
which is not provided. Its functionality is inferred from its name.
"""

from threading import Event, Thread, Lock
from Queue import Queue
from barrier import ReusableBarrier


class Device(object):
    """
    Represents a single device in the network, managing its own control thread
    and a pool of persistent worker threads that consume from a shared queue.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device, its control thread, and its worker pool.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.setup_done = Event()
        self.scripts_already_parsed = Event()
        self.queue = Queue()            # Shared task queue for workers.
        self.barrier = None             # Shared barrier for step synchronization.
        self.location_locks = None      # Shared dict of locks for each location.
        self.neighbours = None
        self.thread = DeviceThread(self)
        self.thread.start()
        
        # Create and start a persistent pool of worker threads.
        self.workers = []
        for _ in xrange(8):
            worker = Worker(self)
            worker.start()
            self.workers.append(worker)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization objects (barrier, locks).
        Device 0 acts as the leader, creating the objects and distributing
        them to all other devices.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices)) # Assumed from import
            self.location_locks = {}
            for device in devices:
                device.location_locks = self.location_locks
                device.barrier = self.barrier
                device.setup_done.set() # Signal followers that setup is complete.
        else:
            self.setup_done.wait() # Followers wait for the leader.

    def assign_script(self, script, location):
        """
        Assigns a script to the device. The script is added to a local list and
        also placed on the worker queue if the main thread is ready.
        """
        if script is not None:
            # Dynamically create a lock for a location if it's the first time seeing it.
            if location not in self.location_locks:
                self.location_locks[location] = Lock()
            self.scripts.append((script, location))
            # If workers are already waiting for queue items, put it directly.
            if self.scripts_already_parsed.is_set():
                self.queue.put((script, location))
        else:
            # A None script signals the end of script assignment for this step.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread and its worker pool."""
        self.thread.join()
        # Send a "poison pill" for each worker to terminate it.
        for _ in xrange(len(self.workers)):
            self.queue.put((None, None))
        for worker in self.workers:
            worker.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device, orchestrating the simulation steps.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        while True:
            self.device.scripts_already_parsed.clear()

            # Get the current list of neighbors.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                break # End of simulation.

            # Producer step: Put all accumulated scripts for this step onto the queue.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location))
            self.device.scripts_already_parsed.set()

            # Wait for script assignment to finish for this step.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            
            # Wait for all workers to finish processing all items for this step.
            self.device.queue.join()
            
            # Synchronize with all other devices before proceeding to the next step.
            self.device.barrier.wait()


class Worker(Thread):
    """
    A persistent worker thread that consumes tasks from the shared queue.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Worker")
        self.device = device

    def run(self):
        """Continuously processes tasks from the queue until a poison pill is received."""
        while True:
            (script, location) = self.device.queue.get()
            if script is None:
                break # Poison pill received, terminate.

            # Use a 'with' statement for safe lock acquisition and release.
            with self.device.location_locks[location]:
                script_data = []
                
                # Gather data from neighbors.
                for n_device in self.device.neighbours:
                    data = n_device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from the parent device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Execute the script with the collected data.
                    result = script.run(script_data)
                    # Propagate the result to the parent device and all its neighbors.
                    for n_device in self.device.neighbours:
                        n_device.set_data(location, result)
                    self.device.set_data(location, result)

            self.device.queue.task_done()
