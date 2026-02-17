
"""
Models a distributed device simulation using a producer-consumer pattern.

This module, written for Python 2, implements a device simulation framework
that uses a classic producer-consumer architecture with a `Queue` for work
distribution. Each device maintains a fixed-size pool of persistent worker
threads, providing an efficient model that avoids thread creation overhead
on each cycle.
"""

from threading import Event, Thread, Lock
from Queue import Queue
from barrier import ReusableBarrierSem

class Device(object):
    """Represents a single device that processes scripts via a worker pool.

    This device acts as the central manager for its own set of worker threads
    and the queue used to feed them tasks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device, its work queue, and main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = []
        self.threads = [] # Pool of worker threads
        self.barrier = None
        self.timepoint_done = Event()
        self.thread_queue = Queue() # Producer-consumer queue for tasks
        self.locks = {} # Dictionary of locks, keyed by location
        self.thread = DeviceThread(self)
        self.thread.start()


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Creates the worker thread pool and sets up shared synchronization objects.

        This method is intended to be called on one device, which then
        distributes the shared barrier object to all other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Create and start a fixed pool of 8 persistent worker threads.
        for _ in range(8):
            thread = Worker(self)
            thread.start()
            self.threads.append(thread)
        
        for device in devices:
            if device is not None:
                self.devices.append(device)
        
        # Create and distribute the shared barrier if it doesn't exist.
        if self.barrier is None:
            self.barrier = ReusableBarrierSem(len(self.devices))
        
        for device in self.devices:
            if device is not None:
                if device.barrier is None:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to the device.

        This method also ensures that a lock exists for the script's location
        and is shared among all devices.
        """
        if script is not None:
            self.scripts.append((script, location))
            
            # Lazily create a lock for a location if not already present.
            if location is not None:
                if not self.locks.has_key(location):
                    self.locks[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

        # Ensure all devices are aware of the same lock for a given location.
        for device in self.devices:
            if not device.locks.has_key(location):
                if self.locks.has_key(location):
                    device.locks[location] = self.locks[location]

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device and its worker threads."""
        self.thread.join()


class Worker(Thread):
    """A persistent worker thread that consumes tasks from a queue."""

    def __init__(self, device):
        """Initializes the worker."""
        Thread.__init__(self)
        self.device = device

    def run(self):
        """Main loop of the worker, consuming tasks from the queue."""
        while True:
            # Block until a task is available on the queue.
            script_loc_neigh = self.device.thread_queue.get()
            
            # Shutdown Signal: A "poison pill" tuple (None, None, None) indicates
            # that the worker should terminate.
            if script_loc_neigh[0] is None:
                if script_loc_neigh[1] is None:
                    if script_loc_neigh[2] is None:
                        self.device.thread_queue.task_done()
                        break
            
            script_data = []
            
            # Acquire the lock for the specific location to ensure data consistency.
            self.device.locks[script_loc_neigh[1]].acquire()

            # Aggregate data from neighbor devices.
            for device in script_loc_neigh[2]:
                data = device.get_data(script_loc_neigh[1])
                if data is not None:
                    script_data.append(data)

            # Aggregate data from the worker's own device.
            data = self.device.get_data(script_loc_neigh[1])
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Execute the script with the aggregated data.
                result = script_loc_neigh[0].run(script_data)
                
                # Disseminate the result to all involved devices.
                for device in script_loc_neigh[2]:
                    device.set_data(script_loc_neigh[1], result)
                self.device.set_data(script_loc_neigh[1], result)
            
            # Release the lock for the location.
            self.device.locks[script_loc_neigh[1]].release()
            # Signal to the queue that this task is complete.
            self.device.thread_queue.task_done()


class DeviceThread(Thread):
    """The main control thread (producer) for a Device."""

    def __init__(self, device):
        """Initializes the producer thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The producer loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for the supervisor to signal that all scripts for the
            # current timepoint have been assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            
            # Act as the producer: add all script tasks to the work queue.
            for (script, location) in self.device.scripts:
                self.device.thread_queue.put((script, location, neighbours))
            
            # Block until all items in the queue have been processed by the
            # consumer (worker) threads.
            self.device.thread_queue.join()
            
            # Synchronize with all other devices before starting the next timepoint.
            self.device.barrier.wait()
        
        # --- Shutdown Sequence ---
        # Send a "poison pill" for each worker thread to signal termination.
        for _ in range(len(self.device.threads)):
            self.device.thread_queue.put((None, None, None))
        # Wait for all worker threads to finish.
        for thread in self.device.threads:
            thread.join()
