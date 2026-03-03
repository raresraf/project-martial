"""
This module implements a distributed device simulation using a classic
producer-consumer pattern, where each device manages a persistent worker pool.

Architectural Overview:
- `Device`: Represents a device node. It holds a task queue for its workers.
  Device 0 is responsible for creating and distributing shared synchronization
  objects like a barrier and a dictionary for location-based locks.
- `DeviceThread`: This class serves a dual role. Its `run` method acts as the
  main control loop (the "producer") that adds tasks to a queue. Its
  `do_in_parallel` method serves as the logic for the worker threads (the
  "consumers").
- Worker Pool: The `DeviceThread` initializes a fixed-size pool of persistent
  threads that all run the `do_in_parallel` method, consuming tasks from the
  shared queue.
- Synchronization:
  - An imported `ReusableBarrierCond` is used for inter-device synchronization.
  - A `Queue` is used to safely distribute tasks from the main thread to the
    worker threads.
  - A shared dictionary of fine-grained `Lock`s is used for data access,
    though its lazy initialization in `lock_location` is not thread-safe.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
import Queue

class Device(object):
    """
    Represents a device in the simulation. Manages its own state and a main
    `DeviceThread` which in turn manages a worker pool.
    """
    
    def set_shared_barrier(self, shared_barrier):
        """Setter for the shared inter-device barrier."""
        self.shared_barrier = shared_barrier

    def set_shared_location_locks(self, shared_location_locks):
        """Setter for the shared dictionary of location locks."""
        self.shared_location_locks = shared_location_locks

    def lock_location(self, location):
        """
        Acquires a lock for a specific location, creating it if it doesn't exist.

        WARNING: The check for the lock's existence and its creation are not
        atomic. This can lead to a race condition if multiple threads try to
        create a lock for the same new location simultaneously.
        """
        if location not in self.shared_location_locks:
            self.shared_location_locks[location] = Lock()
        self.shared_location_locks[location].acquire()

    def release_location(self, location):
        """Releases the lock for a specific location."""
        self.shared_location_locks[location].release()

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        # Coarse-grained lock for this device's sensor data dictionary.
        self.is_available = Lock()
        self.neighbours = []
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects. Device 0
        is responsible for creating the objects.
        """
        if self.device_id == 0:
            shared_barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                device.set_shared_barrier(shared_barrier)

        if self.device_id == 0:
            shared_location_locks = {}
            for device in devices:
                device.set_shared_location_locks(shared_location_locks)
        
    def assign_script(self, script, location):
        """Assigns a script to the device's workload."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
        self.script_received.set()

    def get_data(self, location):
        """Safely retrieves data from the local sensor dictionary."""
        with self.is_available:
            return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Safely sets data in the local sensor dictionary."""
        with self.is_available:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread and its workers to complete."""
        self.thread.join()

class DeviceThread(Thread):
    """
    Main control thread for a device and manager of its worker pool.
    """

    def do_in_parallel(self):
        """
        The target function for worker threads (the "consumer" logic).
        Continuously fetches tasks from the queue and executes them.
        """
        while True:
            args = self.queue.get()
            script = args["script"]
            location = args["location"]

            if script is None: # Sentinel value for shutdown.
                self.queue.task_done()
                break

            self.device.lock_location(location)

            # Gather data from neighbors and self.
            script_data = []
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Execute script and update data.
            if script_data:
                result = script.run(script_data)
                # This block mixes direct access and method calls for updates.
                with self.device.is_available:
                    if location in self.device.sensor_data:
                        self.device.sensor_data[location] = result
                for device in self.neighbours:
                    device.set_data(location, result)

            self.device.release_location(location)
            self.queue.task_done()

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue.Queue()
        self.threads = []
        self.neighbours = None
        # Create and start the persistent worker pool.
        for _ in range(8):
            worker_thread = Thread(target=self.do_in_parallel)
            worker_thread.daemon = True
            self.threads.append(worker_thread)
            worker_thread.start()

    def run(self):
        """
        The main control loop (the "producer" logic).
        """
        while True:
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                # Send shutdown signals to all worker threads.
                for _ in range(8):
                    self.queue.put({"script":None, "location":None})
                for thread in self.threads:
                    thread.join()
                self.threads = []
                break

            # Wait for supervisor to finish assigning all scripts.
            self.device.timepoint_done.wait()
            
            # Add all assigned scripts for this step into the work queue.
            for (script, location) in self.device.scripts:
                self.queue.put({"script":script, "location": location})

            # Wait for the worker pool to process all items in the queue.
            self.queue.join()
            
            # Synchronize with all other devices before the next step.
            self.device.shared_barrier.wait()
            self.device.timepoint_done.clear()
