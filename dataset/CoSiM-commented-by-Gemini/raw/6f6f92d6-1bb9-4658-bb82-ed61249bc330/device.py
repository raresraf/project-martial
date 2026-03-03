"""
This module implements a distributed device simulation using a classic
producer-consumer pattern with a persistent worker pool.

Architectural Overview:
- `Device`: Represents a device node. It holds a task queue (`script_queue`)
  for its workers. Device 0 is responsible for creating and distributing shared
  synchronization objects.
- `DeviceThread`: This class serves a dual role. Its `run` method acts as the
  main control loop and the "producer" that adds tasks to the queue for each
  simulation step. Its `script_compute` method serves as the "consumer" logic
  for the worker threads in its pool.
- `ScriptObject`: A simple data class used to pass tasks (and shutdown signals)
  to the workers via the queue.
- Worker Pool: The `DeviceThread` initializes a fixed-size pool of persistent
  threads that all run the `script_compute` method.
- Synchronization:
  - A `ReusableBarrierCond` (imported) is used for inter-device synchronization.
  - A `Queue` is used to safely distribute tasks from the main thread to the
    worker threads.
  - A shared dictionary of fine-grained `Lock`s is used for data access,
    though its initialization is not thread-safe.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
import Queue

class Device(object):
    """
    Represents a device in the simulation. Manages a task queue and a pool of
    worker threads managed by its main `DeviceThread`.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        # A coarse-grained lock protecting this device's sensor_data dictionary.
        self.is_available = Lock()
        self.neighbours = []
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.script_queue = Queue.Queue()
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization objects. Device 0 creates a shared
        barrier and a shared dictionary for location locks.
        """
        if self.device_id == 0:
            shared_barrier = ReusableBarrierCond(len(devices))
            location_lock = {}
            for device in devices:
                device.shared_barrier = shared_barrier
                device.location_lock = location_lock

    def assign_script(self, script, location):
        """Assigns a script to the device's workload for the current step."""
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

class ScriptObject(object):
    """A wrapper class for tasks placed in the queue."""
    def __init__(self, script, location, stop_execution):
        self.script = script
        self.location = location
        self.stop_execution = stop_execution

class DeviceThread(Thread):
    """
    Main control thread for a device. It starts a pool of worker threads
    and then acts as a producer, feeding tasks into a shared queue.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None
        # Create and start the persistent worker pool.
        self.threads = []
        for _ in range(8):
            worker_thread = Thread(target=self.script_compute)
            self.threads.append(worker_thread)
            worker_thread.start()

    def script_compute(self):
        """
        The target function for worker threads (the "consumer" logic).
        Continuously fetches tasks from the queue and executes them.
        """
        while True:
            script_object = self.device.script_queue.get()
            if script_object.stop_execution:
                self.device.script_queue.task_done()
                break # Shutdown signal received.

            script = script_object.script
            location = script_object.location
            
            # Lazy initialization of location lock.
            # WARNING: This check-then-create pattern is not atomic and can
            # lead to a race condition if multiple workers try to create a
            # lock for the same new location simultaneously.
            if location not in self.device.location_lock:
                self.device.location_lock[location] = Lock()
            
            with self.device.location_lock[location]:
                script_data = [data for device in self.neighbours 
                               if (data := device.get_data(location)) is not None]
                if (data := self.device.get_data(location)) is not None:
                    script_data.append(data)
                
                if script_data:
                    result = script.run(script_data)
                    # The worker updates its own device's data directly.
                    with self.device.is_available:
                        if location in self.device.sensor_data:
                            self.device.sensor_data[location] = result
                    # It then calls the public set_data method for neighbors.
                    for device in self.neighbours:
                        device.set_data(location, result)
            
            self.device.script_queue.task_done()

    def run(self):
        """
        The main control loop (the "producer" logic).
        """
        while True:
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                # Send shutdown signals to all worker threads.
                for _ in range(8):
                    self.device.script_queue.put(ScriptObject(None, None, True))
                self.stop_all_threads()
                break

            # Wait for supervisor to finish assigning all scripts.
            self.device.timepoint_done.wait()
            
            # Add all assigned scripts for this step into the work queue.
            for (script, location) in self.device.scripts:
                self.device.script_queue.put(ScriptObject(script, location, False))

            # Wait for the worker pool to process all items in the queue.
            self.device.script_queue.join()
            
            # Synchronize with all other devices before the next step.
            self.device.shared_barrier.wait()
            self.device.timepoint_done.clear()

    def stop_all_threads(self):
        """Waits for all worker threads to terminate."""
        for thread in self.threads:
            thread.join()
        self.threads = []
