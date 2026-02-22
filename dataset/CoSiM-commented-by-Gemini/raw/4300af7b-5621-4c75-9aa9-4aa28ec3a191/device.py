"""
This file models a distributed sensor network simulation using a classic
producer-consumer pattern with a fixed-size thread pool.

The architecture is composed of:
- A `DeviceThread` (producer) for each device, which orchestrates the time step
  and places tasks on a shared queue.
- A pool of `MyThread` (consumer) workers for each device that continuously pull
  tasks from the queue and execute them.
- A thread-safe `Queue.Queue` to manage the distribution of tasks.
- A reusable barrier to synchronize all devices at the end of each time step.

Classes:
    Device: Represents a node, holding state and managing its threads.
    MyThread: A long-lived consumer thread that processes tasks from a queue.
    DeviceThread: The producer thread that manages the task queue and the overall
                  lifecycle of a simulation time step for one device.
"""


from threading import Event, Thread, Semaphore
from Queue import Queue
from barrier import ReusableBarrierCond # Assumed to be a correct reusable barrier


class Device(object):
    """Represents a single device node in the network."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device and starts its main producer thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Not used in this implementation's logic.
        self.scripts = []
        self.timepoint_done = Event() # Signals all scripts for a time step are assigned.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.locations = [] # Discovered data locations.
        self.location_locks = None # Shared dictionary of locks for each location.
        self.barrier = None # Shared barrier to sync all devices.

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (barrier, locks) to all devices.

        The first device to run this will discover all data locations, create the
        shared resources, and then assign them to all other uninitialized devices.
        """
        lock = {}
        barrier = ReusableBarrierCond(len(devices))

        if self.barrier is None:
            # Discover all unique locations from all devices.
            self.get_all_locations(devices)
            for location in self.locations:
                lock[location] = Semaphore(1)

            # Assign the created shared resources to all devices.
            for device in devices:
                if device.barrier is None and device.location_locks is None:
                    device.barrier = barrier
                    device.location_locks = lock

    def get_all_locations(self, devices):
        """Helper method to populate a list of all unique locations."""
        for device in devices:
            for location in device.sensor_data:
                if location not in self.locations:
                    self.locations.append(location)

    def assign_script(self, script, location):
        """Assigns a script for the upcoming time step."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class MyThread(Thread):
    """A long-lived consumer/worker thread."""

    def __init__(self, device, tasks):
        """
        Initializes the worker.
        Args:
            device (Device): The parent device instance.
            tasks (Queue): The shared task queue to pull work from.
        """
        Thread.__init__(self, name="MyThread %d" % device.device_id)
        self.device = device
        self.tasks = tasks

    def run(self):
        """Continuously fetches and executes tasks from the queue."""
        while True:
            # Block until a task is available on the queue.
            neighbours, script, location = self.tasks.get()

            # A `None` value is a "poison pill" used to signal termination.
            if neighbours is None:
                self.tasks.task_done()
                return

            # Acquire the specific lock for the data location.
            self.device.location_locks[location].acquire()

            # Aggregate data from neighbors and self.
            script_data = []
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
            
            # Run computation and disseminate results.
            if script_data != []:
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)

            self.device.location_locks[location].release()

            # Signal that this specific task is complete.
            self.tasks.task_done()

class DeviceThread(Thread):
    """The main producer/orchestrator thread for a device."""

    def __init__(self, device):
        """Initializes the thread and its pool of worker threads."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.max_threads = 8
        self.tasks = Queue(self.max_threads)

        self.thread_list = []
        for _ in range(self.max_threads):
            self.thread_list.append(MyThread(self.device, self.tasks))

    def run(self):
        """
        Main simulation loop: produces tasks and synchronizes time steps.
        """
        # Start all consumer threads in the pool. They will block waiting for tasks.
        for thread in self.thread_list:
            thread.start()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # --- Shutdown Sequence ---
                # Put a "poison pill" on the queue for each worker to consume.
                for _ in self.thread_list:
                    self.tasks.put((None, None, None))
                # Wait for all workers to finish consuming the poison pills.
                self.tasks.join()
                break

            # Wait for supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()

            # --- Producer Logic ---
            # Put all assigned script tasks onto the shared queue.
            for (script, location) in self.device.scripts:
                self.tasks.put((neighbours, script, location))

            # --- Synchronization ---
            # Block until consumers call `task_done()` for all items put on the queue.
            # This cleanly waits for all work in the time step to complete.
            self.tasks.join()

            # Synchronize with all other devices before the next time step.
            self.device.barrier.wait()

            # Reset for the next cycle.
            self.device.timepoint_done.clear()

        # Final cleanup of worker threads.
        for thread in self.thread_list:
            thread.join()
        self.thread_list = []
