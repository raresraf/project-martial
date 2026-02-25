"""
Models a device in a distributed simulation using a master-worker pattern
with a shared task queue.

This module defines a simulation of networked devices. Each device operates on
a master-worker model, where a single master thread distributes computational
tasks (scripts) to a fixed pool of worker threads via a thread-safe queue.
Synchronization across all devices is handled by a reusable barrier.

Classes:
    ReusableBarrier: A standard, condition-based reusable thread barrier.
    Device: The main class representing a device, its data, and its threads.
    DeviceThreadMaster: The control thread that manages timepoints and distributes work.
    DeviceThreadWorker: A worker thread that consumes tasks from the work queue.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue, Empty


class ReusableBarrier(object):
    """
    A standard implementation of a reusable barrier for thread synchronization.
    
    This barrier uses a `Condition` variable to block threads until the required
    number of threads have reached the barrier.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Blocks the calling thread until all participating threads have also called wait."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # The last thread arrives, notifies all others, and resets the barrier count.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    Represents a device, managing its data, scripts, and a master-worker thread pool.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.neighbours = []

        # Shared resources for master-worker communication and synchronization.
        self.barrier = None
        self.locks = []  # A list of locks for data locations.
        self.timepoint_done = Event()  # Set by supervisor when script assignment is done.
        self.tasks_ready = Event()  # Set by master to activate workers.
        self.tasks = Queue()  # Thread-safe queue for distributing scripts to workers.
        self.simulation_ended = False

        # Each device has one master thread and a pool of worker threads.
        self.master = DeviceThreadMaster(self)
        self.master.start()

        self.workers = []
        for i in xrange(8): # Creates a fixed-size pool of 8 worker threads.
            worker = DeviceThreadWorker(self, i)
            self.workers.append(worker)
            worker.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device [%d]" % self.device_id

    def setup_devices(self, devices):
        """
        Performs a centralized, one-time setup for all devices.

        This is intended to be run by one device (e.g., device 0) to initialize
        resources shared by all devices, such as the barrier and location locks.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            # Assumes a fixed number of 24 possible locations.
            locks = [Lock() for _ in xrange(24)]
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """
        Assigns a script to be processed in the current timepoint.

        Args:
            script (Script): The script to execute. If None, it signals the
                             end of script assignments for this timepoint.
            location (any): The data location for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts for this timepoint have been received.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates the sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its master and worker threads."""
        self.master.join()
        for worker in self.workers:
            worker.join()


class DeviceThreadMaster(Thread):
    """
    The master thread that orchestrates the device's activity in each time step.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device [%d] Thread Master" % device.device_id)
        self.device = device

    def run(self):
        """The main control loop for the device."""
        while True:
            # Get neighbors from the supervisor. If None, it's a shutdown signal.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                # Signal workers to terminate and then exit the master thread.
                self.device.simulation_ended = True
                self.device.tasks_ready.set()
                break

            # Wait until the supervisor has assigned all scripts for this timepoint.
            self.device.timepoint_done.wait()

            # Add all assigned scripts to the shared work queue.
            for task in self.device.scripts:
                self.device.tasks.put(task)

            # Signal to the worker threads that the queue is ready for processing.
            self.device.tasks_ready.set()

            # Block until the workers have processed all items in the queue.
            self.device.tasks.join()

            # Reset events for the next timepoint.
            self.device.tasks_ready.clear()
            self.device.timepoint_done.clear()

            # Wait at the barrier for all other devices to finish the timepoint.
            self.device.barrier.wait()


class DeviceThreadWorker(Thread):
    """
    A worker thread that consumes and processes tasks from a shared queue.
    """
    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device [%d] Thread Worker [%d]" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main loop for the worker; processes tasks until the simulation ends."""
        while not self.device.simulation_ended:
            # Wait for the master to signal that tasks are available.
            self.device.tasks_ready.wait()

            try:
                # Fetch a task from the queue without blocking.
                script, location = self.device.tasks.get(block=False)

                # Acquire the lock for the specific data location to ensure atomicity.
                self.device.locks[location].acquire()

                script_data = []

                # Gather data from neighboring devices.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Gather data from the parent device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                # Execute the script if any data was gathered.
                if len(script_data) > 0:
                    result = script.run(script_data)

                    # Propagate the result to all involved devices.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
                
                self.device.locks[location].release()

                # Signal to the queue that this task is complete.
                self.device.tasks.task_done()
            except Empty:
                # This occurs if the thread wakes up but the queue is already empty.
                pass