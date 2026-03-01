"""
@file device.py
@brief A distributed device simulation featuring a custom thread pool for concurrency.

This script models a network of devices operating in synchronized time steps. Each
device uses a single management thread that delegates parallel script execution to a
custom-built, queue-based thread pool. Synchronization across devices is
managed by a shared barrier.
"""

from threading import Event, Thread, Lock
# The 'barrier' and 'threadpool' modules are assumed to be part of the project.
from barrier import Barrier
from threadpool import ThreadPool

class Device(object):
    """
    Represents a node in the distributed network. It manages its own state and
    a single control thread (`DeviceThread`) which in turn manages a thread pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device.
        @param device_id A unique identifier.
        @param sensor_data A dictionary of the device's local data.
        @param supervisor A reference to the central supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Events to coordinate the main device thread.
        self.script_received = Event()
        self.timepoint_done = Event()
        self.scripts = []
        
        # The main control thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

        # A shared barrier to synchronize with other devices.
        self.barrier = None
        # A dictionary of locks, one for each data location this device owns.
        self.locks = {location : Lock() for location in sensor_data}

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes the shared barrier, orchestrated by device 0."""
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A None script signals the end of
        assignments for the current timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()
            # Also set script_received to unblock the main thread.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. It manages the device's lifecycle and
    delegates work to a custom ThreadPool.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Each device has its own thread pool for parallel script execution.
        self.thread_pool = ThreadPool(7, device)

    def run(self):
        """Main execution loop, processing timepoints and managing script execution."""
        # This outer loop represents the progression through discrete timepoints.
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal from supervisor.

            # Wait until scripts start arriving and until all scripts for the timepoint are assigned.
            self.device.script_received.wait()
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Submit all assigned scripts to the thread pool for execution.
            for (script, location) in self.device.scripts:
                self.thread_pool.submit(neighbours, script, location)

            # @bug Synchronization Flaw: The thread proceeds to the global barrier
            # immediately after submitting tasks, without waiting for the thread pool
            # to finish processing them. This creates a race condition. A call like
            # `self.thread_pool.queue.join()` is needed here to wait for completion.
            self.device.barrier.wait()

        self.thread_pool.shutdown()


from Queue import Queue
from threading import Thread

class ThreadPool(object):
    """
    A custom, queue-based thread pool implementation for executing tasks.
    """

    def __init__(self, threads_count, device):
        """
        Initializes the thread pool and starts the worker threads.
        @param threads_count The number of worker threads.
        @param device The parent device, used for context in script execution.
        """
        self.queue = Queue(threads_count)
        self.threads = []
        self.device = device

        # Create and start the worker threads.
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)
            new_thread.start()

    def execute(self):
        """The main loop for a worker thread."""
        while True:
            # Block until a task is available in the queue.
            now = self.queue.get()
            # A 'None' task is a "poison pill" signal to terminate the thread.
            if now is None:
                self.queue.task_done()
                return

            self.run_script(now)
            self.queue.task_done()

    def run_script(self, script_env_data):
        """
        Executes a single script, managing distributed locking.
        @warning This function's distributed locking pattern (acquiring locks on
        other devices) carries a high risk of deadlock.
        """
        neighbours, script, location = script_env_data
        script_data = []

        # Block Logic: Acquire locks on neighbors.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                if location in device.sensor_data:
                    device.locks[location].acquire()
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Block Logic: Acquire lock on self.
        if location in self.device.sensor_data:
            self.device.locks[location].acquire()
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        # If data was collected, run script and disseminate results.
        if script_data != []:
            result = script.run(script_data)

            # Release locks on neighbors.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)
                    if location in device.sensor_data:
                        device.locks[location].release()

            # Release lock on self.
            self.device.set_data(location, result)
            if location in self.device.sensor_data:
                self.device.locks[location].release()

    def submit(self, neighbours, script, location):
        """Adds a new script execution task to the queue."""
        self.queue.put((neighbours, script, location))

    def shutdown(self):
        """Gracefully shuts down the thread pool."""
        # Wait for all tasks in the queue to be completed.
        self.queue.join()

        # Send a "poison pill" for each thread to signal termination.
        for _ in self.threads:
            self.queue.put(None)

        # Wait for all worker threads to finish.
        for thread in self.threads:
            thread.join()
