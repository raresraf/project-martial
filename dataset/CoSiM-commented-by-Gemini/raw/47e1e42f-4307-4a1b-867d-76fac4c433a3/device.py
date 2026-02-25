"""
Models a device in a distributed simulation using a dedicated thread pool
for task management.

This module defines a simulation of networked devices where each device has its
own thread pool to process computational scripts. A master thread on each device
submits tasks to the pool. Synchronization across the entire system of devices
is managed by a reusable barrier.

Classes:
    Device: Represents a device, its data, and its thread pool.
    DeviceThread: The master control thread for a device.
    ThreadPool: A class that manages a queue and a pool of worker threads.
"""

from threading import Event, Thread, Lock

from reusable_barrier import ReusableBarrier
from thread_pool import ThreadPool

class Device(object):
    """
    Represents a single device, its sensor data, and its execution environment.

    Each device has a master thread and a dedicated thread pool for executing
    scripts. It manages its own locks for local sensor data locations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        self.barrier = None

        # Each device has its own set of locks for its sensor locations.
        self.locks = {}
        for location in sensor_data:
            self.locks[location] = Lock()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs a centralized setup, primarily to create and distribute a shared barrier.
        """
        
        nr_devices = len(devices)
        # Device 0 is responsible for creating the single shared barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(nr_devices)
            
            for device in devices:
                if device.device_id: # Assigns barrier to all other devices.
                    device.barrier = self.barrier
        

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A None script signals the end of assignments
        for the current timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data. Flawed implementation: acquires a lock but never releases it.
        """
        if location in self.sensor_data:
            self.locks[location].acquire() # This lock is never released.

        return self.sensor_data[location] if location in self.sensor_data else None


    def set_data(self, location, data):
        """
        Updates sensor data. Flawed implementation: releases a lock it does not hold.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release() # Releases a lock that was not acquired here.

    def shutdown(self):
        """Shuts down the device by joining its master thread."""
        self.thread.join()

class DeviceThread(Thread):
    """The master control thread for a device."""

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        # Each device gets its own pool of 8 worker threads.
        self.thread_pool = ThreadPool(8, self.device)

    def run(self):
        """
        The main control loop for the device.
        
        This loop waits for a timepoint to start, submits all assigned scripts
        to the thread pool, and then immediately waits at the system-wide barrier.
        """
        while True:

            # Get neighbors from the supervisor; None is the shutdown signal.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to finish assigning all scripts for the timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Remove self from neighbor list if present.
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    neighbours.remove(device)
                    break

            # Submit all assigned scripts to the thread pool for execution.
            for (script, location) in self.device.scripts:
                self.thread_pool.submit_task(neighbours, script, location)

            # Flawed Synchronization: The thread proceeds to the barrier immediately
            # after submitting tasks, without waiting for the thread pool to finish them.
            self.device.barrier.wait()

        # Signal the thread pool to shut down and wait for workers to terminate.
        self.thread_pool.join_threads()


from threading import Thread
from Queue import Queue

class ThreadPool(object):
    """
    A simple thread pool implementation to manage a queue of tasks for worker threads.
    """

    def __init__(self, threads_count, device):
        
        self.threads = []
        self.device = device
        self.threads_count = threads_count
        self.queue = Queue(threads_count)

        self.create_threads(threads_count)

    def create_threads(self, threads_count):
        """Creates and starts the pool of worker threads."""
        i = 0
        while i < threads_count:
            thread = Thread(target=self.execute_task)
            self.threads.append(thread)
            i += 1

        for thread in self.threads:
            thread.start()

    def submit_task(self, neighbours, script, location):
        """Adds a task to the work queue."""
        self.queue.put((neighbours, script, location))

    def execute_task(self):
        """Target function for worker threads; fetches and runs tasks from the queue."""

        while True:
            # Block until a task is available in the queue.
            elem = self.queue.get()
            neighbours = elem[0]
            script = elem[1]
            location = elem[2]

            # A 'None' task is a poison pill to terminate the thread.
            if neighbours is None and script is None and location is None:
                return

            self.run_script(neighbours, script, location)

    def run_script(self, neighbours, script, location):
        """
        The core logic for executing a single script.
        
        Gathers data, runs the script, and propagates the results.
        This method relies on the flawed get_data/set_data locking in the Device class.
        """
        script_data = []

        # Gather data from neighbors.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Gather data from the parent device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Run the script on the collected data.
            result = script.run(script_data)

            # Propagate results back to all devices.
            for device in neighbours:
                device.set_data(location, result)

            self.device.set_data(location, result)

    def join_threads(self):
        """Shuts down the thread pool by sending a 'poison pill' to each thread."""
        i = 0
        while i < self.threads_count:
            self.submit_task(None, None, None)
            i += 1

        for thread in self.threads:
            thread.join()