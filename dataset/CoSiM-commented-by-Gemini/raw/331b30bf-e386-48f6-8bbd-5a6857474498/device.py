"""
A simulation framework for a network of devices using a worker thread pool
and a central work queue.

This module implements a device simulation where a master device initializes
shared resources, including a dictionary of locks for data locations and a
barrier. Each device's main thread then spawns a pool of worker threads for
each time step. These workers fetch tasks from a shared queue. The implementation
contains a logical flaw where completed tasks are re-added to the queue,
leading to an infinite processing loop and deadlock.
"""

from threading import Thread, Lock
from barrier import ReusableBarrierCond
from Queue import Queue


class Device(object):
    """
    Represents a single device that manages a queue of scripts for worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's internal sensor data.
            supervisor: The central supervisor managing the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.queue = Queue()
        self.num_threads = 8

        self.location_locks = None
        self.lock = None
        self.barrier = None

        self.thread = None

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources for all devices.

        The device with `device_id == 0` acts as a master, creating the shared
        `location_locks` dictionary, a global lock, and a barrier. These are
        then assigned to all other devices. Each device's main thread is
        started only after this setup is complete.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            self.location_locks = {}
            self.lock = Lock()
            self.barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.location_locks = self.location_locks
                    device.lock = self.lock
                    device.barrier = self.barrier
        self.thread = DeviceThread(self)
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script by adding it to the central work queue.

        When the supervisor signals the end of assignments by sending `None`,
        this method places a "poison pill" (a `(None, None)` tuple) on the
        queue for each worker thread to find, signaling them to terminate.

        Args:
            script: The script object to execute.
            location: The data location the script will operate on.
        """
        if script is not None:
            # Block Logic: A global lock protects the dynamic creation of new
            # locks for previously unseen locations.
            with self.lock:
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()
            self.queue.put((script, location))
        else:
            # Add a poison pill for each worker thread to consume.
            for _ in range(self.num_threads):
                self.queue.put((None, None))

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[
            location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device.

    At each timepoint, this thread creates, starts, and joins a pool of
    `WorkerThread`s, which are responsible for the actual script execution.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self)
        self.device = device

    def run(self):
        """The main execution loop, organized into discrete timepoints."""
        while True:
            # Get neighbours for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Block Logic: Create a new pool of worker threads for each timepoint.
            # This is an inefficient pattern but encapsulates the work.
            worker_threads = [WorkerThread(self.device, neighbours) for _ in
                              range(self.device.num_threads)]

            for thread in worker_threads:
                thread.start()
            # Wait for all worker threads to consume their poison pills and terminate.
            for thread in worker_threads:
                thread.join()

            # Invariant: Synchronize with all other devices before the next timepoint.
            self.device.barrier.wait()


class WorkerThread(Thread):
    """
    A worker thread that executes scripts from a shared queue.
    """

    def __init__(self, device, neighbours):
        """
        Initializes the WorkerThread.
        
        Args:
            device (Device): The parent device, which holds the work queue.
            neighbours (list): A list of neighbouring devices.
        """
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours

    def run_script(self, script, location):
        """
        Executes a single script, including data aggregation and result broadcast.
        
        Args:
            script: The script to execute.
            location: The location to operate on.
        """
        script_data = []
        
        # Aggregate data from neighbours and self.
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        
        # Invariant: Script only runs if there is data to process.
        if script_data:
            result = script.run(script_data)

            # Broadcast the result to all participants.
            for device in self.neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result)

    def run(self):
        """
        The main loop for the worker.
        
        Fetches a script from the queue, executes it under a location-specific
        lock, and then re-queues the same script, causing an infinite loop.
        The thread only terminates upon fetching a `None` script (poison pill).
        """
        while True:
            script, location = self.device.queue.get()
            # The "poison pill" tuple (None, None) signals termination.
            if script is None:
                return
            # Pre-condition: Acquire lock for the specific location before processing.
            with self.device.location_locks[location]:
                self.run_script(script, location)
            # CRITICAL: The script is put back on the queue after execution,
            # which will cause the workers to process the same scripts indefinitely
            # and never reach the poison pills, resulting in a deadlock.
            self.device.queue.put((script, location))