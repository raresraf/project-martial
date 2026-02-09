


"""
This module defines a distributed device simulation framework that uses a
producer-consumer model with a thread pool to execute scripts.

It features a `Device` class that acts as a producer, adding scripts to a shared
queue. A pool of `WorkerThread` instances act as consumers, processing scripts
from the queue. A `DeviceThread` manages the lifecycle of the worker threads.
"""

from threading import Thread, Lock
from barrier import ReusableBarrierCond
from Queue import Queue


class Device(object):
    """
    Represents a device in a distributed network that can process sensor data.

    This device acts as a central point for a pool of worker threads, assigning
    scripts to them via a shared queue. It is responsible for setting up the
    synchronization primitives (barrier and locks) used by the workers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of sensor data, keyed by location.
            supervisor (Supervisor): A supervisor object that manages the network.
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
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared synchronization objects for all devices.
        Only the device with ID 0 is responsible for creating these objects.

        Args:
            devices (list): A list of all devices in the network.
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
        Assigns a script to be processed by the worker threads by adding it to the queue.

        Args:
            script (Script): The script object to execute.
            location (int): The location identifier associated with the script's data.
        """
        
        if script is not None:
            with self.lock:
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()
            self.queue.put((script, location))
        else:
            for _ in range(self.num_threads):
                self.queue.put((None, None))

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location to retrieve data for.

        Returns:
            The sensor data, or None if the location is not found.
        """
        
        return self.sensor_data[
            location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location.

        Args:
            location (int): The location to update data for.
            data: The new data value.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a device, responsible for managing worker threads.
    """

    def __init__(self, device):
        """
        Initializes the main device thread.

        Args:
            device (Device): The device this thread belongs to.
        """
        
        Thread.__init__(self)


        self.device = device

    def run(self):
        """
        The main execution loop.

        This loop creates and starts a pool of worker threads, waits for them to
        complete their work for the current timepoint, and then synchronizes
        with other devices at a global barrier.
        """
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            worker_threads = [WorkerThread(self.device, neighbours) for _ in
                              range(self.device.num_threads)]


            for thread in worker_threads:
                thread.start()
            for thread in worker_threads:
                thread.join()

            self.device.barrier.wait()


class WorkerThread(Thread):
    """
    A worker thread that consumes scripts from a queue and executes them.
    """

    def __init__(self, device, neighbours):
        """
        Initializes a worker thread.

        Args:
            device (Device): The parent device, which holds the script queue.
            neighbours (list): A list of neighboring devices.
        """
        
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours

    def run_script(self, script, location):
        """
        Executes a single script.

        Args:
            script (Script): The script to execute.
            location (int): The data location for the script.
        """
        
        script_data = []
        
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            
            result = script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result)

    def run(self):
        """
        The main loop for a worker thread.

        It continuously fetches scripts from the device's queue, executes them
        under a location-specific lock, and then re-queues them. The loop
        terminates when it receives a `None` script (a poison pill).
        """
        
        while True:
            script, location = self.device.queue.get()
            if script is None:
                return
            with self.device.location_locks[location]:
                self.run_script(script, location)
            self.device.queue.put((script, location))
