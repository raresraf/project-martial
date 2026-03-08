# -*- coding: utf-8 -*-
"""
Models a distributed system of concurrent devices using a producer-consumer
pattern with a persistent worker pool.

NOTE: This implementation contains a critically flawed distributed locking mechanism
that is prone to deadlocks and runtime errors.
"""

import threading
from threading import Thread
from Queue import Queue
from cond_barrier import ReusableBarrier


class Device(object):
    """
    Represents a device that manages a pool of worker threads to process scripts.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): The device's local sensor data.
        location_locks (dict): A dictionary mapping locations to device-local locks.
        supervisor (object): A reference to the central supervisor.
        scripts_queue (Queue): A queue to receive scripts from the supervisor.
        thread (DeviceThread): The main producer thread for this device.
        barrier (ReusableBarrier): A shared barrier for global time step synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device, its locks, and its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        # Each device owns its own set of locks for its locations.
        self.location_locks = {loc: threading.Lock() for loc in self.sensor_data}
        self.supervisor = supervisor
        self.scripts = []  # Stores scripts for reprocessing, a likely bug.

        self.thread = DeviceThread(self)
        self.thread.start()

        self.scripts_queue = Queue()
        # This queue is created but immediately shadowed in DeviceThread.run.
        self.workers_queue = Queue()

        self.barrier = None


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up a shared barrier for all devices."""
        if self.device_id == 0: # Device 0 acts as leader.
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = barrier

    def assign_script(self, script, location):
        """Receives a script from the supervisor and puts it on a queue."""
        self.scripts_queue.put((script, location))

    def get_data(self, location):
        """Non-locking method to get data."""
        return self.sensor_data.get(location)

    def get_data_synchronize(self, location):
        """
        FLAWED LOCKING: Acquires a lock on this device's location.
        This is part of a broken distributed lock pattern.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """Non-locking method to set data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def set_data_synchronize(self, location, data):
        """
        FLAWED LOCKING: Releases a lock on this device's location.
        This is part of a broken distributed lock pattern, as the thread releasing
        the lock may not be the same one that acquired it across different objects.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main producer thread for a device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.stop = False

    def run(self):
        """Main loop: transfers scripts to a worker queue and synchronizes."""
        num_workers = 16
        workers = []
        workers_queue = Queue()  # This is the actual queue used by workers.

        # Create and start a persistent pool of worker threads.
        for i in range(num_workers):
            workers.append(WorkerThread(self.device, i, workers_queue))
        for worker in workers:
            worker.start()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # The device should not be its own neighbor.
            neighbours = [x for x in neighbours if x != self.device]
            for worker in workers:
                worker.neighbours = neighbours

            # BUG: This re-queues scripts from the previous time step.
            for script in self.device.scripts:
                workers_queue.put(script)

            # Transfer scripts from the supervisor queue to the worker queue.
            while True:
                script, location = self.device.scripts_queue.get()
                if script is None:  # Sentinel for end of time step.
                    break
                
                self.device.scripts.append((script, location)) # Save for next loop (bug).
                workers_queue.put((script, location))

            # Wait for all items in the worker queue to be processed.
            workers_queue.join()
            
            # Synchronize with all other devices.
            self.device.barrier.wait()

        # Shutdown sequence for worker threads.
        for _ in workers:
            workers_queue.put((None, None))
        for worker in workers:
            worker.join()


class WorkerThread(Thread):
    """A consumer thread that processes scripts from the worker queue."""
    def __init__(self, device, worker_id, queue):
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.device = device
        self.neighbours = []
        self.worker_id = worker_id
        self.queue = queue

    def run(self):
        """Main consumer loop."""
        while True:
            script, location = self.queue.get()
            if script is None: # Shutdown sentinel.
                self.queue.task_done()
                break

            # --- FLAWED DISTRIBUTED LOCKING ---
            # The following block attempts to create a distributed lock by having
            # this single thread acquire locks on multiple different device objects.
            # This is not a correct or safe way to implement this pattern.
            script_data = []
            
            # Acquire locks on all neighbor devices.
            for device in self.neighbours:
                data = device.get_data_synchronize(location)
                if data is not None:
                    script_data.append(data)
            
            # Acquire lock on the local device.
            data = self.device.get_data_synchronize(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                result = script.run(script_data)

                # Release locks on all neighbor devices.
                for device in self.neighbours:
                    device.set_data_synchronize(location, result)
                
                # Release lock on the local device.
                self.device.set_data_synchronize(location, result)
            
            # Signal that this task from the queue is complete.
            self.queue.task_done()
