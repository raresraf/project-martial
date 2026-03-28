"""
Defines a device simulation model using a two-stage queue and a thread pool.

This module implements a producer-consumer pattern where a `DeviceThread` acts
as a dispatcher, moving jobs from a supervisor-facing queue to an internal
worker queue, which is then serviced by a pool of `WorkerThread` consumers.
Synchronization relies on `Queue.join()`, a global barrier, and a custom,
potentially hazardous, cross-device locking mechanism.
"""

import threading
from threading import Thread
from Queue import Queue
from cond_barrier import ReusableBarrier


class Device(object):
    """
    Represents a device node with a two-queue system for script processing.

    This device receives scripts from a supervisor on one queue and has its main
    thread (`DeviceThread`) forward them to an internal worker queue. It manages
    its own set of locks for its data locations.

    Attributes:
        device_id (int): Unique ID for the device.
        sensor_data (dict): The device's local sensor data.
        location_locks (dict): A dictionary of locks for this device's data locations.
        supervisor: The simulation supervisor.
        scripts_queue (Queue): A queue where the supervisor places incoming scripts.
        thread (DeviceThread): The manager thread for this device.
        barrier (ReusableBarrier): A global barrier to synchronize all devices.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device, its locks, queues, and manager thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.location_locks = {loc: threading.Lock() for loc in self.sensor_data}
        self.supervisor = supervisor
        self.scripts = []
        self.scripts_queue = Queue()
        self.workers_queue = Queue() # This seems unused here, managed in DeviceThread.
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared resources; Device 0 creates and distributes the barrier."""
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = barrier

    def assign_script(self, script, location):
        """Called by the supervisor to add a script to the processing queue."""
        self.scripts_queue.put((script, location))

    def get_data(self, location):
        """Performs a non-blocking read of sensor data."""
        return self.sensor_data.get(location)

    def get_data_synchronize(self, location):
        """
        Acquires a lock and then reads sensor data.

        Warning: This method acquires a lock that is expected to be released
        externally by `set_data_synchronize`. This is a fragile pattern.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """Performs a non-blocking write of sensor data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def set_data_synchronize(self, location, data):
        """
        Writes sensor data and releases a lock.

        Warning: This method releases a lock that it did not acquire. It is
        intended to be paired with a call to `get_data_synchronize`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()

    def shutdown(self):
        """Shuts down the device by joining its manager thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    A manager thread that dispatches jobs from a supervisor queue to a worker queue.
    """
    def __init__(self, device):
        """Initializes the manager thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.stop = False

    def run(self):
        """
        Main loop for the manager thread.
        
        It creates and starts a pool of workers, then enters a loop to forward
        jobs from the supervisor's queue to the workers' queue for each timepoint.
        """
        num_workers = 16
        workers = []
        workers_queue = Queue()

        # Create and start the pool of worker threads.
        for i in range(num_workers):
            workers.append(WorkerThread(self.device, i, workers_queue))
        for worker in workers:
            worker.start()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Shutdown signal
                break
            
            # Update workers with the current list of neighbors.
            neighbours = [x for x in neighbours if x != self.device]
            for worker in workers:
                worker.neighbours = neighbours
            
            # This appears to be leftover logic; scripts are now handled by the queue.
            for script in self.device.scripts:
                workers_queue.put(script)

            # Consume scripts from the supervisor queue and forward them to the worker queue.
            while True:
                script, location = self.device.scripts_queue.get()
                if script is None: # End of timepoint signal from supervisor.
                    break
                self.device.scripts.append((script, location))
                workers_queue.put((script, location))

            # Wait for all workers to complete all jobs for this timepoint.
            workers_queue.join()
            # Wait at the global barrier for all other devices to finish.
            self.device.barrier.wait()

        # Initiate clean shutdown of the worker pool.
        for worker in workers:
            workers_queue.put((None, None)) # Sentinel value
        for worker in workers:
            worker.join()


class WorkerThread(Thread):
    """
    A worker thread that consumes and processes scripts from a queue.
    """
    def __init__(self, device, worker_id, queue):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.device = device
        self.neighbours = []
        self.worker_id = worker_id
        self.queue = queue

    def run(self):
        """
        Main loop for the worker thread.
        
        Continuously pulls jobs from the queue, executes them, and signals completion.
        The loop terminates when it receives a (None, None) sentinel value.
        """
        while True:
            script, location = self.queue.get()
            if script is None: # Shutdown sentinel
                self.queue.task_done()
                break

            script_data = []
            
            # Acquire locks and gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data_synchronize(location)
                if data is not None:
                    script_data.append(data)
            
            # Acquire lock and gather data from the local device.
            data = self.device.get_data_synchronize(location)
            if data is not None:
                script_data.append(data)

            # If data was collected, run the script and update data, releasing locks.
            if script_data:
                result = script.run(script_data)
                
                # Update neighbors, which also releases their locks.
                for device in self.neighbours:
                    device.set_data_synchronize(location, result)
                # Update local device, releasing its lock.
                self.device.set_data_synchronize(location, result)

            self.queue.task_done()
