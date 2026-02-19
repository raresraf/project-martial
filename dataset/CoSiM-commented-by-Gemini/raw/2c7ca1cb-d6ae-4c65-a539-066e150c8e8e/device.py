"""
@file device.py
@brief A simulation of a distributed device network using a worker thread pool manager.
@details This module implements a producer-consumer-like pattern where a main device thread
dispatches script execution tasks to a pool of worker threads managed by a `WorkerManager`.
It uses a shared barrier for synchronization between devices and a dictionary of locks
for data locations.
"""

import Queue as q
from threading import Event, Thread, Lock
# Assumes the existence of a ReusableBarrier class in a "Barrier" module.
from Barrier import ReusableBarrier


class Device(object):
    """
    @brief Represents a single device in the simulation.
    @details Manages sensor data, script execution, and synchronization. It uses a
    `WorkerManager` to handle concurrent script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.setup_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.data_locks = None
        self.scripts_arrived = False

    def __str__(self):
        """@brief Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def assign_barrier(self, barrier):
        """@brief Assigns the shared barrier object to this device."""
        self.barrier = barrier

    def setup_devices(self, devices):
        """
        @brief Initializes shared resources for the network of devices.
        @details The device with ID 0 is responsible for creating a shared ReusableBarrier
        and distributing it to all other devices. Each device initializes its own data locks.
        """
        self.data_locks = {loc: Lock() for loc in self.sensor_data}

        # The root device (ID 0) creates and distributes the barrier.
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))

            for device in devices:
                device.assign_barrier(barrier)
        
        # Signal that this device has completed its setup.
        self.setup_done.set()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device or signals the end of a timepoint.
        """
        if script is not None:
            # Dynamically create a lock for a new location if it doesn't exist.
            if location not in self.data_locks:
                self.data_locks[location] = Lock()

            self.scripts.append((script, location))
            self.script_received.set()
            self.scripts_arrived = True
        else:
            # A None script signals that all scripts for the timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Acquires a lock and retrieves sensor data from a specific location.
        @warning This method acquires a lock but does not release it. The lock is expected
        to be released by a subsequent call to `set_data`. This is a dangerous pattern
        and can easily lead to deadlocks if `set_data` is not called.
        @return The sensor data, or None if the location does not exist.
        """
        if location in self.sensor_data:
            self.data_locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Updates sensor data and releases the lock for that location.
        @warning This method releases a lock that it did not acquire. It is designed to be
        paired with a previous call to `get_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_locks[location].release()

    def shutdown(self):
        """@brief Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main control thread for a Device.
    @details It manages a pool of worker threads and orchestrates the device's lifecycle,
    including fetching neighbors, dispatching jobs, and handling synchronization.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """@brief The main execution loop for the device's control thread."""
        workers = WorkerManager(device=self.device, num_workers=8)

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            
            # A None value for neighbors is the signal to shut down.
            if neighbours is None:
                break

            # Block Logic: Main event loop for a single timepoint.
            while True:
                # Pre-condition: Waits for new scripts to arrive or for the timepoint to be marked as done.
                if self.device.scripts_arrived or self.device.timepoint_done.wait():
                    # If scripts have arrived, dispatch them to the worker pool.
                    if self.device.scripts_arrived:
                        self.device.scripts_arrived = False
                        for (script, location) in self.device.scripts:
                            workers.add_job(script, location, neighbours)
                    # If the timepoint is done, exit the inner loop to proceed to barrier synchronization.
                    else:
                        self.device.timepoint_done.clear()
                        self.device.scripts_arrived = True
                        break
            
            # Wait for all worker threads to finish processing the jobs for this timepoint.
            workers.wait_all()

            # Synchronize with all other devices.
            self.device.barrier.wait()

        # Cleanly shut down the worker threads.
        workers.end()


class Worker(Thread):
    """@brief A worker thread that executes script jobs from a queue."""
    def __init__(self, device, jobs):
        super(Worker, self).__init__()
        self.device = device
        self.jobs = jobs

    def run(self):
        """@brief The main loop for the worker thread, processing jobs from the queue."""
        while True:
            script, location, neighbours = self.jobs.get()

            # A None job is a "poison pill" used to signal shutdown.
            if script is None and neighbours is None:
                self.jobs.task_done()
                break
            
            # Block Logic: Gather data from neighbors and the local device.
            # This relies on the problematic get_data/set_data locking pattern.
            script_data = []
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # If data was found, execute the script and broadcast the result.
            if script_data != []:
                result = script.run(script_data)

                # Update data on neighbor devices.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)

                # Update data on the local device. This call also releases the lock.
                self.device.set_data(location, result)

            self.jobs.task_done()


class WorkerManager:
    """
    @brief Manages a pool of worker threads and a job queue.
    @details Implements a standard thread pool pattern to manage concurrent execution of tasks.
    """
    def __init__(self, device, num_workers):
        self.device = device
        self.jobs = q.Queue()
        self.workers = []

        for _ in range(num_workers):
            thread = Worker(device, self.jobs)
            self.workers.append(thread)

        for worker in self.workers:
            worker.start()

    def add_job(self, script, location, neighbours):
        """@brief Adds a new job to the queue for the workers to process."""
        self.jobs.put((script, location, neighbours))

    def wait_all(self):
        """@brief Blocks until all jobs in the queue are completed."""
        self.jobs.join()

    def end(self):
        """@brief Shuts down the worker pool by sending a 'poison pill' to each worker and joining them."""
        for _ in self.workers:
            self.add_job(None, None, None)

        for worker in self.workers:
            worker.join()
