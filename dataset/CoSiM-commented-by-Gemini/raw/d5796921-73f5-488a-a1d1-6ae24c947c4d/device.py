"""
This module provides a more complex simulation of a distributed device system,
utilizing a worker pool pattern for concurrent script execution.
"""

from threading import Event, Thread, Lock, Condition
import Queue  # Note: 'queue' in Python 3


class ReusableBarrier():
    """
    A reusable barrier implementation using a Condition variable.
    WARNING: This implementation is not fully correct and can be prone to race
    conditions (the 'lapper' problem) under certain thread scheduling scenarios.
    A two-phase barrier would be a more robust solution.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Blocks until all `num_threads` have called this method.
        """
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                self.cond.notify_all()
                self.count_threads = self.num_threads
            else:
                self.cond.wait()

    def acquire(self):
        """Acquires the underlying condition lock."""
        self.cond.acquire()


class Worker(Thread):
    """
    A worker thread that consumes and executes jobs from a queue.
    """

    def __init__(self, scripts_buffer, device):
        """
        Initializes the Worker thread.

        Args:
            scripts_buffer (Queue.Queue): The queue to fetch jobs from.
            device (Device): The parent device this worker belongs to.
        """
        Thread.__init__(self)
        self.device = device
        self.script_buffer = scripts_buffer

    def get_script_data(self, job):
        """
        Gathers data from the parent device and its neighbors for a given job.

        Args:
            job (Job): The job being executed.

        Returns:
            list: A list of sensor data.
        """
        script_data = [dev.get_data(job.location) for dev in job.neighbours if dev.get_data(job.location) is not None]
        data = self.device.get_data(job.location)
        if data is not None:
            script_data.append(data)
        return script_data

    def update_data_on_neighbours(self, job, result):
        """
        Propagates the result of a script execution to the device and its neighbors.

        Args:
            job (Job): The job that was executed.
            result: The result of the script execution.
        """
        for device in job.neighbours:
            device.set_data(job.location, result)
        self.device.set_data(job.location, result)

    def run(self):
        """
        The main loop for the worker thread. Continuously fetches jobs from the
        queue and executes them until a poison pill (None script) is received.
        """
        while True:
            job = self.script_buffer.get()

            # A None script is a "poison pill" to terminate the worker.
            if job.script is None:
                self.script_buffer.task_done()
                break

            with self.device.sync.get_location_lock(job.location):
                script_data = self.get_script_data(job)
                if script_data:
                    result = job.script.run(script_data)
                    self.update_data_on_neighbours(job, result)

            self.script_buffer.task_done()


class WorkerPool(object):
    """
    Manages a pool of worker threads.
    """

    def __init__(self, workers, device):
        """
        Initializes the worker pool.

        Args:
            workers (int): The number of worker threads in the pool.
            device (Device): The parent device.
        """
        self.workers = workers
        self.workers_scripts = []
        self.scripts_buffer = Queue.Queue()
        self.device = device
        self.start_workers()

    def start_workers(self):
        """Creates and starts the worker threads."""
        for i in range(self.workers):
            worker = Worker(self.scripts_buffer, self.device)
            self.workers_scripts.append(worker)
            worker.start()

    def add_job(self, job):
        """Adds a job to the worker queue."""
        self.scripts_buffer.put(job)

    def join_workers(self):
        """Waits for all jobs in the queue to be completed and joins the workers."""
        self.scripts_buffer.join()
        for worker in self.workers_scripts:
            worker.join()

    def make_workers_stop(self):
        """Stops all worker threads by sending poison pills."""
        for _ in range(self.workers):
            self.add_job(Job(None, None, None))
        self.join_workers()


class Job():
    """
    A data class representing a single unit of work for a worker thread.
    """

    def __init__(self, neighbours, script, location):
        self.neighbours = neighbours
        self.script = script
        self.location = location


class DeviceSync(object):
    """
    Encapsulates synchronization primitives for a device.
    """

    def __init__(self):
        self.setup = Event()
        self.scripts_received = Event()
        self.location_locks = []
        self.barrier = None

    def init_location_locks(self, locations):
        self.location_locks = [Lock() for _ in range(locations)]

    def init_barrier(self, threads):
        self.barrier = ReusableBarrier(threads)

    def set_setup_event(self):
        self.setup.set()

    def wait_setup_event(self):
        self.setup.wait()

    def set_scripts_received(self):
        self.scripts_received.set()

    def wait_scripts_received(self):
        self.scripts_received.wait()

    def clear_scripts_received(self):
        self.scripts_received.clear()

    def wait_threads(self):
        self.barrier.wait()

    def get_location_lock(self, location):
        return self.location_locks[location]


class Device(object):
    """
    Represents a device, which has its own coordinator thread and a pool of worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        self.sync = DeviceSync()
        self.worker_pool = WorkerPool(8, self)

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return f"Device {self.device_id}"

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects.
        This is a centralized setup method, intended to be called on one device.
        """
        if self.device_id == len(devices) - 1:
            self.sync.init_location_locks(25)
            self.sync.init_barrier(len(devices))

            for device in devices:
                device.sync.barrier = self.sync.barrier
                device.sync.location_locks = self.sync.location_locks
                device.sync.set_setup_event()

    def add_job(self, job):
        self.worker_pool.add_job(job)

    def assign_script(self, script, location):
        """
        Assigns a script to the device for later execution.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.sync.set_scripts_received()

    def get_data(self, location):
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's coordinator thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main coordinator thread for a device. It manages the device's
    participation in the synchronized time-steps of the simulation.
    """

    def __init__(self, device):
        Thread.__init__(self, name=f"Device Thread {device.device_id}")
        self.device = device

    def run(self):
        """
        Main simulation loop for the device.
        """
        self.device.sync.wait_setup_event()
        while True:
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                self.device.worker_pool.make_workers_stop()
                break

            # Synchronize with all other device coordinators before processing scripts.
            self.device.sync.wait_threads()
            self.device.sync.wait_scripts_received()

            # Add all assigned scripts for the current time-step to the worker pool queue.
            for (script, location) in self.device.scripts:
                self.device.add_job(Job(neighbours, script, location))

            # Synchronize again to ensure all devices have submitted their jobs.
            self.device.sync.wait_threads()
            self.device.sync.clear_scripts_received()
