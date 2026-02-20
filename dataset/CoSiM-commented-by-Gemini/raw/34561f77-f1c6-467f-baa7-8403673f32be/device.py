"""
This module provides a structured simulation of a distributed device network.
It employs a clear separation of concerns, with distinct classes for devices,
synchronization primitives, worker threads, and job management. This version
uses a worker pool and a job queue for concurrent script execution.
"""

from threading import Event, Thread, Condition, Lock
import Queue

WORKERS_PER_DEVICE = 8
NUM_LOCATION_LOCKS = 25

class Barrier():
    """
    A simple, non-reusable barrier to synchronize a fixed number of threads.
    
    Note: This implementation is likely incorrect for reuse in a loop, as the
    thread count is not reliably reset for all threads in all conditions. It works
    for a single synchronization point.
    """
    def __init__(self, num_threads):
        
        self.condition = Condition()
        self.count_threads = 0
        self.num_threads = num_threads

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have called this method.
        """
        self.condition.acquire()
        self.count_threads = self.count_threads + 1

        
        if self.count_threads == self.num_threads:
            # Last thread arrives, notifies all waiting threads.
            self.condition.notify_all()
            self.count_threads = 0 # Reset for potential reuse.
        else:
            # Wait to be notified by the last thread.
            self.condition.wait()

        self.condition.release()

class Job():
    """A data class representing a single task for a worker thread."""
    def __init__(self, location, neighbours, script):
        """
        Initializes a Job.
        
        Args:
            location (any): The data location this job pertains to.
            neighbours (list): A list of neighboring Device objects.
            script (obj): The script object to be executed.
        """
        self.location = location
        self.neighbours = neighbours
        self.script = script

    def get_location(self):
        """Returns the job's location."""
        return self.location

    def get_neighbours(self):
        """Returns the list of neighbors for this job."""
        return self.neighbours

    def get_script(self):
        """Returns the script to be executed."""
        return self.script

class DeviceSync(object):
    """
    A container for all synchronization primitives associated with a device.
    This class centralizes the management of barriers, locks, and events.
    """
    def __init__(self):
        """Initializes all synchronization objects to None."""
        self.barrier = None
        self.location_locks = []
        self.receive_scripts =  Event()
        self.setup = Event()

    def init_barrier(self, num_threads):
        """Creates the barrier for a given number of threads."""
        self.barrier = Barrier(num_threads)

    def wait_threads(self):
        """Waits on the barrier."""
        self.barrier.wait()

    def init_location_locks(self, num_locations):
        """Pre-allocates a list of locks for data locations."""
        for location in range(0, num_locations):
            self.location_locks.append(Lock())

    def get_location_lock(self, location):
        """
        Retrieves the lock for a specific location.
        
        Args:
            location (int): The index of the location.
        
        Returns:
            Lock: The lock object for that location.
        """
        return self.location_locks[location]



    def set_receive_scripts(self):
        """Signals that all scripts for a timepoint have been received."""
        self.receive_scripts.set()

    def wait_receive_scripts(self):
        """Blocks until all scripts for a timepoint have been received."""
        self.receive_scripts.wait()

    def clear_scripts(self):
        """Resets the script reception event for the next timepoint."""
        self.receive_scripts.clear()

    def set_setup_event(self):
        """Signals that the initial device setup is complete."""
        self.setup.set()

    def wait_setup_event(self):
        """Blocks until the initial device setup is complete."""
        self.setup.wait()

class Worker(Thread):
    """
    A worker thread that processes jobs from a queue.
    """
    def __init__(self, device, scripts):
        """
        Initializes the worker.
        
        Args:
            device (Device): The parent device.
            scripts (Queue.Queue): The job queue to pull from.
        """
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts

    def get_neighbours_data(self, job):
        """
        Collects data for a job's location from the local device and all neighbors.
        """
        all_data = []

        
        for device in job.neighbours:
            data = device.get_data(job.get_location())
            all_data.append(data)

        
        data = self.device.get_data(job.get_location())
        all_data.append(data)

        # Filter out None values if a device doesn't have data for the location.
        all_data = list(filter(None, all_data))

        
        return all_data

    def update_neighbours_data(self, job, new_data):
        """Updates the computed data on the local device and all neighbors."""
        for device in job.neighbours:
            device.set_data(job.get_location(), new_data)
        self.device.set_data(job.get_location(), new_data)

    def run(self):
        """
        The main loop for the worker. It continuously fetches and processes jobs.
        """
        while True:
            
            job = self.scripts.get()

            # Pre-condition: A None script is the signal to terminate.
            if job.script is None:
                
                
                self.scripts.task_done()
                break

            # Block Logic: Acquire a lock for the specific location to ensure
            # serialized access to shared data.
            with self.device.syncronization.get_location_lock(job.get_location()):
                data = self.get_neighbours_data(job)
                if data != []:
                    new_data = job.script.run(data)
                    self.update_neighbours_data(job, new_data)

            
            self.scripts.task_done()

class WorkerPool(object):
    """Manages a pool of worker threads for a device."""


    def __init__(self, device, num_workers):
        """
        Initializes the worker pool and starts the workers.
        
        Args:
            device (Device): The parent device.
            num_workers (int): The number of workers in the pool.
        """
        self.device = device
        self.num_workers = num_workers

        self.scripts = Queue.Queue()
        self.workers_scripts = []

        self.start_workers()

    def start_workers(self):
        """Creates and starts all worker threads."""
        for worker_id in range(0, self.num_workers):
            self.workers_scripts.append(Worker(self.device, self.scripts))
            self.workers_scripts[worker_id].start()

    def join_workers(self):
        """Waits for all jobs in the queue to complete and workers to terminate."""


        for worker_id in range(0, self.num_workers - 1):
            self.scripts.join()
            self.workers_scripts[worker_id].join()

    def stop_workers(self):
        """Stops all worker threads by sending a termination job (None)."""
        
        for worker_id in range(0, WORKERS_PER_DEVICE):
            self.add_job(Job(None, None, None))

        self.join_workers()

    def delete_workers(self):
        """Removes worker threads from the list."""
        for worker_id in range(0, self.num_workers - 1):
            del self.workers_scripts[-1]

    def add_job(self, job):
        """Adds a new job to the worker queue."""
        self.scripts.put(job)

class Device(object):
    """
    The main device class, orchestrating the worker pool and synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.scripts = []
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.syncronization = DeviceSync()
        self.worker_pool = WorkerPool(self, WORKERS_PER_DEVICE)
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources. The device with the highest ID acts
        as the master for this one-time setup.
        """
        
        if self.device_id == len(devices) - 1:
            self.syncronization.init_barrier(len(devices))
            self.syncronization.init_location_locks(NUM_LOCATION_LOCKS)


            # Distribute the shared synchronization objects to all devices.
            for device in devices:
                device.syncronization.barrier = self.syncronization.barrier
                device.syncronization.location_locks = self.syncronization.location_locks
                device.syncronization.set_setup_event()

    def assign_script(self, script, location):
        """Assigns a script to be processed or signals the end of a timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.syncronization.set_receive_scripts()

    def add_job(self, job):
        """Adds a job to the device's worker pool."""
        self.worker_pool.add_job(job)

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a device's lifecycle."""

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop. It synchronizes devices at the start of each
        timepoint, processes jobs, and synchronizes again before the next timepoint.
        """
        # Block until the initial setup is complete.
        self.device.syncronization.wait_setup_event()

        
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            # Pre-condition: A None value for neighbours signals the end of the simulation.
            if neighbours is None:
                self.device.worker_pool.stop_workers()
                break

            # Block Logic: A two-phase synchronization pattern for each timepoint.
            # 1. Wait at the barrier for all devices to be ready for the new timepoint.
            self.device.syncronization.wait_threads()
            # 2. Wait for the supervisor to assign all scripts for this timepoint.
            self.device.syncronization.wait_receive_scripts()

            
            for (script, location) in self.device.scripts:
                self.device.add_job(Job(location, neighbours, script))

            
            # 3. Wait again for all devices to finish processing their jobs.
            self.device.syncronization.wait_threads()
            # 4. Clear the script reception event for the next cycle.
            self.device.syncronization.clear_scripts()
