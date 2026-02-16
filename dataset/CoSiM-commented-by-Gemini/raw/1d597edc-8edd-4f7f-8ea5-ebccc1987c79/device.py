"""
This module implements a sophisticated producer-consumer simulation of a distributed
device network, using a Thread Pool architecture for efficient task execution.

Each `Device` manages its own `ThreadPool` of long-lived `Worker` threads. The main
`DeviceThread` acts as a dispatcher, reading scripts from a queue and assigning
them as tasks to its thread pool. This avoids the overhead of creating new threads
for every task. Synchronization between devices is handled by a shared barrier, and
data consistency is maintained through a shared dictionary of location-based locks.
"""
from threading import Thread, Lock
from Queue import Queue
from barrier import Barrier
from thread_pool import ThreadPool

class Device(object):
    """
    Represents a device that manages a thread pool to execute computational tasks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device, its thread pool, and its main dispatcher thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # A thread-safe queue for receiving new script assignments.
        self.scripts_queue = Queue()
        # A shared dictionary mapping a location to its specific Lock.
        self.lock = {}
        # A global lock for safely initializing the shared lock dictionary.
        self.global_lock = None
        # A shared barrier for synchronizing all devices at the end of a time step.
        self.barrier = None

        # A lock used as a gate to prevent the main thread from running before
        # setup_devices is complete. It starts in a locked state.
        self.received_commun_data = Lock()
        self.received_commun_data.acquire()

        self.iteration_is_running = False

        # Each device has its own pool of worker threads.
        self.thread_pool = ThreadPool(8, device_id)
        self.thread_pool.start()

        # The main thread for this device, which acts as a task dispatcher.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs centralized setup for all devices, run by the master device (id 0).
        It creates and distributes the shared barrier and lock dictionary.
        """
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            self.lock = {}
            self.global_lock = Lock()

            # Distribute the shared objects to all other devices.
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier
                    device.lock = self.lock
                    device.global_lock = self.global_lock
                    # Release the gate lock for each slave device so it can start running.
                    device.received_commun_data.release()

            # Release the master device's own gate lock.
            self.received_commun_data.release()

    def assign_script(self, script, location):
        """
        Assigns a script by adding it to the queue. Safely initializes locks for
        new locations.
        """
        if script is not None:
            # Use a global lock to safely add a new lock to the shared dictionary
            # if a new location is seen for the first time.
            with self.global_lock:
                if location not in self.lock:
                    self.lock[location] = Lock()
        self.scripts_queue.put((script, location))

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        data = None
        if location in self.sensor_data:
            data = self.sensor_data[location]
        return data

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()

    def acquire_location(self, location):
        """Acquires the lock for a specific location."""
        self.lock[location].acquire()

    def release_location(self, location):
        """Releases the lock for a specific location."""
        self.lock[location].release()


class DeviceThread(Thread):
    """
    The main dispatcher thread for a device.

    It pulls scripts from a queue and dispatches them as tasks to the device's
    thread pool.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main dispatcher loop.
        """
        # Wait until setup_devices is complete before starting.
        self.device.received_commun_data.acquire()
        supervisor = self.device.supervisor
        barrier = self.device.barrier
        thread_pool = self.device.thread_pool

        while True:
            neighbours = supervisor.get_neighbours()

            if neighbours is None:
                # Signal the end of the simulation.
                thread_pool.shutdown()
                barrier.wait()
                break

            # Ensure the device itself is in its list of neighbors for data processing.
            if neighbours.count(self.device) == 0:
                neighbours.append(self.device)

            # Block Logic: Dispatch all scripts for the current time step.
            if not self.device.iteration_is_running:
                self.device.iteration_is_running = True
                # Dispatch any scripts that were already in the list.
                for (script, location) in self.device.scripts:
                    task = (script, location, neighbours)
                    thread_pool.add_task(task)

            # Block Logic: Continuously pull new scripts from the queue and dispatch them.
            while True:
                (script, location) = self.device.scripts_queue.get()

                # A (None, None) tuple is the signal that all scripts for this
                # time step have been assigned.
                if script is None and location is None:
                    self.device.iteration_is_running = False
                    break
                else:
                    task = (script, location, neighbours)
                    thread_pool.add_task(task)
                    self.device.scripts.append((script, location))

            # Wait for all dispatched tasks for this time step to be completed by the workers.
            thread_pool.wait_finish()
            # Synchronize with all other devices.
            barrier.wait()


class Worker(Thread):
    """
    A long-lived worker thread that is part of a ThreadPool.
    It continuously fetches and executes tasks from a shared queue.
    """
    def __init__(self, worker_id, thread_pool):
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.worker_id = worker_id
        self.thread_pool = thread_pool

    def run(self):
        """
        The main loop for the worker. Fetches a task and executes it.
        """
        while True:
            task = self.thread_pool.get_task()
            (script, location, devices) = task

            # A 'None' task is a "poison pill" used to signal shutdown.
            if script is None and location is None and devices is None:
                self.thread_pool.task_done()
                break

            script_data = []

            # Use one of the devices (assumed to be device 0) to acquire the
            # location-specific lock. This is a safe locking protocol.
            chosen_device = devices[0]
            chosen_device.acquire_location(location)

            # Block Logic: While holding the location lock, gather data.
            working_devices = []
            for device in devices:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
                    working_devices.append(device)

            if script_data != []:
                result = script.run(script_data)
                # Disseminate the result to all relevant devices.
                for device in working_devices:
                    device.set_data(location, result)

            # Release the location lock so other workers can process this location.
            chosen_device.release_location(location)
            self.thread_pool.task_done()

class ThreadPool(object):
    """
    A simple thread pool implementation to manage a fixed number of worker threads.
    """
    def __init__(self, num_threads, pool_id):
        self.num_threads = num_threads
        self.queue_taks = Queue()
        self.pool_id = pool_id

        self.worker = []
        for i in xrange(self.num_threads):
            self.worker.append(Worker(i, self))

    def start(self):
        """Starts all worker threads in the pool."""
        for i in xrange(self.num_threads):
            self.worker[i].start()

    def add_task(self, task):
        """Adds a task to the queue for a worker to execute."""
        self.queue_taks.put(task)

    def get_task(self):
        """Retrieves a task from the queue. Blocks if the queue is empty."""
        return self.queue_taks.get()

    def task_done(self):
        """Signals that a formerly enqueued task is complete."""
        self.queue_taks.task_done()

    def wait_finish(self):
        """Blocks until all items in the queue have been gotten and processed."""
        self.queue_taks.join()

    def shutdown(self):
        """Shuts down the thread pool gracefully."""
        self. wait_finish()
        # Add a "poison pill" for each worker to make them exit their loops.
        for i in xrange(self.num_threads):
            self.add_task((None, None, None))
        # Wait for all worker threads to terminate.
        for i in xrange(self.num_threads):
            self.worker[i].join()
