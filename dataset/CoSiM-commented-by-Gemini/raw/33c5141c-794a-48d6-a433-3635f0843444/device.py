"""
@file device.py
@brief A device simulation using a custom producer-consumer thread pool.

This module contains a full implementation of a device simulation, including a
`ThreadPool` class to manage worker threads that process script execution jobs
from a queue.
"""


from threading import Event, Thread, Lock
from Queue import Queue

# The original file imported Barrier and ThreadPool from separate modules.
# For this context, their implementations are included here.


def execute_script(location, script, local_devices):
    """
    The core task function executed by worker threads.

    It gathers data from a list of devices for a specific location, runs a
    script on the data, and distributes the result back to the devices. This
    function is responsible for acquiring and releasing locks on each device
    it interacts with via the get_data/set_data calls.
    """

    data_collection = []

    # Aggregate data from all relevant devices.
    for device in local_devices:
        data = device.get_data(location)
        if data is not None:
            data_collection.append(data)

    if data_collection:
        result = script.run(data_collection)

        # Distribute the result back to all devices.
        for device in local_devices:
            device.set_data(location, result)

class ThreadPool(object):
    """
    A classic producer-consumer thread pool implementation.

    Workers are started on initialization and pull tasks from a shared queue.
    """

    def __init__(self, threads_count):
        """Initializes the queue and starts the worker threads."""
        self.tasks = Queue(threads_count)
        self.threads = []

        for _ in xrange(threads_count):
            self.threads.append(Thread(target=self.run))

        for thread in self.threads:
            thread.start()


    def run(self):
        """The main loop for a worker thread."""
        while True:
            # Block and retrieve a task from the queue.
            location, script, local_devices = self.tasks.get()

            # Check for the "poison pill" to terminate the thread.
            if script is None and local_devices is None:
                self.tasks.task_done()
                return

            execute_script(location, script, local_devices)
            # Signal that the task from the queue is complete.
            self.tasks.task_done()

    def add_task(self, location, script, local_devices):
        """Adds a new task to the queue for a worker to process."""
        self.tasks.put((location, script, local_devices))

    def wait_tasks(self):
        """Blocks until all tasks in the queue have been processed."""
        self.tasks.join()

    def join_threads(self):
        """Shuts down the thread pool gracefully."""
        self.wait_tasks()

        # Send a "poison pill" for each thread to ensure termination.
        for _ in xrange(len(self.threads)):
            self.add_task(None, None, None)

        for thread in self.threads:
            thread.join()


class Device(object):
    """
    Represents a device in the simulation, managing its own state, locks, and
    a master `DeviceThread`.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        self.barrier = None
        # Decentralized locking: each device owns the locks for its own data.
        self.locks = {}
        self.script_received = False

        for location in sensor_data:
            self.locks[location] = Lock()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes the shared barrier for inter-device synchronization."""
        if self.device_id == 0:
            # Device 0 acts as coordinator to create and distribute the barrier.
            # Imported 'Barrier' assumed to be a ReusableBarrier implementation.
            self.barrier = Barrier(len(devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to be processed in the next timepoint."""
        if script is not None:
            self.script_received = True
            self.scripts.append((location, script))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data for a location and acquires the lock for it.
        
        @warning Implements a dangerous locking pattern. The caller MUST ensure
        `set_data` is eventually called for the same location to release the lock,
        otherwise a deadlock will occur.
        """
        if location not in self.sensor_data:
            return None

        self.locks[location].acquire()
        return self.sensor_data[location]

    def set_data(self, location, data):
        """
        Updates data for a location and releases its lock.
        @see get_data
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """Shuts down the device's master thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The master thread for a device, acting as a producer that dispatches
    tasks to a `ThreadPool`.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.pool = ThreadPool(8)

    def run(self):
        """The main simulation loop for the master thread."""
        while True:
            local_devices = self.device.supervisor.get_neighbours()
            if local_devices is None:
                break # End of simulation.

            # Create a list of all devices involved in this timepoint's calculations.
            temp_set = set(local_devices)
            temp_set.add(self.device)
            local_devices = list(temp_set)
            
            # This convoluted loop waits for scripts and dispatches them. A simpler
            # design would be to wait for the event and then dispatch all scripts at once.
            while True:
                # Wait for scripts to be assigned OR for the timepoint to be marked done.
                if self.device.script_received or self.device.timepoint_done.wait():
                    if self.device.script_received:
                        self.device.script_received = False
                        # Add tasks to the pool as they are received.
                        for (location, script) in self.device.scripts:
                            self.pool.add_task(location, script, local_devices)
                    else:
                        # The timepoint is done, exit the script-adding loop.
                        self.device.timepoint_done.clear()
                        self.device.script_received = True
                        break
            
            # Wait for all dispatched tasks for this timepoint to be completed by the pool.
            self.pool.wait_tasks()
            # Synchronize with all other devices.
            self.device.barrier.wait()

        # Gracefully shut down the thread pool.
        self.pool.join_threads()
