"""
This module implements a distributed device simulation using a worker pool.
NOTE: This file appears to be a composite of several different sources, with
duplicate class definitions and incorrect synchronization logic that will lead
to deadlocks. The comments below describe the intended, but flawed, logic.
"""

from threading import Event, Thread, Lock
from queue import Queue

# --- Worker and ThreadPool Implementation ---
# These classes define a generic thread pool for executing tasks.

class Worker(Thread):
    """A worker thread that consumes tasks from a queue."""
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        """Continuously fetches and executes tasks from the queue."""
        while True:
            func, args, kargs = self.tasks.get()
            try:
                func(*args, **kargs)
            except Exception as e:
                print(e)
            self.tasks.task_done()

class ThreadPool:
    """A pool of worker threads that execute tasks from a queue."""
    def __init__(self, num_threads):
        self.tasks = Queue()
        self.workers = [Worker(self.tasks) for _ in range(num_threads)]

    def add_task(self, func, *args, **kargs):
        """Adds a task to the queue."""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Blocks until all tasks in the queue are processed."""
        self.tasks.join()

# --- Main Simulation Logic ---

class Device(object):
    """
    Represents a device in the simulation. This implementation uses a
    thread pool to execute scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.wait_neighbours = Event()
        self.scripts = []
        self.neighbours = []
        self.allDevices = []
        self.locks = []
        self.pool = ThreadPool(8)
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None # This will be set during setup.

    def __str__(self):
        return f"Device {self.device_id}"

    def setup_devices(self, devices):
        """
        Configures the device with shared objects.
        FLAW: Each device creates its own barrier. They need to share a single
        barrier instance for inter-device synchronization to work.
        """
        self.allDevices = devices
        # This creates a new barrier for each device, which is incorrect.
        self.barrier = ReusableBarrier(len(devices))
        self.locks = [Lock() for _ in range(50)] # Pre-allocates a fixed number of locks.

    def assign_script(self, script, location):
        """
        Assigns a script and immediately adds its execution as a task to the thread pool.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.pool.add_task(self.executeScript, script, location)
        else:
            self.script_received.set()

    def get_data(self, location):
        """Gets data for a location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data for a location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's coordinator thread."""
        self.thread.join()

    def executeScript(self, script, location):
        """
        The method executed by worker threads to run a script.
        """
        self.wait_neighbours.wait() # Wait until neighbors are known for this time step.
        script_data = []

        # Gather data from neighbors with fine-grained locking.
        if self.neighbours is not None:
            for device in self.neighbours:
                with device.locks[location]:
                    data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
        
        # Gather data from self.
        with self.locks[location]:
            data = self.get_data(location)
        if data is not None:
            script_data.append(data)

        # Run script and propagate results with fine-grained locking.
        if script_data:
            result = script.run(script_data)
            if self.neighbours is not None:
                for device in self.neighbours:
                    with device.locks[location]:
                        device.set_data(location, result)
            with self.locks[location]:
                self.set_data(location, result)


class DeviceThread(Thread):
    """
    The coordinator thread for a device, managing time-steps.
    """

    def __init__(self, device):
        Thread.__init__(self, name=f"Device Thread {device.device_id}")
        self.device = device

    def run(self):
        """
        Main simulation loop.
        FLAW: The barrier logic at the end will cause a deadlock because each
        device has its own unique barrier instance from the flawed setup.
        """
        while True:
            # Reset events for the new time-step.
            self.device.script_received.clear()
            self.device.wait_neighbours.clear()

            # Get neighbors and signal workers that they can proceed.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.wait_neighbours.set()

            if self.device.neighbours is None:
                # End of simulation.
                self.device.pool.wait_completion()
                # The custom ThreadPool does not have terminateWorkers or threadJoin methods.
                # This part of the code would fail.
                return

            # Wait for the supervisor to assign all scripts for this time-step.
            self.device.script_received.wait()
            # Wait for the worker pool to finish all tasks for this time-step.
            self.device.pool.wait_completion()

            # This loop will cause a deadlock. Each device waits on its own
            # barrier, but there are no other threads waiting on that same instance.
            for dev in self.device.allDevices:
                if dev.barrier:
                    dev.barrier.wait()


# Note: The following ReusableBarrier is defined but not used by the main logic,
# which references an imported version. This is likely a copy-paste remnant.
class ReusableBarrier():
    """A correct two-phase reusable barrier implementation."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
