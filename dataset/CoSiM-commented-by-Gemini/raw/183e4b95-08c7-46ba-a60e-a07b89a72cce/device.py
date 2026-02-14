"""
This module defines a simulation framework for distributed devices using a
thread-pool-based architecture.

It includes several key components concatenated into one file:
- ThreadPool & Worker: A classic thread pool implementation where tasks are
  functions to be executed, managed via a queue.
- ReusableBarrier: A semaphore-based, two-phase reusable barrier for
  synchronizing multiple threads.
- Device: Represents a node in the network, containing its own thread pool and
  a main control thread.
- DeviceThread: The main control thread for a device, orchestrating the
  simulation time steps.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue

# Note: The original file had a dependency on a 'reusable_barrier_semaphore'
# module, but a definition for ReusableBarrier is also included below.
# The following classes appear to be library/helper code.

class Worker(Thread):
    """A worker thread that consumes tasks from a shared queue."""
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        """The main loop of the worker, executing tasks from the queue."""
        while True:
            func, args, kargs = self.tasks.get()
            # A None function is the sentinel value to signal termination.
            if func is None:
                self.tasks.task_done()
                break
            try:
                func(*args, **kargs)
            except Exception as e:
                print(e)
            self.tasks.task_done()


class ThreadPool:
    """A classic thread pool implementation."""
    def __init__(self, num_threads):
        self.tasks = Queue(99999)
        self.workers = []
        for _ in range(num_threads):
            self.workers.append(Worker(self.tasks))

    def add_task(self, func, *args, **kargs):
        """Adds a task (a function and its arguments) to the queue."""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Blocks until all tasks in the queue have been processed."""
        self.tasks.join()

    def terminateWorkers(self):
        """Sends a termination signal to all worker threads."""
        for _ in self.workers:
            self.tasks.put((None, None, None))

    def threadJoin(self):
        """Waits for all worker threads to complete."""
        for worker in self.workers:
            worker.join()


class ReusableBarrier():
    """
    A reusable barrier implemented with semaphores and a lock.

    This uses a two-phase signaling mechanism to ensure that threads from one
    barrier wait cycle do not proceed before all threads have been released,
    making it safe for repeated use in a loop.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads_phase1 = self.num_threads
        self.count_threads_phase2 = self.num_threads
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes the calling thread to wait at the barrier."""
        # Phase 1: Threads arrive and wait.
        self._phase(self.count_threads_phase1, self.threads_sem1, is_phase1=True)
        # Phase 2: Ensures all threads have left phase 1 before resetting.
        self._phase(self.count_threads_phase2, self.threads_sem2, is_phase1=False)

    def _phase(self, count, threads_sem, is_phase1):
        with self.count_lock:
            if is_phase1:
                self.count_threads_phase1 -= 1
                if self.count_threads_phase1 == 0:
                    for _ in range(self.num_threads):
                        threads_sem.release()
                    self.count_threads_phase2 = self.num_threads
            else:
                self.count_threads_phase2 -= 1
                if self.count_threads_phase2 == 0:
                    for _ in range(self.num_threads):
                        threads_sem.release()
                    self.count_threads_phase1 = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Represents a device in the simulation, managing its own thread pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's local data.
            supervisor (Supervisor): The central simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # --- Synchronization and State ---
        self.script_received = Event() # Signals end of script assignment for a time step.
        self.wait_neighbours = Event() # Signals that the neighbor list is ready for workers.
        self.scripts = []
        self.neighbours = []
        self.allDevices = []
        self.locks = [] # A list of locks for location-based data access.
        self.pool = ThreadPool(8)

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global synchronization objects for the simulation.

        Note: This implementation gives each device its own barrier, but they
        are functionally independent. True global sync is managed in DeviceThread.
        It also creates a fixed-size list of 50 locks.
        """
        self.allDevices = devices
        self.barrier = ReusableBarrier(len(devices))
        for _ in range(50):
            self.locks.append(Lock())

    def assign_script(self, script, location):
        """
        Assigns a script to the device by adding it directly to the thread pool.

        Args:
            script (Script): The script to execute.
            location (any): The data location for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Immediately adds the script execution task to the worker queue.
            self.pool.add_task(self.executeScript, script, location)
        else:
            # A None script signals the main DeviceThread that assignment is done.
            self.script_received.set()

    def get_data(self, location):
        """Gets data from a specific location (not intrinsically thread-safe)."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data at a specific location (not intrinsically thread-safe)."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main control thread of the device."""
        self.thread.join()

    def executeScript(self, script, location):
        """
        The task executed by a worker thread to run a single script.

        Args:
            script (Script): The script object to run.
            location (any): The data location to operate on.
        """
        # Wait until the main DeviceThread has fetched the neighbor list for this time step.
        self.wait_neighbours.wait()
        script_data = []

        # Gather data from neighbors, using location-based locks.
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

        # Execute script and write results back.
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
    The main control thread for a Device, orchestrating simulation time steps.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main lifecycle loop of the device."""
        while True:
            self.device.script_received.clear()
            self.device.wait_neighbours.clear()
            
            # Fetch the neighbor list for the new time step.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            
            # Unblock any worker threads waiting for the neighbor list.
            self.device.wait_neighbours.set()

            if self.device.neighbours is None:
                # End of simulation: gracefully shut down the thread pool.
                self.device.pool.wait_completion()
                self.device.pool.terminateWorkers()
                self.device.pool.threadJoin()
                return

            # This loop re-adds all previously seen scripts to the pool every
            # time step, which is likely a bug.
            for (script, location) in self.device.scripts:
                self.device.pool.add_task(self.device.executeScript, script, location)

            # Wait for the supervisor to signal that script assignment is complete.
            self.device.script_received.wait()
            # Wait for all tasks in the pool for this time step to finish.
            self.device.pool.wait_completion()

            # Synchronize with all other devices before starting the next time step.
            for dev in self.device.allDevices:
                dev.barrier.wait()