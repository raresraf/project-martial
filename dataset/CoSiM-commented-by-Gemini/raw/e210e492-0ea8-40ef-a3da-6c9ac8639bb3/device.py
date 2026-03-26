"""
device.py

@brief Simulates a device in a distributed system using a task-based thread pool architecture.
@description This module defines a simulation framework for a network of devices that
process scripts in synchronized timepoints. Each device operates a master thread
(`DeviceThread`) which manages a dedicated thread pool (`ThreadPool`) for executing tasks.
Global synchronization is achieved through a shared `CyclicBarrier`. The design uses
a queue for script assignments and fine-grained locking for data locations.
"""

from threading import Thread, Lock, Condition, Semaphore, Event
from Queue import Queue

# --- Custom Concurrency Utilities ---
# These classes are defined within the same file to create a self-contained module.

class CyclicBarrier:
    """
    A custom, reusable barrier implementation that allows a set of threads to all wait
    for each other to reach a common barrier point.
    """
    def __init__(self, parties):
        """Initializes the barrier for a given number of parties (threads)."""
        self.parties = parties
        self.count = 0
        self.condition = Condition()

    def wait(self):
        """
        Causes the current thread to wait until all parties have called wait() on this
        barrier. When the last party arrives, all waiting threads are released.
        """
        with self.condition:
            self.count += 1
            if self.count == self.parties:
                # Last thread arrived; wake up all waiting threads.
                self.condition.notifyAll()
                self.count = 0  # Reset for the next use.
            else:
                # Not all threads have arrived yet; wait.
                self.condition.wait()


class ThreadPool:
    """
    A simple thread pool implementation for executing tasks concurrently.
    """
    def __init__(self, num_threads):
        """Initializes the pool and starts the worker threads."""
        self.task_queue = Queue()
        # Semaphore to track the number of pending tasks.
        self.num_tasks = Semaphore(0)
        # Events to signal shutdown states.
        self.stop_signal = Event()      # Signal for workers to terminate permanently.
        self.shutdown_signal = Event()  # Signal to stop accepting new tasks.

        self.threads = [Worker(self.task_queue, self.num_tasks, self.stop_signal) for _ in range(num_threads)]
        for t in self.threads:
            t.start()

    def submit(self, task):
        """Adds a new task to the queue for execution."""
        if not self.shutdown_signal.is_set():
            self.task_queue.put(task)
            self.num_tasks.release()

    def shutdown(self):
        """Stops the pool from accepting new tasks."""
        self.shutdown_signal.set()

    def wait_termination(self, end=True):
        """
        Waits for all tasks to complete and optionally terminates the worker threads.

        Args:
            end (bool): If True, workers are terminated. If False, the pool is
                        reset to accept new tasks for the next timepoint.
        """
        self.task_queue.join()  # Wait for all tasks in the queue to be processed.
        if end:
            self.stop_signal.set()
            # Unblock any waiting workers so they can see the stop signal.
            for _ in self.threads:
                self.task_queue.put(None)
                self.num_tasks.release()
            for t in self.threads:
                t.join()
        else:
            self.shutdown_signal.clear()


class Worker(Thread):
    """
    A worker thread that consumes and executes tasks from a queue.
    """
    def __init__(self, task_queue, num_tasks, stop_signal):
        Thread.__init__(self)
        self.task_queue = task_queue
        self.num_tasks = num_tasks
        self.stop_signal = stop_signal

    def run(self):
        """Main loop: waits for a task, executes it, and repeats."""
        while True:
            self.num_tasks.acquire()  # Wait for a task to become available.
            if self.stop_signal.is_set():
                break  # Terminate if the stop signal is received.
            task = self.task_queue.get()
            if task:
                task.run()
            self.task_queue.task_done()

# --- Main Simulation Classes ---

class SharedDeviceData:
    """
    A container for data shared across all devices in the simulation.
    This includes global synchronization primitives and shared data structures.
    """
    def __init__(self, num_devices):
        self.num_devices = num_devices
        # A barrier to synchronize all devices at the end of each timepoint.
        self.timepoint_barrier = CyclicBarrier(num_devices)
        # A dictionary of fine-grained locks, one for each data location.
        self.location_locks = {}
        # A lock to protect the location_locks dictionary itself during modifications.
        self.ll_lock = Lock()


class RunScript:
    """A callable task object that encapsulates a single script execution."""
    def __init__(self, script, location, neighbours, device):
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """
        The core logic for executing one script. This is called by a Worker thread.
        """
        # Safely get the lock for the target location.
        with self.device.shared_data.ll_lock:
            lock = self.device.shared_data.location_locks[self.location]

        # Block Logic: Gather data, execute script, and write back results, all
        # while holding the lock for this specific location.
        with lock:
            # 1. Gather data from neighbors and the local device.
            script_data = []
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # 2. Execute the script if there is data to process.
            if script_data:
                result = self.script.run(script_data)
                # 3. Write back the result to all involved devices.
                # NOTE: The set_data calls are not individually locked, which could be a
                # race condition if another script on another device tries to write to
                # this device's sensor_data for a *different* location at the same time.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)


class Device(object):
    """
    Represents a single device (node) in the distributed network.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.num_cores = 8
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []  # List of scripts to re-run every timepoint.
        self.new_scripts = Queue() # Queue for newly assigned scripts.
        # Each device has one master thread that orchestrates its thread pool.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the shared data object for the entire simulation.
        This must be called on one "master" device (device_id 0).
        """
        if self.device_id == 0:
            shared_data = SharedDeviceData(len(devices))
            # Pre-populate location locks for initially known locations.
            for data_loc in self.sensor_data:
                if data_loc not in shared_data.location_locks:
                    shared_data.location_locks[data_loc] = Lock()
            # Distribute the shared data object to all devices.
            for dev in devices:
                dev.shared_data = shared_data

    def assign_script(self, script, location):
        """Adds a script to the queue for processing in the next timepoint."""
        self.new_scripts.put((script, location))

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the master device thread to shut down cleanly."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The master control thread for a single Device. It manages a thread pool
    to execute script tasks in synchronized timepoints (supersteps).
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main simulation loop, processing one timepoint per iteration."""
        thread_pool = ThreadPool(self.device.num_cores)
        while True:
            # --- Communication Phase ---
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break  # Supervisor signal to shut down.

            # --- Computation Phase ---
            # 1. Re-submit persistent scripts from previous timepoints.
            for (script, location) in self.device.scripts:
                thread_pool.submit(RunScript(script, location, neighbours, self.device))

            # 2. Process newly assigned scripts for this timepoint from the queue.
            while True:
                (script, location) = self.device.new_scripts.get()
                if script is None:
                    break # End of script assignments for this timepoint.

                # Dynamically add new location locks if needed.
                with self.device.shared_data.ll_lock:
                    if location not in self.device.shared_data.location_locks:
                        self.device.shared_data.location_locks[location] = Lock()

                thread_pool.submit(RunScript(script, location, neighbours, self.device))
                self.device.scripts.append((script, location)) # Add to persistent list.

            # --- Synchronization Phase ---
            # Wait for all submitted tasks for this timepoint to finish.
            thread_pool.shutdown()
            thread_pool.wait_termination(False) # 'False' resets the pool for the next superstep.

            # Wait at the global barrier for all other devices to finish their timepoint.
            self.device.shared_data.timepoint_barrier.wait()

        # Final shutdown of the thread pool.
        thread_pool.wait_termination()
