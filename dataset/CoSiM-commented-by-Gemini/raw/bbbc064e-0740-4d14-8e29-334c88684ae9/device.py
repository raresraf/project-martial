"""
This module implements a simulation framework for a network of communicating devices.

The simulation is structured around a `Device` class, which manages its own worker
threads (`WorkerThread`) and a task queue. A key feature of this implementation is the
centralized lock management performed in the `setup_devices` method, where one device
creates a shared set of locks for all data locations across the entire network,
ensuring thread-safe data access during script execution. Synchronization between
time-steps is handled by a reusable barrier.
"""

from Queue import Queue
from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """A reusable barrier for synchronizing a fixed number of threads.

    This implementation uses a two-phase protocol with two semaphores to ensure
    that threads from one "wave" do not overlap with threads from the next.
    """

    def __init__(self, num_threads):
        """
        Args:
            num_threads (int): The number of threads that will synchronize on this barrier.
        """
        self.num_threads = num_threads
        # Counters are stored in a list to be mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier. Consists of two phases."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the counter for the current phase.
            threads_sem (Semaphore): The semaphore for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all other waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """Represents a single device node in the simulation.

    Each device manages its own task queue and a pool of worker threads. It holds
    local data and references to a supervisor, a shared barrier, and a shared
    set of locks for all data locations in the network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and its worker threads.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): The local data held by this device.
            supervisor (object): The central supervisor for network information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.queue = Queue()
        self.workers = [WorkerThread(self) for _ in range(8)]
        self.thread = DeviceThread(self)
        self.thread.start()
        for thread in self.workers:
            thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared resources across all devices.

        One device (id 0) acts as the master to create a single barrier and a
        global dictionary of locks for every unique data location in the network.
        These shared resources are then distributed to all other devices.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            locks = {}
            
            # Block Logic: Creates a unique lock for each data location across all devices.
            for device in devices:
                for location in device.sensor_data:
                    if not location in locks:
                        locks[location] = Lock()

            # Block Logic: Assigns the shared barrier and locks to every device.
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """Assigns a script to the device for the current time-step.

        Args:
            script (object): The script object to be run.
            location (any): The data location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of a time-step's script assignments.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific location on this device."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """Updates data at a specific location on this device."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class WorkerThread(Thread):
    """A worker thread that executes tasks from a device's queue."""

    def __init__(self, device):
        """
        Args:
            device (Device): The parent device whose queue this worker serves.
        """
        Thread.__init__(self)
        self.device = device

    def run(self):
        """The main execution loop for the worker.

        It continuously pulls tasks (script, location) from the queue. For each task,
        it safely acquires a lock for that location, gathers data from neighbors and
        itself, runs the script, and propagates the result.
        """
        while True:
            item = self.device.queue.get()
            if item is None:
                # None is the signal to terminate the worker thread.
                break

            (script, location) = item

            # Invariant: The `with` statement ensures that the lock for a specific
            # data location is held for the entire duration of data gathering,
            # execution, and updating, preventing race conditions between workers.
            with self.device.locks[location]:
                script_data = []

                # Block Logic: Gathers data from all neighboring devices.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Gathers data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    result = script.run(script_data)

                    # Block Logic: Propagates the result to all neighbors and the local device.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

            self.device.queue.task_done()


class DeviceThread(Thread):
    """The main control loop for a device, managing simulation time-steps."""

    def __init__(self, device):
        """
        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop, executed once per time-step."""
        while True:
            # Fetches the current set of neighbors from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                # A None value for neighbors signals the end of the simulation.
                break
            
            # Invariant: Waits for the supervisor to signal that all scripts for the
            # current time-step have been assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Puts all assigned scripts for the current time-step into the task queue.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location))

            # Invariant: Waits for all worker threads to complete their tasks for this step.
            self.device.queue.join()

            # Invariant: Synchronizes with all other devices before starting the next time-step.
            self.device.barrier.wait()

        # Block Logic: Gracefully shuts down all worker threads.
        for _ in range(8):
            self.device.queue.put(None)
        for thread in self.device.workers:
            thread.join()
