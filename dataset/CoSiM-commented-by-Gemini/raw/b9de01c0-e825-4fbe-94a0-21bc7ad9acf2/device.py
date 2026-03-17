



"""
This module implements a simulation of a distributed network of devices.

It includes classes for managing a thread pool (`ThreadPool`), a reusable barrier for synchronization
(`ReusableBarrierSem`), and the core `Device` and `DeviceThread` classes that model the behavior
and lifecycle of each node in the network. The simulation appears to be time-point based,
where devices execute scripts on local and neighboring data, and synchronize at the end of each time step.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue


class ThreadPool(object):
    """Manages a pool of worker threads to execute tasks concurrently.

    This thread pool is designed to pull tasks from a shared queue, where each task involves
    gathering data, running a script, and disseminating the results.
    """
    
    def __init__(self, num_threads, device):
        """Initializes the thread pool.

        Args:
            num_threads (int): The number of worker threads to create.
            device (Device): A reference to the parent device that owns this pool.
        """
        self.__device = device
        self.__queue = Queue(num_threads)
        self.__threads = [Thread(target=self.work) for _ in range(num_threads)]

        for thread in self.__threads:
            thread.start()

    def work(self):
        """The main work loop for each thread in the pool.

        A worker thread continuously fetches tasks from the queue. Each task consists of
        a script to execute, a data location, and a set of neighboring devices.
        The thread gathers data from the neighbors and its own device, runs the script,
        and then writes the result back to all involved devices.
        """
        while True:
            script, location, neighbours = self.__queue.get()

            # Shutdown signal for the thread is a tuple of Nones.
            if not script and not neighbours:
                self.__queue.task_done()
                break

            script_data = []

            # Block Logic: Gather data from all neighboring devices for the specified location.
            for device in neighbours:
                if self.__device.device_id != device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

            # Block Logic: Gather data from the local device itself.
            data = self.__device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Functional Utility: Executes the assigned script with the aggregated data.
                result = script.run(script_data)

                # Block Logic: Propagate the result of the script execution back to the neighbors.
                for device in neighbours:
                    if self.__device.device_id != device.device_id:
                        device.set_data(location, result)

                # Block Logic: Update the local device's data with the result.
                self.__device.set_data(location, result)
            self.__queue.task_done()

    def add_tasks(self, scripts, neighbours):
        """Adds a list of script-based tasks to the processing queue.

        Args:
            scripts (list): A list of tuples, where each tuple is (script, location).
            neighbours (list): A list of neighboring Device objects.
        """
        for script, location in scripts:
            self.__queue.put((script, location, neighbours))

    def wait_threads(self):
        """Blocks until all tasks in the queue have been processed."""
        self.__queue.join()

    def stop_threads(self):
        """Stops all worker threads in the pool gracefully.

        It first waits for the queue to be empty and then posts a shutdown
        signal (a None tuple) for each thread.
        """
        self.__queue.join()

        for thread in self.__threads:
            self.__queue.put((None, None, None))

        for thread in self.__threads:
            thread.join()


class ReusableBarrierSem(object):
    """A reusable barrier implementation using semaphores.

    This synchronization primitive ensures that all participating threads wait for each other
    at a certain point (the barrier) before any of them are allowed to continue.
    It uses a two-phase approach to allow the barrier to be reused multiple times.
    """
    
    def __init__(self, num_threads):
        """Initializes the reusable barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that will be synchronizing on this barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               # Mutex for thread-safe counter modification.
        self.threads_sem1 = Semaphore(0)         # Semaphore for the first phase of the barrier.
        self.threads_sem2 = Semaphore(0)         # Semaphore for the second phase.

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have reached it."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread to arrive releases all other waiting threads.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next use.

        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase of the barrier, ensuring threads don't loop around too quickly."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # The last thread to arrive releases all others for the second phase.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for next use.

        self.threads_sem2.acquire()


class Device(object):
    """Represents a single device (or node) in the distributed simulation."""
    
    num_threads = 8

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a device.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary representing the device's local data, keyed by location.
            supervisor (object): A central supervisor object that manages the network topology.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.script_received.clear()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.locks = dict()
        for loc in sensor_data:
            self.locks[loc] = Lock()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the shared barrier for all devices in the simulation.

        This method should be called on one designated device (e.g., device_id 0)
        to create and distribute the barrier.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a new script to be executed by this device.

        Args:
            script (object): The script object with a `run` method.
            location (any): The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script is a signal that the current timepoint is done.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific location on this device in a thread-safe manner.

        Args:
            location (any): The key for the desired data in `sensor_data`.

        Returns:
            The data at the given location, or None if the location does not exist.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]

        return None

    def set_data(self, location, data):
        """Updates data at a specific location on this device in a thread-safe manner.

        Args:
            location (any): The key for the data in `sensor_data`.
            data (any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main thread of execution for a single Device."""

    def __init__(self, device):
        """Initializes the device's main thread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.pool = ThreadPool(Device.num_threads, device)

    def run(self):
        """The main simulation loop for the device.

        This loop represents the device's lifecycle. It repeatedly gets its network
        neighbors, processes assigned scripts for a timepoint, and synchronizes
        with other devices at a barrier. The loop terminates when the supervisor
        signals a shutdown.
        """
        while True:
            # Get the current network topology for this device.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals shutdown.
                break
            
            # This inner loop represents a single timepoint in the simulation.
            while True:
                # Pre-condition: Wait for a signal that a script has been assigned.
                if self.device.script_received.is_set():
                    self.pool.add_tasks(self.device.scripts, neighbours)
                    self.device.script_received.clear()
                
                # Pre-condition: Wait for a signal that the timepoint has concluded.
                if self.device.timepoint_done.is_set():
                    self.device.timepoint_done.clear()
                    self.device.script_received.set()
                    break

            # Invariant: All tasks for the current timepoint must be completed before the barrier.
            self.pool.wait_threads()
            # Invariant: All devices must synchronize here before starting the next timepoint.
            self.device.barrier.wait()

        self.pool.stop_threads()
