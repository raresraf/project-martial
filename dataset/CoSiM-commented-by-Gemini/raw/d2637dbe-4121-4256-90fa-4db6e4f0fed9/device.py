"""
This module simulates a distributed system of devices that process sensor data
in synchronized time steps. It uses a multi-layered threading model where each
device runs its own thread and manages a pool of worker threads to execute
computational scripts. Synchronization across all devices is handled by a
reusable barrier.

Note: This script appears to be written for Python 2, indicated by `from Queue import Queue`.
"""

from threading import Lock, Thread, Semaphore, Event
from Queue import Queue


class ReusableBarrier(object):
    """
    A reusable barrier that synchronizes a fixed number of threads.

    This barrier ensures that all participating threads wait for each other at a
    synchronization point before any of them are allowed to continue. It is reusable,
    meaning it can be used multiple times (e.g., in a loop). It uses a two-phase
    protocol to prevent threads from one iteration from proceeding before all threads
    from the previous iteration have exited the barrier.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        :param num_threads: The number of threads that must call wait() before they
                            are all released.
        """
        self.num_threads = num_threads
        # Using a list to hold the counter is a way to have a mutable integer
        # that can be modified by different threads.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to block until all `num_threads` have called this method.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier synchronization.

        :param count_threads: The counter for the current phase.
        :param threads_sem: The semaphore for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # The last thread to arrive releases all other threads.
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device (or node) in the distributed simulation.

    Each device has its own sensor data, executes scripts, and synchronizes
    with other devices. It runs a dedicated control thread (`DeviceThread`).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        :param device_id: A unique identifier for the device.
        :param sensor_data: A dictionary mapping locations to sensor values.
        :param supervisor: A central supervisor object that provides neighborhood info.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None  # The shared barrier for all devices.

        # A dictionary of locks, one for each sensor location, to allow
        # concurrent access to different data points on the same device.
        self.locks = {}
        for spot in sensor_data:
            self.locks[spot] = Lock()
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """

        Initializes the shared barrier and distributes it to all devices.

        This method should be called on one device only (e.g., device 0) to
        ensure a single shared barrier instance is used across the system.
        
        :param devices: A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed for the current timepoint.

        :param script: The script object to execute.
        :param location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is a sentinel indicating all scripts for the
            # timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Thread-safely retrieves sensor data for a given location.

        :param location: The sensor location to read from.
        :return: The sensor data, or None if the location is not found.
        """
        for loc in self.sensor_data:
            if loc == location:
                self.locks[loc].acquire()
                return self.sensor_data[loc]
        return None

    def set_data(self, location, data):
        """
        Thread-safely updates sensor data for a given location.

        :param location: The sensor location to write to.
        :param data: The new data to be written.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device instance.

    It orchestrates the device's lifecycle through discrete, synchronized timepoints,
    managing a thread pool to execute assigned scripts.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.
        
        :param device: The Device object this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Each device has its own thread pool for script execution.
        self.dev_threads = ThreadsForEachDevice(8)

    def run(self):
        """
        The main loop for the device, processing scripts at each timepoint.
        """
        self.dev_threads.device = self.device

        while True:
            # Get the list of neighbors for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break  # Supervisor signals termination.

            # Wait until all scripts for this timepoint are assigned.
            self.device.timepoint_done.wait()

            # Dispatch assigned scripts as jobs to the worker thread pool.
            for (script, location) in self.device.scripts:
                self.dev_threads.jobs_to_be_done.put(
                    (neighbours, script, location))

            self.device.timepoint_done.clear()

            # Wait for the thread pool to finish all jobs for this timepoint.
            self.dev_threads.jobs_to_be_done.join()
            
            # Synchronize with all other devices before starting the next timepoint.
            self.device.barrier.wait()

        # --- Shutdown sequence ---
        self.dev_threads.jobs_to_be_done.join()

        # Send sentinel values to terminate the worker threads.
        for _ in range(len(self.dev_threads.threads)):
            self.dev_threads.jobs_to_be_done.put((None, None, None))

        for d_th in self.dev_threads.threads:
            d_th.join()


class ThreadsForEachDevice(object):
    """

    A thread pool for executing script jobs on behalf of a single Device.
    """

    def __init__(self, number_of_threads):
        """
        Initializes the thread pool.

        :param number_of_threads: The number of worker threads to create.
        """
        self.device = None
        self.jobs_to_be_done = Queue(number_of_threads)
        self.threads = []
        self.create_threads(number_of_threads)
        self.start_threads()

    def create_threads(self, number_of_threads):
        """Creates the worker threads."""
        for _ in range(number_of_threads):
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

    def start_threads(self):
        """Starts the worker threads."""
        for i_th in self.threads:
            i_th.start()

    def execute(self):
        """
        The target function for worker threads.
        
        It continuously fetches jobs from a queue, gathers data from neighboring
        devices, runs a script, and distributes the results back.
        """
        while True:
            # Get a job from the queue. Blocks until a job is available.
            neighbours, script, location = self.jobs_to_be_done.get()
            
            # Sentinel check for thread termination.
            if neighbours is None and script is None:
                self.jobs_to_be_done.task_done()
                return

            data_for_script = []
            
            # Gather data from all neighboring devices.
            # This involves direct, thread-safe method calls on other Device objects.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        data_for_script.append(data)
            
            # Gather data from the local device.
            data = self.device.get_data(location)
            if data is not None:
                data_for_script.append(data)

            if data_for_script:
                # Execute the script with the collected data.
                scripted_data = script.run(data_for_script)

                # Broadcast the computed data back to all neighbors.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, scripted_data)
                
                # Update the local device's data.
                self.device.set_data(location, scripted_data)

            self.jobs_to_be_done.task_done()
