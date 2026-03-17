"""
This module implements a distributed device simulation using a thread pool (worker) model.
Each device operates on a set of local data and can execute scripts that also involve
data from neighboring devices. Synchronization is managed through a system-wide barrier
and per-location semaphores to ensure data consistency during concurrent script execution.
"""
from threading import Semaphore, Lock, Event, Thread
from Queue import Queue
# Assumes the presence of a 'barrier.py' file with a ReusableBarrier implementation.
from barrier import ReusableBarrier

class Device(object):
    """
    Represents a single device in the simulation. It holds sensor data, a list of
    scripts to execute, and manages a main control thread (`DeviceThread`).
    """
    
    # Defines the maximum number of distinct data locations in the system.
    MAX_LOCATIONS = 100
    # A named constant for clarity.
    CONST_ONE = 1

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device object.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data.
            supervisor (object): The central supervisor object that manages the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Event to signal that a timepoint has concluded and scripts are ready to be processed.
        self.timepoint_done = Event()
        # Holds the list of per-location semaphores, to be populated by setup_devices.
        self.semaphore = []
        self.thread = DeviceThread(self)
        # Initializes a placeholder barrier, to be replaced by the shared one.
        self.barrier = ReusableBarrier(Device.CONST_ONE)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def get_distributed_objs(self, barrier, semaphore):
        """
        Callback method to receive and store the globally shared synchronization objects.
        This is called by the `setup_devices` method on all devices.
        """
        self.barrier = barrier
        self.semaphore = semaphore
        # The device's main thread is only started after the synchronization objects are in place.
        self.thread.start()

    def setup_devices(self, devices):
        """
        A centralized setup routine, intended to be called by one device (device_id 0).
        It creates and distributes the shared barrier and per-location semaphores
        to all devices in the simulation.
        """
        # Pre-condition: This block should only be executed once for the entire system.
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            semaphore = []
            i = Device.MAX_LOCATIONS
            # Block Logic: Creates a unique semaphore for each possible data location.
            # This is the core mechanism for preventing race conditions on shared data.
            while i > 0:
                semaphore.append(Semaphore(value=Device.CONST_ONE))
                i = i - 1

            # Invariant: Distributes the same set of synchronization objects to every device.
            for device in devices:
                device.get_distributed_objs(barrier, semaphore)

    def assign_script(self, script, location):
        """
        Assigns a script to the device. Called by the supervisor. A `None` script
        indicates that all scripts for the current timepoint have been assigned.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # All scripts for the timepoint are assigned; signal the device thread to start processing.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data for a given location from the device's local sensor data."""
        if location in self.sensor_data:
            obj = self.sensor_data[location]
        else:
            obj = None
        return obj

    def set_data(self, location, data):
        """Updates data for a given location in the device's local sensor data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Ensures the main device thread is properly joined upon simulation shutdown."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. It manages a pool of worker threads and a queue
    to process scripts for each simulation timepoint.
    """
    
    THREADS_TO_START = 8
    MAX_SCRIPTS = 100

    def __init__(self, device):
        """Initializes the thread, its worker pool, and the script queue."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = []
        self.neighbours = []
        self.queue = Queue(maxsize=DeviceThread.MAX_SCRIPTS)

    def run(self):
        """The main execution loop for the device, organized by timepoints."""
        lock = Lock()
        # Block Logic: Initializes and starts a fixed pool of worker threads.
        for i in range(DeviceThread.THREADS_TO_START):
            self.threads.append(WorkerThread(self, i, self.device, self.queue, lock))
            self.threads[i].setDaemon(True)
        for thread in self.threads:
            thread.start()

        # This loop represents the discrete time steps of the simulation.
        while True:
            # Gets the current set of neighbors from the supervisor for this timepoint.
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                # A `None` value for neighbors is the signal to terminate the simulation.
                break

            # Block Logic: Waits for the supervisor to signal that all scripts for this timepoint are assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Enqueues all assigned scripts for the worker threads to process.
            for script in self.device.scripts:
                self.queue.put(script)
            # Synchronization Point: Waits for the worker threads to process all items in the queue.
            self.queue.join()

            # Resets the event for the next timepoint.
            self.device.timepoint_done.clear()

            # Synchronization Point: Waits at the barrier until all other devices in the
            # simulation have also completed their work for the current timepoint.
            self.device.barrier.wait()

        # Block Logic: Ensures all worker threads are cleaned up upon termination.
        for thread in self.threads:
            thread.join()


class WorkerThread(Thread):
    """
    A worker thread that continuously pulls scripts from a shared queue and executes them.
    """

    def __init__(self, master, worker_id, device, queue, lock):
        """
        Initializes the worker.
        Args:
            master (DeviceThread): The parent controller thread.
            worker_id (int): A unique ID for this worker within its pool.
            device (Device): The parent device.
            queue (Queue): The shared queue from which to pull scripts.
            lock (Lock): A lock for safely accessing the queue (though the queue itself is thread-safe).
        """
        Thread.__init__(self, name="Worker Thread %d %d" % (worker_id, device.device_id))
        self.master = master
        self.device = device
        self.queue = queue
        self.lock = lock

    def run(self):
        """The main loop for the worker, processing scripts from the queue."""

        while True:
            # The lock here is used to ensure atomicity of checking if the queue is empty and getting an item.
            # While `queue.get()` is blocking and thread-safe, this structure is explicitly chosen.
            self.lock.acquire()
            value = self.queue.empty()
            if value is False:
                (script, location) = self.queue.get()
            self.lock.release()

            if value is False:
                script_data = []

                # Synchronization Point: Acquires the specific semaphore for the data location.
                # This prevents any other thread in the entire simulation from processing the same location.
                self.device.semaphore[location].acquire()

                # Block Logic: Gathers data from neighbors and the local device.
                for device in self.master.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.master.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # The script is only run if there is data to process.
                if script_data != []:
                    # Executes the script's logic.
                    result = script.run(script_data)

                    # Block Logic: Propagates the result back to all neighbors and the local device.
                    for device in self.master.neighbours:
                        device.set_data(location, result)
                    
                    self.master.device.set_data(location, result)

                # Releases the semaphore, allowing other threads to work on this location.
                self.device.semaphore[location].release()
                # Signals that the task taken from the queue is complete.
                self.queue.task_done()
            
            # Termination condition for the worker thread loop.
            if self.master.neighbours is None:
                break