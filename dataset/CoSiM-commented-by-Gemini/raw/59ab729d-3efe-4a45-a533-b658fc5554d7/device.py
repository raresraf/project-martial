"""
Models a distributed device network using a classic Producer-Consumer pattern.

This script simulates a network where each `Device` contains a producer thread
(`DeviceThread`) and a pool of consumer threads (`WorkerThread`). The producer
waits for a timepoint's work to be defined, then places all tasks onto a
thread-safe `Queue`. The consumers continually pull tasks from this queue and
execute them. The system uses `queue.join()` for intra-device synchronization and
a global barrier for inter-device synchronization between timepoints.
"""

from Queue import Queue
from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    A reusable barrier implemented with two semaphores for two-phase synchronization.

    This prevents race conditions where fast threads could loop around and re-enter
    the barrier before slow threads have exited. The use of a list for the counter
    is an idiom for creating a mutable integer in Python 2.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the two-phase barrier."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Represents a device node, encapsulating a producer-consumer thread model.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device, its work queue, and its threads."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        # Shared resources to be populated by setup_devices.
        self.barrier = None
        self.locks = {} # Shared location-based locks.
        
        # Intra-device communication and worker pool.
        self.queue = Queue()
        self.workers = [WorkerThread(self) for _ in range(8)]

        # The producer thread that feeds the queue.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Start all consumer threads. They will block on queue.get().
        for thread in self.workers:
            thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources for the entire network.
        """
        if self.device_id == 0:
            # The master device creates a global barrier for all devices.
            barrier = ReusableBarrier(len(devices))
            locks = {}

            # Create a unique lock for each data location across all devices.
            for device in devices:
                for location in device.sensor_data:
                    if not location in locks:
                        locks[location] = Lock()

            # Distribute the shared barrier and locks to all devices.
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """Assigns a script to this device's to-do list for the timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # `None` script signals that all work for the timepoint has been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its producer thread."""
        self.thread.join()


class WorkerThread(Thread):
    """
    A consumer thread that continuously fetches and executes work from a Queue.
    """

    def __init__(self, device):
        Thread.__init__(self)
        self.device = device

    def run(self):
        """Main loop: get work from queue, execute, repeat."""
        while True:
            # Blocks until an item is available on the queue.
            item = self.device.queue.get()
            
            # A `None` item is the poison pill to signal termination.
            if item is None:
                break

            (script, location) = item
            
            # Use a 'with' statement for the location-specific lock to ensure it's always released.
            with self.device.locks[location]:
                script_data = []

                # Gather data from neighbors.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Gather data from self.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Execute the script and disseminate the results.
                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

            # Signal to the queue that one task has been completed.
            self.device.queue.task_done()


class DeviceThread(Thread):
    """
    The producer thread for a device. It manages the work for each timepoint.
    """

    def __init__(self, device):
        """Initializes the producer thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main loop for managing timepoints."""
        while True:
            # Get neighbors for the current timepoint.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                break # Shutdown signal.

            # Wait for the supervisor to signal that script assignment is complete.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # --- Producer phase ---
            # Place all assigned scripts for this timepoint onto the work queue.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location))

            # --- Intra-device synchronization ---
            # Block until all items placed on the queue are processed by workers.
            self.device.queue.join()

            # --- Inter-device synchronization ---
            # Block at the global barrier until all devices have finished their timepoint.
            self.device.barrier.wait()

        # --- Shutdown phase ---
        # Send a "poison pill" (None) to each worker thread to make them exit.
        for _ in range(8):
            self.device.queue.put(None)

        # Wait for all worker threads to terminate.
        for thread in self.device.workers:
            thread.join()
