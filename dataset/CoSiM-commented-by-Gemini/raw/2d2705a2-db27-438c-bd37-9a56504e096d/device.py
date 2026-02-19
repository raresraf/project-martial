"""
@file device.py
@brief A distributed device simulation using a queue-based worker thread pool.
@details This module defines a device that uses a master thread (`DeviceThread`) to produce tasks
and a pool of `WorkerThread`s to consume and execute them from a shared queue.
The implementation contains several critical synchronization bugs.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue
# Note: The file imports from 'reusable_barrier_condition' but also defines its own 'ReusableBarrier' at the bottom.
from reusable_barrier_condition import ReusableBarrier


class Device(object):
    """
    @brief Represents a single device in the simulation.
    @details It initializes a master thread and a queue for dispatching work to a pool of worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        # Centralized dictionary of locks, one for each data location.
        self.location_locks = {}
        # Shared barrier for inter-device synchronization.
        self.barrier = None
        self.num_threads = 8
        # A queue to pass jobs from the main thread to the worker threads.
        self.queue = Queue(self.num_threads)
        self.thread.start()

    def __str__(self):
        """@brief Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared barrier and location locks for all devices.
        @details The first device to call this method initializes the shared resources and
        distributes them to all other devices in the simulation.
        """
        if self.barrier is None:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = self.barrier
                for location in device.sensor_data:
                    if location not in self.location_locks:
                        self.location_locks[location] = Lock()
            # Ensure all devices share the same lock dictionary.
            for device in devices:
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """@brief Adds a script to the list or signals that the timepoint is done."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """@brief Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """@brief Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """@brief Shuts down the device by joining its main thread."""
        self.thread.join()

class WorkerThread(Thread):
    """
    @brief A worker thread that consumes script execution jobs from a queue.
    @warning This implementation is buggy as it never calls `self.queue.task_done()`,
    which would prevent a `self.queue.join()` call from ever completing.
    """
    def __init__(self, queue, device):
        Thread.__init__(self)
        self.queue = queue
        self.device = device

    def run(self):
        """@brief Main loop to get and process jobs from the queue."""
        while True:
            data_tuple = self.queue.get()

            # A None tuple is a "poison pill" to signal termination.
            if data_tuple == (None, None, None):
                break

            # Acquire lock for the specific data location.
            self.device.location_locks[data_tuple[1]].acquire()
            
            script_data = []
            # Gather data from neighbors and the local device.
            for device in data_tuple[2]:
                data = device.get_data(data_tuple[1])
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(data_tuple[1])
            if data is not None:
                script_data.append(data)
            
            # If data is available, run the script and update data on all relevant devices.
            if script_data:
                result = data_tuple[0].run(script_data)
                for device in data_tuple[2]:
                    device.set_data(data_tuple[1], result)
                self.device.set_data(data_tuple[1], result)
            
            # Release the lock for the location.
            self.device.location_locks[data_tuple[1]].release()
            # CRITICAL BUG: self.queue.task_done() is missing here.

class DeviceThread(Thread):
    """
    @brief The master thread for a device, which produces jobs for worker threads.
    @warning This implementation is buggy. It does not wait for the worker threads to finish their
    jobs before proceeding to the barrier, as there is no `self.device.queue.join()` call.
    This breaks the intended synchronization of the computation phase.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        threads = []
        # Create and start the pool of worker threads.
        for i in range(self.device.num_threads):
            thread = WorkerThread(self.device.queue, self.device)
            threads.append(thread)
            threads[i].start()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal from supervisor.

            # Wait for the signal that all scripts for the timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Add all script jobs to the queue for the worker threads.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location, neighbours))
            
            # CRITICAL BUG: Missing a call to `self.device.queue.join()` here.
            # The thread proceeds to the barrier without waiting for workers to finish.
            
            # Synchronize with other devices.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()

        # Send poison pills to terminate the worker threads.
        for i in range(self.device.num_threads):
            self.device.queue.put((None, None, None))
        # Wait for all worker threads to terminate.
        for i in range(self.device.num_threads):
            threads[i].join()


class ReusableBarrier(object):
    """
    @brief A barrier implementation using a `threading.Condition`.
    @warning This is not a safe reusable barrier due to potential race conditions (wake-up race).
    A thread could loop and re-enter `wait()` before all other threads have woken up from the previous wait.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()
