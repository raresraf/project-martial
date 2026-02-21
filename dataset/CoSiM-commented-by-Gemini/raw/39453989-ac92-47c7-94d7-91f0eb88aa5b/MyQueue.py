"""
Defines a distributed device simulation framework with a worker-pool architecture.

This file contains several classes for simulating a network of devices.
- `MyQueue`: A thread pool that processes script execution tasks.
- `ReusableBarrier`: A custom thread barrier for synchronization.
- `Device`: Represents a node in the network.
- `DeviceThread`: The main control loop for a device, which dispatches tasks
  to the `MyQueue`.

The overall architecture uses a persistent pool of worker threads (`MyQueue`) to
execute computational tasks, decoupled from the main device lifecycle thread.
"""

from Queue import Queue
from threading import Thread, Lock, Semaphore, Event


class MyQueue():
    """A worker thread pool for executing device scripts.

    This class manages a queue of tasks and a fixed number of worker threads
    that continuously pull tasks from the queue and execute them.
    """

    def __init__(self, num_threads):
        """Initializes the queue and starts the worker threads.

        Args:
            num_threads (int): The number of worker threads to create in the pool.
        """
        self.queue = Queue(num_threads)
        self.threads = []
        self.device = None  # The device this queue is serving.

        # Create and start the pool of worker threads.
        for _ in xrange(num_threads):
            thread = Thread(target=self.run)
            self.threads.append(thread)
            thread.start()

    def run(self):
        """The target method for worker threads.

        This method runs in an infinite loop, processing tasks from the queue.
        A task consists of gathering data from neighboring devices, executing a
        script, and propagating the results. A sentinel value of (None, None, None)
        signals the thread to terminate.
        """
        while True:
            # Block until a task is available in the queue.
            neighbours, script, location = self.queue.get()

            # Sentinel value check for thread termination.
            if neighbours is None and script is None:
                self.queue.task_done()
                return

            script_data = []
            # Gather data from neighbors.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

            # Gather data from the local device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Execute the computational script.
                result = script.run(script_data)

                # Disseminate the result to neighbors.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)

                # Update the local device's data.
                self.device.set_data(location, result)

            self.queue.task_done()

    def finish(self):
        """Shuts down the worker thread pool gracefully."""
        # Wait for all pending tasks in the queue to be completed.
        self.queue.join()

        # Send a sentinel value for each thread to signal termination.
        for _ in xrange(len(self.threads)):
            self.queue.put((None, None, None))

        # Wait for all worker threads to finish.
        for thread in self.threads:
            thread.join()


class ReusableBarrier():
    """A custom implementation of a reusable, two-phase barrier."""
    
    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)  # Semaphore for the first phase.
        self.threads_sem2 = Semaphore(0)  # Semaphore for the second phase.

    def wait(self):
        """Causes a thread to wait at the barrier.

        The barrier uses two phases to ensure that it can be reused without
        race conditions. All threads must enter and exit the first phase
        before any can exit the second phase of the next `wait()` cycle.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier wait.

        Args:
            count_threads (list[int]): A mutable integer (as a list) to count arrivals.
            threads_sem (Semaphore): The semaphore for this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all other waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads  # Reset for reuse.
        threads_sem.acquire()  # All threads wait here until released.


class Device(object):
    """Represents a node in the distributed simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.location_locks = {location: Lock() for location in self.sensor_data}
        self.scripts_available = False
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the barrier for the device group.

        The device with ID 0 is designated as the master and is responsible
        for creating and distributing the shared `ReusableBarrier`.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to the device for a later timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            # A None script signals all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Intended to atomically get data, but contains a bug.
        It acquires a lock but never releases it, which will lead to a deadlock
        if called more than once on the same location.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Intended to atomically set data, but contains a bug.
        It releases a lock that was not acquired within this method, which
        will raise a `ThreadError`. It relies on `get_data` having been
        called first by the same conceptual task.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
        else:
            return None

    def shutdown(self):
        """Shuts down the device thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a device."""

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = MyQueue(8)  # Each device has its own worker pool.

    def run(self):
        """Main device lifecycle.

        Dispatches tasks to its `MyQueue` worker pool and synchronizes with
        other devices at a barrier at the end of each timepoint.
        """
        self.queue.device = self.device
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals shutdown.
                break

            # Inner loop to handle script assignments for one timepoint.
            while True:
                if self.device.scripts_available or self.device.timepoint_done.wait():
                    if self.device.scripts_available:
                        self.device.scripts_available = False
                        # Dispatch all available scripts to the worker queue.
                        for (script, location) in self.device.scripts:
                            self.queue.queue.put((neighbours, script, location))
                    else:
                        # End of the timepoint.
                        self.device.timepoint_done.clear()
                        self.device.scripts_available = True
                        break

            # Wait for all dispatched tasks for this timepoint to be processed.
            self.queue.queue.join()
            # Wait for all other devices to finish their timepoint.
            self.device.barrier.wait()

        # Cleanly shut down the worker pool.
        self.queue.finish()
