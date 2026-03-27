"""
This module implements a distributed device simulation framework.

The architecture is based on a producer-consumer pattern. Each `Device` has a
main `DeviceThread` (the producer) that receives tasks and puts them on a queue.
A persistent pool of `WorkerThread`s (the consumers) process these tasks
concurrently. Synchronization across all devices is managed by a custom
`ReusableBarrier`.

- Device: A node in the network, holds sensor data and a task queue.
- DeviceThread: The main control thread, acts as the producer.
- WorkerThread: A long-running worker thread, acts as a consumer.
- ReusableBarrier: A custom barrier for synchronizing all devices between cycles.
"""
from threading import Event, Thread, Lock
from Queue import Queue
from reusable_barrier_condition import ReusableBarrier


class Device(object):
    """Represents a single device in the distributed network.

    Each device holds sensor data and manages a task queue which is processed
    by a pool of worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        # Event to signal that all scripts for a cycle have been assigned.
        self.timepoint_done = Event()
        # The main control thread (producer) for this device.
        self.thread = DeviceThread(self)
        # Location-based locks are shared across all devices.
        self.location_locks = {}
        self.barrier = None
        self.num_threads = 8  # Size of the persistent worker thread pool.
        self.queue = Queue(self.num_threads) # Task queue for worker threads.
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared resources across all devices.

        The first device to enter this method creates the shared barrier and
        location locks, which are then assigned to all other devices.
        """
        if self.barrier is None:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = self.barrier
                # Create a lock for each unique sensor location.
                for location in device.sensor_data:
                    if location not in self.location_locks:
                        self.location_locks[location] = Lock()
            # Ensure all devices share the same set of locks.
            for device in devices:
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """Assigns a script to be processed in the current cycle.

        A 'None' script is a sentinel value indicating the end of script
        assignment for the current cycle.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal to the DeviceThread that all scripts have been received.
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets sensor data for a specific location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main control thread."""
        self.thread.join()

class WorkerThread(Thread):
    """A persistent worker thread that consumes and processes tasks from a queue."""

    def __init__(self, queue, device):
        """Initializes the worker thread."""
        Thread.__init__(self)
        self.queue = queue
        self.device = device

    def run(self):
        """Main loop of the worker, processing tasks until a sentinel is received."""
        while True:
            # Block until a task is available on the queue.
            script, location, neighbours = self.queue.get()

            # A 'None' task is the sentinel to terminate the thread.
            if script is None:
                break

            # Acquire the lock for the specific location to ensure data consistency.
            with self.device.location_locks[location]:
                script_data = []
                # Gather data for the location from all neighboring devices.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from the parent device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Execute the script on the collected data.
                    result = script.run(script_data)

                    # Distribute the result back to the parent and all neighbors.
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)


class DeviceThread(Thread):
    """The main control thread for a Device, acting as a task producer."""

    def __init__(self, device):
        """Initializes the main thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop for the device.

        This loop initializes a pool of worker threads, then enters a cycle of
        receiving tasks, queueing them for the workers, and synchronizing with
        other devices.
        """
        threads = []
        # Create and start the persistent pool of worker threads.
        for i in range(self.device.num_threads):
            thread = WorkerThread(self.device.queue, self.device)
            threads.append(thread)
            threads[i].start()

        while True:
            # Get neighbors for the current cycle from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation signal.

            # Wait until all scripts for the cycle have been assigned.
            self.device.timepoint_done.wait()

            # Producer step: Add all assigned scripts to the task queue.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location, neighbours))

            # Synchronize with all other devices before starting the next cycle.
            self.device.barrier.wait()
            # Reset for the next cycle.
            self.device.scripts = []
            self.device.timepoint_done.clear()

        # === Shutdown Sequence ===
        # Put a "poison pill" or sentinel value on the queue for each worker thread.
        for i in range(self.device.num_threads):
            self.device.queue.put((None, None, None))

        # Wait for all worker threads to terminate.
        for i in range(self.device.num_threads):
            threads[i].join()


from threading import Condition

class ReusableBarrier(object):
    """A reusable barrier implementation using a Condition variable."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()


    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                # Last thread to arrive notifies all waiting threads and resets the barrier.
                self.cond.notify_all()
                self.count_threads = self.num_threads
            else:
                # Wait to be notified by the last thread.
                self.cond.wait()
