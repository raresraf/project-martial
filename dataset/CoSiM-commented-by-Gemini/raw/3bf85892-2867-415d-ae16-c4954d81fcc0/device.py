# -*- coding: utf-8 -*-
"""
@brief A distributed device simulation framework with critical concurrency flaws.
@file device.py

This module implements a simulation of interconnected devices that process
sensor data in parallel time steps. It utilizes a queue-based worker pool
for task execution and a barrier for synchronization.

Algorithm:
- A supervisor node orchestrates the simulation by assigning computational scripts
  to a network of Device objects.
- Each Device operates on its own thread, managing a pool of WorkerThreads.
- Tasks (scripts) for a given time step are placed into a shared queue.
- Worker threads consume tasks, acquiring locks based on data location to
  prevent race conditions during script execution.
- A ReusableBarrier is intended to synchronize all DeviceThreads at the end of
  each time step.

WARNING:
This implementation contains multiple severe concurrency flaws that make it
unsuitable for production use.
1.  **Barrier Misuse:** The main `DeviceThread` does not wait for its worker
    threads to finish their tasks before proceeding to the barrier. This
    defeats the purpose of the barrier, leading to a race condition where
    devices may proceed to the next time step before the current one is complete.
2.  **Buggy Barrier Implementation:** The custom `ReusableBarrier` is not
    correctly implemented and is prone to race conditions (the "spurious wakeup"
    problem is not handled correctly, and it's vulnerable to other race conditions).
3.  **Racy Initialization:** The `setup_devices` method has a race condition where
    multiple devices could attempt to initialize the shared barrier and locks
    simultaneously.
"""

from threading import Event, Thread, Lock
from Queue import Queue
# This local import suggests a project structure that was not preserved.
# A flawed implementation is provided at the end of this file.
from reusable_barrier_condition import ReusableBarrier


class Device(object):
    """
    Represents a single device node in the distributed simulation.

    Each device manages its own data, a set of computational scripts, and a
    pool of worker threads to execute them. It communicates with a central
    supervisor and synchronizes with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device, its worker queue, and its main control thread.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor readings.
            supervisor (Supervisor): The central supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.location_locks = {}
        self.barrier = None
        self.num_threads = 8
        self.queue = Queue(self.num_threads)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device for debugging."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier and location-based locks for all devices.

        This method is intended to be called on one device to initialize
        shared resources for the entire simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.

        WARNING: This setup protocol is racy. The first device to execute this
        block creates the shared resources and distributes them. There is no
        guaranteed master or execution order, which can lead to non-deterministic
        initialization of the shared `barrier` and `location_locks` if multiple
        threads call this concurrently.
        """
        # Pre-condition: self.barrier is None. This check is not atomic.
        if self.barrier is None:
            # Initializes a barrier for all devices.
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = self.barrier
                # Creates a lock for each unique sensor data location.
                for location in device.sensor_data:
                    if location not in self.location_locks:
                        self.location_locks[location] = Lock()
            # Distributes the created locks to all other devices.
            for device in devices:
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """
        Assigns a computational script to be executed in the current time step.

        If the script is None, it signals that no more scripts are coming for
        this time step, and the device can proceed.

        Args:
            script (Script): The script object to be executed.
            location (str): The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Invariant: Setting timepoint_done signals the DeviceThread to start
            # processing the accumulated scripts for the time step.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location.

        Note: This method is not thread-safe by itself. Callers must ensure
        proper synchronization, typically by acquiring the appropriate
        location lock.

        Args:
            location (str): The data location to read from.

        Returns:
            The data at the given location, or None if not found.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location.

        Note: This method is not thread-safe. Callers must ensure proper
        synchronization.

        Args:
            location (str): The data location to write to.
            data: The new data value.
        """
        # Pre-condition: The location should exist in the sensor_data dictionary
        # for the write to be effective.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully shuts down the device by joining its control thread."""
        self.thread.join()

class WorkerThread(Thread):
    """
    A worker thread that processes script execution tasks from a queue.

    It continuously fetches tasks, executes them while holding the
    appropriate lock, and terminates when a sentinel value is received.
    """

    def __init__(self, queue, device):
        """
        Initializes the worker thread.

        Args:
            queue (Queue): The shared queue from which to fetch tasks.
            device (Device): The parent device, used to access shared resources.
        """
        Thread.__init__(self)
        self.queue = queue
        self.device = device

    def run(self):
        """
        The main loop for the worker thread.

        Continuously fetches tasks from the queue and executes them. A sentinel
        value of (None, None, None) signals the thread to terminate.
        """
        # Invariant: The loop continues as long as no sentinel is received.
        while True:
            # Blocks until a task is available in the queue.
            data_tuple = self.queue.get()

            # Pre-condition: Check if the task is the shutdown sentinel.
            if data_tuple == (None, None, None):
                break

            script, location, neighbours = data_tuple

            # Block Logic: Ensures that the script execution for a given location
            # is atomic across all threads and devices that might access it.
            with self.device.location_locks[location]:
                script_data = []
                # Gathers data from neighboring devices for the computation.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Includes the device's own data in the computation.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: Only run the script if there is data to process.
                if script_data:
                    # Executes the computational script.
                    result = script.run(script_data)
                    # Propagates the result to neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    # Updates the device's own data with the result.
                    self.device.set_data(location, result)


class DeviceThread(Thread):
    """
    The main control thread for a single Device.

    This thread orchestrates the device's lifecycle, managing the worker pool
    and synchronizing with other devices at each time step.
    """

    def __init__(self, device):
        """Initializes the main control thread for a device."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main, flawed, lifecycle loop for the device.

        This method initializes the worker thread pool and then enters a loop
        to process tasks for each time step.

        WARNING: This method contains a critical race condition. It puts tasks
        on the queue for its workers but immediately proceeds to the barrier
        `wait()` without waiting for the workers to complete. The purpose of
        the barrier (to wait for all work in a time step to be done) is defeated.
        """
        threads = []
        # Block Logic: Initializes and starts the pool of worker threads.
        for i in range(self.device.num_threads):
            thread = WorkerThread(self.device.queue, self.device)
            threads.append(thread)
            threads[i].start()

        # Invariant: The loop continues as long as the supervisor provides neighbours.
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            # A None value for neighbours is the signal to terminate the simulation.
            if neighbours is None:
                break

            # 1. Wait for the signal that all scripts for the time step have been assigned.
            self.device.timepoint_done.wait()

            # 2. Add all assigned script tasks to the worker queue.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location, neighbours))

            # 3. CRITICAL FLAW: Proceeds immediately to the barrier without
            #    ensuring that the tasks just enqueued have been processed.
            #    A `self.device.queue.join()` is missing here, which would
            #    block until all worker threads have completed their tasks.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()

        # Block Logic: Gracefully shuts down the worker thread pool by sending
        # a sentinel value to each worker.
        for i in range(self.device.num_threads):
            self.device.queue.put((None, None, None))
        for i in range(self.device.num_threads):
            threads[i].join()


from threading import Condition

class ReusableBarrier(object):
    """
    A buggy and fundamentally flawed implementation of a reusable barrier.

    This barrier uses a `threading.Condition` variable to manage thread
    synchronization. It is intended to block a specified number of threads
    until all of them have reached the barrier.

    WARNING: This implementation is not safe. It is susceptible to race
    conditions, particularly the "lost wakeup" problem. If threads that are
    released from a `wait()` call loop around and call `wait()` again before
    all other threads have woken up from the first call, it can lead to a
    premature `notify_all()` or cause threads to miss a notification,
    resulting in deadlock. A correct implementation requires more complex state
    management, often using a second condition or a generation counter.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a fixed number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()


    def wait(self):
        """
        Causes a thread to block until all `num_threads` have called wait.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # Pre-condition: Check if this is the last thread to arrive at the barrier.
        if self.count_threads == 0:
            # Block Logic: If the last thread has arrived, it is responsible for
            # waking up all other waiting threads and then resetting the barrier
            # for the next use.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Block Logic: If not the last thread, wait for a notification from
            # the last thread.
            self.cond.wait()
        self.cond.release()