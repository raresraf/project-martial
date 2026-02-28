# -*- coding: utf-8 -*-
"""
Simulates a distributed network of computational devices using a custom thread pool
and synchronization primitives.

This module defines a framework where `Device` objects operate concurrently to process
shared data. It features a custom `MyThreadPool` for managing worker threads and
a `MyBarrier` class for synchronizing devices at each computational step. This
approach differs from standard library equivalents by providing bespoke implementations
of common concurrency patterns.

Key Components:
- **MyBarrier**: A custom barrier implementation using a `Condition` variable.
- **MyThreadPool**: A custom thread pool that manages a queue of tasks and a set of worker threads.
- **Device**: Represents a node in the network, utilizing the thread pool to execute scripts.
- **DeviceThread**: The main control loop for a `Device`, orchestrating script dispatch and synchronization.
"""

from threading import Thread, Condition, Event, Lock
from Queue import Queue

def stop():
    """
    A sentinel function used to signal worker threads in the thread pool to terminate.
    """
    return

class MyBarrier(object):
    """
    A custom, reusable barrier implementation using a `Condition` variable.

    This barrier blocks a set number of threads until all of them have called the `wait()`
    method, at which point all threads are released and the barrier is reset.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        # Counter for threads currently waiting.
        self.count_threads = self.num_threads
        # Condition variable to manage waiting and notification.
        self.cond = Condition()

    def wait(self):
        """
        Causes the calling thread to block until all `num_threads` have called `wait`.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # If this is the last thread to arrive...
        if self.count_threads == 0:
            # ...notify all waiting threads to wake up.
            self.cond.notify_all()
            # Reset the counter for the next use of the barrier.
            self.count_threads = self.num_threads
        else:
            # Otherwise, wait to be notified.
            self.cond.wait()
        self.cond.release()


class MyWorkerThread(Thread):
    """
    A worker thread for the custom thread pool.

    It continuously fetches tasks from a shared queue and executes them. The thread
    terminates when it receives a specific `stop` sentinel function.
    """
    def __init__(self, tasks_list, lock):
        """
        Initializes and starts the worker thread.

        Args:
            tasks_list (Queue): The shared queue from which to fetch tasks.
            lock (Lock): A lock used for coordinating the shutdown sequence.
        """
        Thread.__init__(self)
        self.tasks_list = tasks_list
        self.daemon = True # Allows the main program to exit even if workers are alive.
        self.stop = False
        self.lock = lock
        self.start()

    def run(self):
        """
        The main loop of the worker thread.
        """
        while True:
            # Block until a task is available in the queue.
            function, params = self.tasks_list.get()
            
            # Check for the shutdown sentinel.
            if function is stop:
                self.tasks_list.task_done()
                self.lock.release() # Signal that one stop task has been processed.
                break

            # Execute the task.
            function(*params)

            # Mark the task as completed.
            self.tasks_list.task_done()

class MyThreadPool(object):
    """
    A custom thread pool implementation.

    Manages a fixed number of worker threads and a queue of tasks.
    """
    def __init__(self, no_threads):
        """
        Initializes the thread pool and creates the worker threads.

        Args:
            no_threads (int): The number of worker threads in the pool.
        """
        self.no_threads = no_threads
        self.tasks_list = Queue(no_threads)
        self.worker_list = []
        self.lock = Lock()
        
        # Create and start the worker threads.
        for _ in xrange(no_threads):
            self.worker_list.append(MyWorkerThread(self.tasks_list, self.lock))

    def add(self, function, *params):
        """
        Adds a task to the thread pool's queue.

        Args:
            function: The function to execute.
            *params: The arguments to pass to the function.
        """
        self.tasks_list.put((function, params))

    def wait(self):
        """
        Waits for all tasks in the queue to complete and then shuts down the workers.
        """
        # Block until all items in the queue have been gotten and processed.
        self.tasks_list.join()
        # To shut down, add one 'stop' sentinel for each worker thread.
        for i in xrange(self.no_threads):
            self.lock.acquire() # This lock ensures we wait for each stop task to be processed.
            self.add(stop, None)
        # Wait for all worker threads to terminate.
        for i in xrange(self.no_threads):
            self.worker_list[i].join()


class Device(object):
    """
    Represents a computational device in the distributed network.

    This device uses a `MyThreadPool` to execute assigned scripts concurrently.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal a script has been received (not used in this version's logic).
        self.script_received = Event()
        self.scripts = []
        # Event to signal that a "timepoint" (batch of scripts) has been assigned.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        self.barrier = None
        self.locks = []
        # Get the total number of unique locations from the supervisor.
        self.no_locations = self.supervisor.supervisor.testcase.num_locations
        
        # Each device has its own thread pool.
        self.pool = MyThreadPool(8)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the setup of all devices, creating a shared barrier and locks.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            # Create a single barrier for all devices.
            barrier = MyBarrier(len(devices))

            # Create a lock for each data location.
            for i in xrange(self.no_locations):
                self.locks.append(Lock())

            # Distribute the shared barrier and locks to all devices.
            for i in xrange(len(devices)):
                devices[i].barrier = barrier
                devices[i].locks = self.locks


    def assign_script(self, script, location):
        """
        Assigns a script or signals the end of a script batch.

        Args:
            script: The script to execute, or None to signal the end of a timepoint.
            location: The location the script applies to.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts for the current timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data[location] if location \
                in self.sensor_data else None

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device, dispatching tasks to a thread pool.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_script(self, params):
        """
        The logic for executing a single script. This method is called by a worker thread.

        Args:
            params (tuple): A tuple containing (neighbours, script, location).
        """
        neighbours, script, location = params
        # Acquire the lock for the specific location to ensure exclusive access.
        self.device.locks[location].acquire()

        script_data = []
        
        # Invariant: Gather data from all neighbors.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Gather data from the local device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        
        # Invariant: Only run the script if there is data.
        if script_data != []:
            result = script.run(script_data)
            
            # Broadcast the result to neighbors and update local data.
            for device in neighbours:
                device.set_data(location, result)
            self.device.set_data(location, result)

        # Release the lock.
        self.device.locks[location].release()

    def run(self):
        """
        The main loop of the device thread.
        """
        while True:
            # Get the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # A None response signals simulation shutdown.
            if neighbours is None:
                # Wait for the thread pool to finish all tasks and shut down.
                self.device.pool.wait()
                break

            # Pre-condition: Wait until the supervisor signals that a full timepoint's
            # worth of scripts has been assigned.
            self.device.timepoint_done.wait()

            # Invariant: For the current timepoint, dispatch all assigned scripts to the thread pool.
            for (script, location) in self.device.scripts:
                params = (neighbours, script, location)
                self.device.pool.add(self.run_script, params)
            
            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()
            # Wait at the barrier for all other devices to finish their timepoint processing.
            self.device.barrier.wait()
