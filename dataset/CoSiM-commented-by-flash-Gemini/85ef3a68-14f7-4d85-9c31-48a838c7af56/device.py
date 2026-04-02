"""
@file device.py
@brief Implements a multi-threaded, distributed device simulation framework.
@author [Original Author - inferred]
@date [Original Date - inferred]

This file defines the core components for a simulation of interacting devices.
It uses a combination of threading, thread pools, and synchronization primitives
(Events, Locks, Barriers) to model concurrent operations on shared data resources.

The architecture consists of:
- `Device`: Represents a node in the network, holding sensor data and assigned scripts.
- `DeviceThread`: The main control loop for a device, managing execution timepoints.
- `ThreadPool`: Manages a pool of `Worker` threads to execute tasks in parallel.
- `Worker`: A thread that performs a single task, including data aggregation from
           neighboring devices and updating shared state under locks.

Domain: Concurrent Programming, Distributed Systems Simulation, Parallel Algorithms.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
from thread_pool import ThreadPool

class Device(object):
    """
    Represents a single device in the simulated network, managing its own state,
    data, and execution thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary representing the device's local sensor readings.
        @param supervisor An external object that manages the network topology (e.g., neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Event to signal that a new timepoint with assigned scripts is ready to be processed.
        self.timepoint_done = Event()

        # A shared barrier for synchronizing all devices at the end of a timepoint.
        self.barrier = None

        # A shared dictionary of locks, mapping a location to a Lock object.
        # This is crucial for ensuring exclusive access to a location across all devices.
        self.location_locks = {}

        # Each device has its own thread for its main execution loop.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and shares synchronization primitives among a list of devices.

        This method should be called once to ensure all devices in the simulation
        share the same barrier and the same set of location-based locks.

        @param devices A list of all Device objects in the simulation.
        """
        # Block Logic: Ensures that the barrier and locks are initialized only once.
        # Invariant: All devices in the 'devices' list will share the same barrier
        # and location_locks dictionary after this method completes.
        if self.barrier is None:
            # A reusable barrier for synchronizing all devices.
            self.barrier = ReusableBarrierCond(len(devices))

            # Pre-condition: self.location_locks is an empty dictionary.
            # Invariant: self.location_locks is populated with Lock objects for every
            # unique location across all devices.
            for device in devices:
                device.barrier = self.barrier

                # Aggregate all unique locations from all devices and create a shared lock for each.
                for location in device.sensor_data:
                    if location not in self.location_locks:
                        self.location_locks[location] = Lock()
                
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location for the next timepoint.

        @param script The script to execute.
        @param location The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Functional Utility: A None script is the signal from the supervisor
            # that all scripts for the current timepoint have been assigned.
            # This unblocks the DeviceThread's main loop.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Gracefully shuts down the device by stopping its main thread.
        """
        # This ensures the DeviceThread finishes its execution before the main program exits.
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device, orchestrating task execution and
    synchronization across timepoints.
    """
    
    # Static Configuration: The number of worker threads in the device's thread pool.
    NO_CORES = 8

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        @param device The parent Device object this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        # Each device manages its own pool of worker threads to execute script tasks.
        self.thread_pool = ThreadPool(self.device, DeviceThread.NO_CORES)

    def run(self):
        """
        The main execution loop of the device.
        """
        # Block Logic: The main simulation loop continues as long as the supervisor
        # provides neighbor information.
        # Invariant: Each iteration of the loop represents one full timepoint.
        while True:
            # Determine neighbors for the upcoming timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Functional Utility: A None value for neighbours is the shutdown signal.
                # Propagate the shutdown signal to all worker threads.
                for _ in xrange(DeviceThread.NO_CORES):
                    self.thread_pool.submit_task(None, None, None)
                
                self.thread_pool.end_workers()
                break

            # Block Logic: Wait until the supervisor has finished assigning all scripts for the timepoint.
            # This event is triggered by the `assign_script` method when a None script is passed.
            self.device.timepoint_done.wait()

            # Pre-condition: self.device.scripts contains all tasks for the current timepoint.
            # Invariant: All scripts are submitted to the thread pool for execution.
            for (script, location) in self.device.scripts:
                self.thread_pool.submit_task(script, location, neighbours)

            # Reset for the next timepoint.
            self.device.timepoint_done.clear()

            # Block Logic: Synchronize with all other devices.
            # This is a critical step that ensures no device starts the next timepoint
            # until all devices have finished processing the current one.
            self.device.barrier.wait()




from threading import Thread
from Queue import Queue

class Worker(Thread):
    """
    A worker thread that executes tasks from a shared queue.
    """

    def __init__(self, device, task_queue):
        """
        Initializes a Worker instance.

        @param device The parent Device object.
        @param task_queue The shared queue from which to pull tasks.
        """
        Thread.__init__(self)
        self.device = device
        self.task_queue = task_queue

    def run(self):
        """The main loop for the worker thread."""
        while True:
            # Blocks until a task is available in the queue.
            script, location, neighbours = self.task_queue.get()

            # Shutdown protocol: A (None, None, None) tuple is the poison pill.
            if (script is None and location is None and neighbours is None):
                self.task_queue.task_done()
                break

            # Block Logic: Critical section to ensure data integrity for a specific location.
            # The lock is shared among all devices, so only one worker thread in the
            # entire system can execute a task for this 'location' at a time.
            # Pre-condition: A task for 'location' is ready.
            # Invariant: The 'run_task' method has exclusive access to the data at 'location'.
            with self.device.location_locks[location]:
                self.run_task(script, location, neighbours)

            # Signal that the task is complete.
            self.task_queue.task_done()

    def run_task(self, script, location, neighbours):
        """
        Executes a single script task.
        
        This involves gathering data from the parent device and its neighbors,
        running the script, and then disseminating the results back.
        """
        script_data = []
        
        # Block Logic: Data Aggregation Phase.
        # Gathers data from all neighbors at the specified location.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        # Also gather data from the local device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Functional Utility: Execute the script with the aggregated data.
            result = script.run(script_data)

            # Block Logic: Result Dissemination Phase.
            # The result of the script is written back to the local device and all neighbors.
            # This occurs within the critical section, ensuring consistency.
            for device in neighbours:
                device.set_data(location, result)

            self.device.set_data(location, result)


class ThreadPool(object):
    """
    A simple thread pool to manage and reuse a fixed number of Worker threads.
    """

    def __init__(self, device, no_workers):
        """
        Initializes the ThreadPool.

        @param device The parent device, passed to each worker.
        @param no_workers The number of worker threads to create.
        """
        self.device = device
        self.no_workers = no_workers
        
        self.task_queue = Queue(no_workers)
        self.workers = []
        self.initialize_workers()

    def initialize_workers(self):
        """Creates and starts the worker threads."""
        for _ in xrange(self.no_workers):
            self.workers.append(Worker(self.device, self.task_queue))

        for worker in self.workers:
            worker.start()

    def end_workers(self):
        """Waits for all tasks to be completed and joins all worker threads."""
        # Wait for the queue to be empty.
        self.task_queue.join()

        # Wait for all worker threads to terminate.
        for worker in self.workers:
            worker.join()

    def submit_task(self, script, location, neighbours):
        """Adds a new task to the task queue."""
        self.task_queue.put((script, location, neighbours))
