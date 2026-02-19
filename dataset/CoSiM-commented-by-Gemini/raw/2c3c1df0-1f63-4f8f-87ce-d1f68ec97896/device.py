"""
@file device.py
@brief Implements a distributed device simulation using a thread pool for script execution.
@details This module defines a simulated environment for a network of devices. Each device runs a single main thread
to coordinate with a supervisor and dispatches computational work (scripts) to a pool of worker threads.
Synchronization between devices is managed by a shared barrier, and data access is controlled by shared locks,
both of which are initialized by a designated "root" device.

Note: This file appears to contain definitions for classes (ReusableBarrierSem, WorkPool, ScriptExecutor)
that are also imported from other modules. The definitions are included at the bottom of the file.
This suggests a consolidation of multiple files into one.
"""

from threading import Event, Thread, Lock
# Note: The following classes are imported but also defined later in this file.
from barrier import ReusableBarrierSem
from workPool import WorkPool

class Device(object):
    """
    @brief Represents a single device in the distributed network simulation.
    @details Each device manages its own sensor data and executes scripts using a pool of worker threads.
    It coordinates with other devices through a shared barrier and locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary representing the device's local sensor data.
        @param supervisor A supervisor object for fetching network information (e.g., neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # A pool of worker threads to execute scripts concurrently.
        self.workpool = WorkPool(4, self)

        self.scripts = []
        self.script_storage = []
        # Shared locks for data locations, assigned by the root device.
        self.locks = []
        # Shared barrier for inter-device synchronization, assigned by the root device.
        self.barrier = None
        self.neighbours = None

        self.script_lock = Lock()
        # Event to signal interaction with the supervisor (e.g., new script).
        self.supervisor_interact = Event()
        # Event to signal that all scripts for a timepoint have been assigned.
        self.timepoint_done = Event()

        # The main control thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (barrier and locks) for all devices.
        @details The device with the minimum ID is designated as the root device and is responsible
        for creating and distributing the shared barrier and locks to all other devices.
        """
        ids = []
        loc = []

        # Gather all device IDs and data locations to determine the required size for locks.
        for device in devices:
            ids.append(device.device_id)
            for location, _ in device.sensor_data.iteritems():
                loc.append(location)

        # Block Logic: The device with the minimum ID acts as the master for setup.
        max_locations = max(loc) + 1
        if self.device_id == min(ids):
            barrier = ReusableBarrierSem(len(ids))
            locks = [Lock() for _ in range(max_locations)]
            # Distribute the shared barrier and locks to all devices.
            for device in devices:
                device.assign_barrier(barrier)
                device.set_locks(locks)

    def assign_barrier(self, barrier):
        """@brief Assigns the shared barrier to this device."""
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device or signals the end of a timepoint.
        """
        # A None script indicates that all scripts for the current timepoint are assigned.
        if script is not None:
            self.script_lock.acquire()
            self.scripts.append((script, location))
            self.script_lock.release()
        else:
            self.timepoint_done.set()
        # Notify the device's main thread of an interaction.
        self.supervisor_interact.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data from a specific location.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def wait_on_scripts(self):
        """@brief Waits for all tasks in the work pool to be completed."""
        self.workpool.wait()

    def set_data(self, location, data):
        """@brief Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """@brief Shuts down the device by joining its main thread."""
        self.thread.join()

    def set_locks(self, locks):
        """@brief Assigns the shared locks to this device."""
        self.locks = locks

    def lock(self, location):
        """@brief Acquires the lock for a specific data location."""
        self.locks[location].acquire()

    def unlock(self, location):
        """@brief Releases the lock for a specific data location."""
        self.locks[location].release()

    def execute_scripts(self):
        """
        @brief Dispatches all currently queued scripts to the work pool.
        """
        self.script_lock.acquire()
        # Move scripts from the incoming queue to storage and add them to the work pool.
        for (script, location) in self.scripts:
            self.script_storage.append((script, location))
            self.workpool.add_data(script, location)
        # Clear the incoming script queue.
        del self.scripts[:]
        self.script_lock.release()

class DeviceThread(Thread):
    """
    @brief The main control thread for a Device instance.
    @details This thread manages the device's lifecycle, interacts with the supervisor,
    dispatches work, and handles synchronization at each timepoint.
    """

    def __init__(self, device):
        """
        @brief Initializes the device's main thread.
        @param device The parent device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """@brief The main execution loop for the device's control thread."""
        while True:
            # Fetch the list of neighbors for the current timepoint.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            # A None value for neighbors is the signal to shut down.
            if self.device.neighbours is None:
                self.device.workpool.shutdown()
                return

            # Execute any scripts that were assigned before the timepoint began.
            self.device.execute_scripts()
            
            # Block Logic: Main event loop for a single timepoint.
            # This loop waits for new scripts or for the signal that the timepoint is complete.
            while True:
                # Wait for an event from the supervisor (new script or timepoint end).
                self.device.supervisor_interact.wait()
                self.device.supervisor_interact.clear()

                # Pre-condition: A supervisor interaction has occurred.
                # Invariant: Any new scripts are moved to the work pool.
                self.device.script_lock.acquire()
                if len(self.device.scripts) > 0:
                    self.device.script_lock.release()
                    self.device.execute_scripts()
                else:
                    self.device.script_lock.release()

                # If the timepoint is marked as done, proceed to synchronization.
                if self.device.timepoint_done.is_set():
                    self.device.timepoint_done.clear()
                    
                    # Ensure any final scripts are executed.
                    if len(self.device.scripts) > 0:
                        self.device.execute_scripts()

                    # Wait for the work pool to finish all script executions for this timepoint.
                    self.device.wait_on_scripts()

                    # Synchronize with all other devices using the shared barrier.
                    self.device.barrier.wait()
                    # Restore processed scripts to the main list for the next timepoint/iteration.
                    self.device.scripts = self.device.script_storage
                    self.device.script_storage = []
                    break

# ======================================================================================
# The following class definitions seem to be included from other files.
# For clarity in a real-world project, these would typically reside in their own modules.
# ======================================================================================

from threading import Thread

class ScriptExecutor(Thread):
    """
    @brief A worker thread that executes scripts from a WorkPool.
    @details This thread fetches tasks from a shared queue, processes them, and waits for new tasks.
    """

    def __init__(self, index, workpool, device):
        """
        @param index A unique ID for the worker thread.
        @param workpool The parent WorkPool that manages this thread.
        @param device The device this worker belongs to, used to access data and neighbors.
        """
        Thread.__init__(self, name="Worker Thread %d" % index)
        self.index = index
        self.workpool = workpool
        self.device = device

    def run(self):
        """@brief The main loop for the worker thread."""
        while True:
            # Wait for data to be available in the queue.
            self.workpool.data.acquire()
            if self.workpool.done:
                return

            (script, location) = self.workpool.q.get()

            # Shutdown signal check.
            if self.device.neighbours is None:
                return

            # Acquire lock for the data location to ensure exclusive access.
            self.device.lock(location)

            # Block Logic: Gather data from neighbors and the local device.
            script_data = []
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Pre-condition: At least one data point was gathered.
            # Invariant: The script runs on the aggregated data, and the result is written back.
            if script_data != []:
                result = script.run(script_data)

                # Broadcast the result to all neighbors.
                for device in self.device.neighbours:
                    device.set_data(location, result)

                # Update the local device's data.
                self.device.set_data(location, result)
            
            # Release the lock and signal task completion.
            self.device.unlock(location)
            self.workpool.q.task_done()

# ======================================================================================

from threading import Semaphore
from Queue import Queue
from scriptexecutor import ScriptExecutor

class WorkPool(object):
    """
    @brief Manages a pool of worker threads to execute tasks concurrently.
    @details This class encapsulates a queue of tasks and a set of ScriptExecutor threads
    that process those tasks.
    """

    def __init__(self, num_threads, device):
        """
        @param num_threads The number of worker threads to create in the pool.
        @param device The parent device, passed to the worker threads.
        """
        self.device = device
        self.executors = []
        self.q = Queue()
        # Semaphore to signal the availability of tasks in the queue.
        self.data = Semaphore(0)
        self.done = False

        # Create and start the worker threads.
        for i in range(num_threads + 1):
            executor = ScriptExecutor(i, self, self.device)
            executor.start()
            self.executors.append(executor)

    def add_data(self, script, location):
        """
        @brief Adds a new task (script and location) to the work queue.
        """
        self.q.put((script, location))
        # Signal that a new item is available.
        self.data.release()

    def wait(self):
        """
        @brief Blocks until all items in the queue have been processed.
        """
        if not self.done:
            self.q.join()

    def shutdown(self):
        """
        @brief Shuts down the work pool, waiting for all tasks to complete and then joining the threads.
        """
        self.wait()
        self.done = True
        
        # Release the semaphore for each worker thread to unblock them so they can exit.
        for _ in self.executors:
            self.data.release()

        for executor in self.executors:
            executor.join()
