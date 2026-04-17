"""
@bc2c400a-5759-440d-9ec2-65636c3ec63f/device.py
@brief Simulation of a distributed sensor network node performing collaborative data processing.
This module defines the architectural framework for a device that manages local sensor state, 
synchronizes with neighbors via barriers and locks, and processes computational tasks (scripts) 
through an internal worker pool.

Domain: Distributed Systems, Parallel Computing, Sensor Networks.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
from workPool import WorkPool

class Device(object):
    """
    Represents a physical or virtual device in a sensor network.
    Functional Utility: Serves as a container for sensor data and manages the lifecycle 
    of scripts and synchronization primitives required for neighborhood-aware processing.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device node with its unique identity and local data.
        @param device_id: Unique integer identifier for the device.
        @param sensor_data: Dictionary mapping locations to sensor values.
        @param supervisor: Control entity providing neighborhood topology information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Functional Utility: Decouples task submission from execution using 4 worker threads.
        self.workpool = WorkPool(4, self)

        self.scripts = []
        self.script_storage = []
        self.locks = []
        self.barrier = None
        self.neighbours = None

        self.script_lock = Lock()
        self.supervisor_interact = Event()
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Orchestrates the global synchronization setup for a group of devices.
        Logic: Elects a leader based on the minimum device ID to initialize shared 
        concurrency primitives (barriers and locks).
        """

        ids = []
        loc = []

        # Block Logic: Aggregates metadata to determine the scope of synchronization.
        for device in devices:
            ids.append(device.device_id)
            for location, _ in device.sensor_data.iteritems():
                loc.append(location)

        # Invariant: Only one device (the leader) initializes and propagates shared resources.
        max_locations = max(loc) + 1
        if self.device_id == min(ids):
            barrier = ReusableBarrierSem(len(ids))
            locks = [Lock() for _ in range(max_locations)]
            for device in devices:
                device.assign_barrier(barrier)
                device.set_locks(locks)

    def assign_barrier(self, barrier):
        """Injects a shared barrier for temporal synchronization across devices."""
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        Receives a processing task from the supervisor.
        Logic: Queues the script or signals completion of the current timepoint.
        """

        if script is not None:
            self.script_lock.acquire()
            self.scripts.append((script, location))
            self.script_lock.release()
        else:
            self.timepoint_done.set()
        self.supervisor_interact.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location if managed by this device."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def wait_on_scripts(self):
        """Blocks until the internal workpool has processed all pending tasks."""
        self.workpool.wait()

    def set_data(self, location, data):
        """Updates the local sensor state for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully terminates the device's management thread."""
        self.thread.join()

    def set_locks(self, locks):
        """Assigns a set of mutual exclusion locks corresponding to sensor locations."""
        self.locks = locks

    def lock(self, location):
        """Acquires a mutex for a specific sensor location to ensure atomic updates."""
        self.locks[location].acquire()

    def unlock(self, location):
        """Releases the mutex for a specific sensor location."""
        self.locks[location].release()

    def execute_scripts(self):
        """
        Transfers pending scripts to the worker pool for parallel execution.
        Logic: Moves tasks from the local queue to the workpool and resets the local buffer.
        """
        self.script_lock.acquire()

        for (script, location) in self.scripts:
            self.script_storage.append((script, location))
            self.workpool.add_data(script, location)

        del self.scripts[:]
        self.script_lock.release()

class DeviceThread(Thread):
    """
    Control thread for a Device node.
    Functional Utility: Manages the state machine of the device, coordinating 
    between supervisor requests and internal task execution.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for the device thread.
        Algorithm: Iterative timepoint processing with barrier-based synchronization.
        """
        while True:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                self.device.workpool.shutdown()
                return

            self.device.execute_scripts()

            # Block Logic: Internal event loop handling supervisor interactions.
            while True:
                # Wait for interaction trigger from the supervisor.
                self.device.supervisor_interact.wait()
                self.device.supervisor_interact.clear()

                # Process any new scripts assigned by the supervisor.
                self.device.script_lock.acquire()
                if len(self.device.scripts) > 0:
                    self.device.script_lock.release()
                    self.device.execute_scripts()
                else:
                    self.device.script_lock.release()

                # Check if the current simulation step (timepoint) is complete.
                if self.device.timepoint_done.is_set():
                    self.device.timepoint_done.clear()
                    
                    # Final task flush before synchronization.
                    if len(self.device.scripts) > 0:
                        self.device.execute_scripts()

                    self.device.wait_on_scripts()

                    # Inline: Ensures all devices in the network reach the same temporal state.
                    self.device.barrier.wait()
                    # Resets script state for the next timepoint.
                    self.device.scripts = self.device.script_storage
                    self.device.script_storage = []
                    break

from threading import Thread


class ScriptExecutor(Thread):
    """
    Worker thread responsible for executing computational scripts.
    Functional Utility: Implements the 'read-modify-write' pattern over the neighborhood graph.
    """

    def __init__(self, index, workpool, device):
        Thread.__init__(self, name="Worker Thread %d" % index)
        self.index = index
        self.workpool = workpool
        self.device = device

    def run(self):
        """
        Worker execution loop.
        Logic: Pulls tasks from the workpool queue and applies them to the sensor network.
        """
        while True:
            # Block until data is available in the pool.
            self.workpool.data.acquire()
            if self.workpool.done:
                return

            (script, location) = self.workpool.q.get()

            if self.device.neighbours is None:
                return

            # Block Logic: Critical section for neighborhood data aggregation and update.
            # Invariant: Only one thread can process a specific sensor location across 
            # the entire device group at a time.
            self.device.lock(location)

            script_data = []
            # Aggregate data from neighbors.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Include own local data.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Functional Utility: Applies the domain-specific logic defined in the script.
                result = script.run(script_data)

                # Propagate results back to neighbors to maintain consensus or state flow.
                for device in self.device.neighbours:
                    device.set_data(location, result)

                # Update own local state.
                self.device.set_data(location, result)

            self.device.unlock(location)
            self.workpool.q.task_done()

from threading import Semaphore
from Queue import Queue
from scriptexecutor import ScriptExecutor

class WorkPool(object):
    """
    Management layer for a set of consumer threads.
    Functional Utility: Provides an asynchronous interface for task submission 
    and life-cycle management of workers.
    """

    def __init__(self, num_threads, device):
        self.device = device
        self.executors = []
        self.q = Queue()
        # Semaphore acts as a counter for available tasks in the queue.
        self.data = Semaphore(0)
        self.done = False

        # Spawns worker threads.
        for i in range(num_threads + 1):
            executor = ScriptExecutor(i, self, self.device)
            executor.start()
            self.executors.append(executor)

    def add_data(self, script, location):
        """Enqueues a new script task and signals available workers."""
        self.q.put((script, location))
        self.data.release()

    def wait(self):
        """Blocks until the queue is empty and all tasks have been acknowledged."""
        if not self.done:
            self.q.join()

    def shutdown(self):
        """Signals all workers to terminate and cleans up thread resources."""
        self.wait()
        self.done = True

        # Release all blocked workers.
        for _ in self.executors:
            self.data.release()

        # Join threads to ensure proper cleanup.
        for executor in self.executors:
            executor.join()
