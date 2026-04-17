"""
@bc2c400a-5759-440d-9ec2-65636c3ec63f/device.py
@brief Distributed sensor simulation framework for concurrent script execution across a network of devices.
* Algorithm: Multi-threaded producer-consumer model with barrier synchronization and shared memory locking.
* Functional Utility: Orchestrates sensor data processing by distributing scripts to a thread pool and synchronizing state transitions.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
from workPool import WorkPool

class Device(object):
    """
    @brief Represents a physical or virtual sensor node in a distributed system.
    Orchestrates local sensor data, neighbor interactions, and script execution via a dedicated thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device with its unique identity, initial sensor state, and supervisor reference.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
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
        @brief Global synchronization setup to establish shared barriers and locks across all participating devices.
        Pre-condition: All device instances must be instantiated and passed as a collection.
        Invariant: Only the device with the minimum ID initializes the shared synchronization primitives to ensure single-instance initialization.
        """
        ids = []
        loc = []

        # Logic: Collects metadata from all devices to determine the scope of synchronization.
        for device in devices:
            ids.append(device.device_id)
            for location, _ in device.sensor_data.iteritems():
                loc.append(location)

        # Logic: Allocates a global lock set and barrier if this instance is the designated coordinator (min ID).
        max_locations = max(loc) + 1
        if self.device_id == min(ids):
            barrier = ReusableBarrierSem(len(ids))
            locks = [Lock() for _ in range(max_locations)]
            for device in devices:
                device.assign_barrier(barrier)
                device.set_locks(locks)

    def assign_barrier(self, barrier):
        """
        @brief Assigns a shared barrier for temporal synchronization between timepoints.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Queues a script for execution at a specific sensor location.
        Functional Utility: Acts as the primary interface for the supervisor to inject work into the device.
        """
        if script is not None:
            self.script_lock.acquire()
            self.scripts.append((script, location))
            self.script_lock.release()
        else:
            # Logic: Signals completion of current timepoint if no more scripts are provided.
            self.timepoint_done.set()
        self.supervisor_interact.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def wait_on_scripts(self):
        """
        @brief Blocks until all currently queued scripts in the workpool have finished processing.
        """
        self.workpool.wait()

    def set_data(self, location, data):
        """
        @brief Updates the sensor data for a specific location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device's execution thread.
        """
        self.thread.join()

    def set_locks(self, locks):
        """
        @brief Assigns the shared lock set used for mutual exclusion at sensor locations.
        """
        self.locks = locks

    def lock(self, location):
        """
        @brief Acquires a lock for a specific location to prevent race conditions during script execution.
        """
        self.locks[location].acquire()

    def unlock(self, location):
        """
        @brief Releases the lock for a specific location.
        """
        self.locks[location].release()

    def execute_scripts(self):
        """
        @brief Transfers queued scripts to the workpool for parallel execution.
        Invariant: Uses script_lock to ensure atomic transfer and clearing of the local script queue.
        """
        self.script_lock.acquire()

        for (script, location) in self.scripts:
            self.script_storage.append((script, location))
            self.workpool.add_data(script, location)

        del self.scripts[:]
        self.script_lock.release()

class DeviceThread(Thread):
    """
    @brief Background thread managing the lifecycle of a device's script execution loop.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core execution loop for the device.
        Functional Utility: Coordinates between neighbor discovery, script execution, and temporal synchronization.
        """
        while True:
            # Logic: Periodically updates neighbor list from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                # Logic: Shutdown signal received.
                self.device.workpool.shutdown()
                return

            self.device.execute_scripts()

            while True:
                # Block Logic: Waits for interactions or notifications from the supervisor.
                self.device.supervisor_interact.wait()
                self.device.supervisor_interact.clear()

                # Logic: Atomic check for new scripts arriving during the execution phase.
                self.device.script_lock.acquire()
                if len(self.device.scripts) > 0:
                    self.device.script_lock.release()
                    self.device.execute_scripts()
                else:
                    self.device.script_lock.release()

                # Block Logic: Handles the transition between simulation timepoints.
                if self.device.timepoint_done.is_set():
                    self.device.timepoint_done.clear()
                    
                    if len(self.device.scripts) > 0:
                        self.device.execute_scripts()

                    self.device.wait_on_scripts()

                    # Invariant: Synchronization barrier ensures all devices complete the current timepoint before advancing.
                    self.device.barrier.wait()
                    self.device.scripts = self.device.script_storage
                    self.device.script_storage = []
                    break

from threading import Thread

class ScriptExecutor(Thread):
    """
    @brief Worker thread within the WorkPool responsible for the actual execution of sensor scripts.
    """

    def __init__(self, index, workpool, device):
        Thread.__init__(self, name="Worker Thread %d" % index)
        self.index = index
        self.workpool = workpool
        self.device = device

    def run(self):
        """
        @brief Main loop for the script executor worker.
        Functional Utility: Consumes tasks from the workpool queue and performs distributed data processing.
        """
        while True:
            # Logic: Blocks on semaphore until new data is available in the queue.
            self.workpool.data.acquire()
            if self.workpool.done:
                return

            (script, location) = self.workpool.q.get()

            if self.device.neighbours is None:
                return

            # Logic: Synchronizes access to shared sensor data using per-location locks.
            self.device.lock(location)

            # Logic: Aggregates sensor data from self and discovered neighbors for the target location.
            script_data = []
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script if data is available and propagates results across the cluster.
            if script_data != []:
                result = script.run(script_data)

                # Logic: Updates remote neighbor states with the computed result.
                for device in self.device.neighbours:
                    device.set_data(location, result)

                # Logic: Updates local state with the computed result.
                self.device.set_data(location, result)

            self.device.unlock(location)
            self.workpool.q.task_done()

from threading import Semaphore
from Queue import Queue
from scriptexecutor import ScriptExecutor

class WorkPool(object):
    """
    @brief Thread pool manager for parallelizing script execution tasks.
    """

    def __init__(self, num_threads, device):
        self.device = device
        self.executors = []
        self.q = Queue()
        self.data = Semaphore(0)
        self.done = False

        # Logic: Spawns worker threads to handle asynchronous task processing.
        for i in range(num_threads + 1):
            executor = ScriptExecutor(i, self, self.device)
            executor.start()
            self.executors.append(executor)

    def add_data(self, script, location):
        """
        @brief Enqueues a new script task and signals available workers.
        """
        self.q.put((script, location))
        self.data.release()

    def wait(self):
        """
        @brief Blocks until all enqueued tasks are completed.
        """
        if not self.done:
            self.q.join()

    def shutdown(self):
        """
        @brief Terminates all worker threads and cleans up resources.
        """
        self.wait()
        self.done = True

        # Logic: Notifies all executors of the shutdown state.
        for _ in self.executors:
            self.data.release()

        for executor in self.executors:
            executor.join()
