"""
@file device.py
@brief Distributed sensor unit simulation with a managed worker pool and cyclic synchronization.
@details Implements a peer-to-peer network of devices that perform synchronized data 
aggregation. Uses a centralized WorkPool for parallel task execution and coordinates 
discrete execution cycles via a global cyclic barrier and location-specific locks.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
from workPool import WorkPool

class Device(object):
    """
    @brief Logic controller for an autonomous sensing entity in a distributed network.
    Functional Utility: Manages local data buffers, stages tasks for the worker pool, 
    and coordinates cluster-wide synchronization resources.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of local sensor location readings.
        @param supervisor entity providing topology discovery and cluster coordination.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Architecture: Dedicated worker pool for concurrent script execution.
        self.workpool = WorkPool(4, self)

        self.scripts = []
        self.script_storage = []
        self.locks = []
        self.barrier = None
        self.neighbours = None

        # Synchronization State.
        self.script_lock = Lock()
        self.supervisor_interact = Event()
        self.timepoint_done = Event()

        # Lifecycle: Spawns the main device management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collaborative initialization of shared synchronization resources.
        Logic: Designates the device with the minimal ID as the master allocator 
        for global barriers and location locks.
        """
        ids = []
        loc = []

        for device in devices:
            ids.append(device.device_id)
            for location, _ in device.sensor_data.iteritems():
                loc.append(location)

        # Allocation: Determines maximum sensor index to size the global lock array.
        max_locations = max(loc) + 1 if loc else 0
        if self.device_id == min(ids):
            barrier = ReusableBarrierSem(len(ids))
            locks = [Lock() for _ in range(max_locations)]
            for device in devices:
                device.assign_barrier(barrier)
                device.set_locks(locks)

    def assign_barrier(self, barrier):
        """
        @brief Internal: sets the reference to the shared cyclic barrier.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing script for the current temporal cycle.
        """
        if script is not None:
            with self.script_lock:
                self.scripts.append((script, location))
        else:
            # Protocol: Signals end of the assignment phase.
            self.timepoint_done.set()
        
        # Wake master thread to handle new assignments.
        self.supervisor_interact.set()

    def get_data(self, location):
        """
        @brief retrieval of local sensor readings.
        """
        return self.sensor_data.get(location)

    def wait_on_scripts(self):
        """
        @brief Blocks until the local worker pool has finalized all tasks.
        """
        self.workpool.wait()

    def set_data(self, location, data):
        """
        @brief Update of local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device orchestrator thread.
        """
        self.thread.join()

    def set_locks(self, locks):
        """
        @brief Internal: sets the reference to the shared lock registry.
        """
        self.locks = locks

    def lock(self, location):
        """
        @brief Cluster-wide mutual exclusion for a specific sensor location.
        """
        self.locks[location].acquire()

    def unlock(self, location):
        """
        @brief Releases the lock for a specific sensor location.
        """
        self.locks[location].release()

    def execute_scripts(self):
        """
        @brief Offloads staged scripts to the WorkPool for execution.
        Logic: Transfers tasks from staging area to the pool queue.
        """
        with self.script_lock:
            for (script, location) in self.scripts:
                # Cache: preserves scripts for potential re-execution.
                self.script_storage.append((script, location))
                self.workpool.add_data(script, location)
            del self.scripts[:]

class DeviceThread(Thread):
    """
    @brief Orchestrator thread managing execution cycles (discovery -> execution -> sync).
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main coordination loop for the device.
        """
        while True:
            # Discovery: Fetches current network neighbors.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                # Termination: shuts down the worker pool.
                self.device.workpool.shutdown()
                return

            # Task Injection: begins processing current staged workload.
            self.device.execute_scripts()

            while True:
                # Sync: Waits for task assignment signals from the supervisor.
                self.device.supervisor_interact.wait()
                self.device.supervisor_interact.clear()

                # Dispatch: Processes any newly assigned scripts immediately.
                with self.device.script_lock:
                    if self.device.scripts:
                        self.device.execute_scripts()

                /**
                 * Block Logic: Timepoint finalization.
                 * Condition: Assignment phase is complete.
                 */
                if self.device.timepoint_done.is_set():
                    self.device.timepoint_done.clear()
                    
                    # Final flush: ensures all scripts are in the pool.
                    if self.device.scripts:
                        self.device.execute_scripts()

                    # Wait: Synchronize with local worker pool completion.
                    self.device.wait_on_scripts()

                    # Global Barrier: ensures cluster-wide temporal alignment.
                    self.device.barrier.wait()
                    
                    # Cycle Reset: Restores cached scripts for the next timepoint.
                    self.device.scripts = self.device.script_storage
                    self.device.script_storage = []
                    break

from threading import Thread

class ScriptExecutor(Thread):
    """
    @brief Persistent worker thread that consumes tasks from the WorkPool.
    Architecture: Implements a distributed Map-Reduce operation.
    """

    def __init__(self, index, workpool, device):
        Thread.__init__(self, name="Executor-%d-%d" % (device.device_id, index))
        self.index = index
        self.workpool = workpool
        self.device = device

    def run(self):
        """
        @brief Worker loop: acquisition -> aggregation -> propagation.
        """
        while True:
            # Sync: waits for new tasks in the pool.
            self.workpool.data.acquire()
            if self.workpool.done:
                return

            # Acquisition: takes next task from the queue.
            (script, location) = self.workpool.q.get()

            if self.device.neighbours is None:
                return

            /**
             * Block Logic: Distributed state computation.
             * Critical Section: Uses global location locks to ensure atomicity.
             */
            self.device.lock(location)

            script_data = []
            # Map Phase: aggregates state from neighbors.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Local state inclusion.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Computation Phase.
                result = script.run(script_data)
                
                # Reduce/Update Phase: propagates state change.
                for device in self.device.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)

            self.device.unlock(location)
            # Notify pool of task finalization.
            self.workpool.q.task_done()

from threading import Semaphore
from Queue import Queue

class WorkPool(object):
    """
    @brief Thread-safe execution environment for data aggregation scripts.
    Functional Utility: Manages a fixed-size pool of ScriptExecutor threads.
    """

    def __init__(self, num_threads, device):
        self.device = device
        self.executors = []
        self.q = Queue()
        self.data = Semaphore(0)
        self.done = False

        # Scalability: Spawns worker pool.
        for i in range(num_threads + 1):
            executor = ScriptExecutor(i, self, self.device)
            executor.start()
            self.executors.append(executor)

    def add_data(self, script, location):
        """
        @brief Submits a new task to the execution pool.
        """
        self.q.put((script, location))
        self.data.release()

    def wait(self):
        """
        @brief Blocks until all tasks in the internal queue are complete.
        """
        if not self.done:
            self.q.join()

    def shutdown(self):
        """
        @brief Gracefully terminates all executor threads in the pool.
        """
        self.wait()
        self.done = True

        # Protocol: Releases all blocked executors.
        for _ in self.executors:
            self.data.release()

        for executor in self.executors:
            executor.join()
