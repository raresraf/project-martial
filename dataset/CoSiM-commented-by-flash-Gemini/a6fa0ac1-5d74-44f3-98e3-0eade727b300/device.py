"""
@file device.py
@brief Distributed sensor unit simulation using a dynamic task-worker architecture.
@details Implements a peer-to-peer network of devices where entities execute aggregation 
scripts via a shared work pool. Utilizes events and barriers for multi-phase synchronization.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrier

class Device(object):
    """
    @brief Controller for a sensing unit in a distributed cluster.
    Functional Utility: Manages local data buffers, location-specific locks, and organizes 
    the lifecycle of the primary management thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of sensor readings.
        @param supervisor Coordinator for neighborhood discovery.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        # Lock Registry: One lock per local sensor location to ensure atomic access.
        self.locks = {}

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization of cluster synchronization resources.
        Logic: Designates device 0 as the barrier allocator and lifecycle initiator.
        """
        for key in self.sensor_data:
            self.locks[key] = Lock()

        if self.device_id == 0:
            # Shared Barrier: All devices synchronize on this instance.
            self.thread.barrier = ReusableBarrier(len(devices))

            # Propagation: Distributes the shared barrier to all cluster members.
            for device in devices:
                device.thread.barrier = self.thread.barrier

            # Execution: Starts the management threads for all devices.
            for device in devices:
                device.thread.start()

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing script for the current temporal unit.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Termination: Signals end of the script assignment phase.
            self.timepoint_done.set()

    def has_data(self, location):
        """
        @brief Checks for the presence of a specific sensor location.
        """
        return location in self.sensor_data

    def get_data(self, location):
        """
        @brief Thread-safe retrieval of sensor data.
        Functional Utility: Acquires the location-specific lock before returning the value.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Thread-safe update of sensor data.
        Functional Utility: Releases the location-specific lock after committing the update.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """
        @brief Gracefully terminates the device lifecycle.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief High-level orchestrator thread for a single device unit.
    Architecture: Manages a pool of worker threads and a task queue.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = None
        # Synchronization: Primitives for work pool management.
        self.work_pool_lock = Lock()
        self.work_pool_empty = Event()
        self.work_ready = Event()
        self.work_pool = []
        self.simulation_complete = False
        # State Initialization.
        self.work_ready.clear()
        self.work_pool_empty.set()

    def run(self):
        """
        @brief Main coordination loop: discovery -> worker scaling -> task injection -> sync.
        """
        workers = []

        while True:
            # Discovery: Fetches current network topology.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block until assignment phase completes.
            self.device.timepoint_done.wait()

            /**
             * Block Logic: Dynamic worker scaling.
             * Logic: Ensures up to 8 workers are available to process the assigned scripts.
             */
            for i in range(len(workers), len(self.device.scripts)):
                if len(workers) < 8:
                    worker = Worker(self.work_pool_empty, self.work_ready, self.work_pool_lock, self)
                    workers.append(worker)
                    worker.start()

            # Global Sync: Ensure all devices have their worker pools ready.
            self.barrier.wait()

            /**
             * Block Logic: Task preparation and injection.
             * Logic: Groups self and neighbors by shared sensor locations to form computation Tasks.
             */
            for (script, location) in self.device.scripts:
                script_devices = []
                for device in neighbours:
                    if device.has_data(location):
                        script_devices.append(device)

                if script_devices:
                    if self.device not in script_devices:
                        script_devices.append(self.device)
                    # Optimization: Consistent ordering of devices to avoid deadlocks.
                    script_devices.sort(key=lambda x: x.device_id, reverse=False)
                    
                    # Work Submission: Injects the task into the pool.
                    self.work_pool_lock.acquire()
                    self.work_pool.append(Task(script, location, script_devices))
                    self.work_ready.set()
                    self.work_pool_empty.clear()
                    self.work_pool_lock.release()

            # Finalization: Blocks until all local work is acknowledged.
            self.work_pool_empty.wait()
            for worker in workers:
                worker.work_done.wait()
            self.device.timepoint_done.clear()

        # Shutdown Protocol: Signals completion to all workers.
        self.work_pool_lock.acquire()
        self.simulation_complete = True
        self.work_ready.set()
        self.work_pool_lock.release()

        for worker in workers:
            worker.join()


class Worker(Thread):
    """
    @brief Low-level execution thread that processes individual tasks.
    Functional Utility: Implements distributed state updates using task-defined device lists.
    """

    def __init__(self, work_pool_empty, work_ready, work_pool_lock, device_thread):
        Thread.__init__(self, name="Worker Thread")
        self.work_pool_lock = work_pool_lock
        self.work_pool_empty = work_pool_empty
        self.work_ready = work_ready
        self.device_thread = device_thread
        self.work_done = Event()
        self.work_done.set()

    def run(self):
        """
        @brief Continuous consumer loop for the device's work pool.
        """
        while True:
            # Wait for available tasks or shutdown signal.
            self.work_ready.wait()
            if self.device_thread.simulation_complete:
                break

            self.work_pool_lock.acquire()
            
            if not self.device_thread.work_pool:
                # Logic: Signal that the pool is exhausted for the current cycle.
                self.work_pool_empty.set()
                if not self.device_thread.simulation_complete:
                    self.work_ready.clear()
                self.work_pool_lock.release()
            else:
                # Task Acquisition.
                self.work_done.clear()
                task = self.device_thread.work_pool.pop(0)
                self.work_pool_lock.release()
                
                /**
                 * Block Logic: Distributed state computation.
                 * Protocol: Collects data (locks), computes result, and propagates (unlocks).
                 */
                data = []
                for device in task.devices:
                    data.append(device.get_data(task.location))

                # Computation: Executes the aggregation logic.
                result = task.script.run(data)

                # Reduce Phase: Distributes the updated state to all participating devices.
                for device in task.devices:
                    device.set_data(task.location, result)

                self.work_done.set()


class Task(object):
    """
    @brief Atomic unit of work representing a script execution on a set of distributed sensors.
    """
    def __init__(self, script, location, devices):
        self.devices = devices
        self.script = script
        self.location = location
