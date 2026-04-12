"""
@file device.py
@brief Distributed sensor unit simulation with a hierarchical master-worker architecture.
@details Implements a peer-to-peer network of devices that perform synchronized data processing. 
Uses a master thread for cycle coordination and a worker pool for concurrent script execution.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
from Queue import Queue

class Device(object):
    """
    @brief Logic controller for a single sensing unit in a distributed cluster.
    Functional Utility: Manages local sensor state, task buffering, and coordinates 
    multi-device synchronization via shared barriers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Dictionary mapping locations to initial readings.
        @param supervisor Coordinator for topology discovery.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.timepoint_done = Event()

        # Task Management: Queue for distributing aggregation work to workers.
        self.buffer = Queue()
        # Staging: Collects new scripts before injection into the master loop.
        self.fresh = []
        # Registry: Caches scripts by location for repeated execution across timepoints.
        self.scripts_by_location = {}

        # Architecture: Primary management thread.
        self.master = DeviceMaster(self)
        self.master.start()

        # Architecture: Fixed-size worker pool (capacity 8).
        self.workers = [DeviceWorker(self) for _ in xrange(8)]
        for worker in self.workers:
            worker.start()

        # Sync: Per-location locks to ensure atomic read-modify-write on sensor state.
        self.local_lock = {loc: Lock() for loc in self.sensor_data.keys()}
        self.barrier = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization of the shared cyclic barrier.
        Logic: Designates device 0 as the allocator.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            for i in xrange(1, len(devices)):
                devices[i].barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Stages a processing script for the current cycle.
        """
        if script is not None:
            self.fresh.append((script, location))
        else:
            # Protocol: Signals end of the script submission phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Thread-safe retrieval of sensor state.
        Functional Utility: Acquires the local lock before returning the current reading.
        """
        data = None
        if location in self.sensor_data:
            self.local_lock[location].acquire()
            data = self.sensor_data[location]
        return data

    def set_data(self, location, data):
        """
        @brief Thread-safe update of sensor state.
        Functional Utility: Commits the update and releases the local location lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.local_lock[location].release()

    def shutdown(self):
        """
        @brief Gracefully terminates the device and its worker threads.
        """
        self.master.join()
        for worker in self.workers:
            worker.join()


class DeviceWorker(Thread):
    """
    @brief Low-level execution unit that processes data aggregation tasks.
    Architecture: Implements a distributed Map-Reduce operation.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Worker-%d" % device.device_id)
        self.device = device
        self.master = self.device.master

    def _run_one_script(self, script, location):
        """
        @brief core aggregation logic for a single script instance.
        Workflow: Aggregation -> Computation -> Propagation.
        """
        script_data = []

        # Map Phase: Collect readings from all topological neighbors.
        for device in self.master.neighbours:
            if device != self.device:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Inclusion: Adds self data to the processing set.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Computation: Executes the user-defined logic.
            result = script.run(script_data)

            # Reduce Phase: Updates the distributed state across all participants.
            for device in self.master.neighbours:
                if device != self.device:
                    device.set_data(location, result)
            self.device.set_data(location, result)

    def _run_all_by_location(self, location):
        """
        @brief Batch execution of all cached scripts for a specific sensor.
        """
        for script in self.master.scripts_by_location[location]:
            self._run_one_script(script, location)

    def run(self):
        """
        @brief Continuous consumer loop for the device's task buffer.
        """
        while True:
            # Task Acquisition.
            (script, location) = self.master.buffer.get()

            # Protocol: Poison pill check for termination.
            if location is None:
                self.master.buffer.task_done()
                break

            if script is None:
                # Batch Task: Re-executes cached logic.
                self._run_all_by_location(location)
            else:
                # Atomic Task: Executes a newly assigned script.
                self._run_one_script(script, location)

            # Signal task completion to the Master.
            self.master.buffer.task_done()


class DeviceMaster(Thread):
    """
    @brief Orchestration thread responsible for cycle management and task injection.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Master-%d" % device.device_id)
        self.device = device
        self.neighbours = None
        self.buffer = self.device.buffer
        self.fresh = self.device.fresh
        self.scripts_by_location = self.device.scripts_by_location

    def run(self):
        """
        @brief Main cycle loop: discovery -> injection -> waiting -> sync.
        """
        while True:
            # Discovery: Fetches neighbors from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                # Termination: Signals shutdown to all workers.
                for _ in self.device.workers:
                    self.buffer.put((None, None))
                break

            # Task Injection: Queues re-execution of previously assigned scripts.
            for loc in self.device.scripts_by_location.keys():
                self.buffer.put((None, loc))

            # Block until assignment for the current timepoint is finished.
            self.device.timepoint_done.wait()

            # Dynamic Scheduling: Registers and enqueues newly assigned scripts.
            while self.fresh:
                elem = (script, location) = self.fresh.pop(0)
                if location not in self.scripts_by_location:
                    self.scripts_by_location[location] = [script]
                else:
                    self.scripts_by_location[location].append(script)
                self.buffer.put(elem)

            # Synchronization: Wait for all workers to finalize their local tasks.
            self.buffer.join()
            
            # Global Sync: Ensure all cluster units are aligned at the barrier.
            self.device.barrier.wait()
            
            # Reset state for next unit of time.
            self.device.timepoint_done.clear()
