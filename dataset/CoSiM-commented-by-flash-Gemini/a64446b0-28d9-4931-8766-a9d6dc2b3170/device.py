"""
@file device.py
@brief Distributed sensor unit simulation with a master-worker concurrency model.
@details Orchestrates a network of devices that perform synchronized data processing. 
Uses a dedicated master thread for coordination and a pool of workers for script execution.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue, Empty

class ReusableBarrier(object):
    """
    @brief Implementation of a cyclic barrier using Condition variables.
    Functional Utility: Provides recurring synchronization points for a group of threads.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        @brief Blocks the calling thread until the barrier is released by the last arrival.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Turnstile: Releasing all waiting threads and resetting state.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    @brief Controller for a sensing entity in a distributed network.
    Functional Utility: Manages local data buffers, task queues, and inter-device 
    synchronization resources (barriers and locks).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique integer identifier.
        @param sensor_data Initial set of local sensor readings.
        @param supervisor Coordination entity for topology discovery.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.neighbours = []

        # Sync Resources: Shared across the cluster during setup.
        self.barrier = None
        self.locks = []
        self.timepoint_done = Event()
        self.tasks_ready = Event()
        self.tasks = Queue()
        self.simulation_ended = False

        # Architecture: Master thread for lifecycle management.
        self.master = DeviceThreadMaster(self)
        self.master.start()

        # Architecture: Fixed-size worker pool for task execution.
        self.workers = []
        for i in xrange(8):
            worker = DeviceThreadWorker(self, i)
            self.workers.append(worker)
            worker.start()

    def __str__(self):
        return "Device [%d]" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization of cluster-wide synchronization primitives.
        Logic: Designates device 0 as the allocator for barriers and sensor locks.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            # Allocation: Creates a dedicated lock for each of the 24 possible sensor locations.
            locks = [Lock() for _ in xrange(24)]
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """
        @brief Schedules a processing script for the current temporal window.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Protocol: Signals end of task assignment phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Thread-safe retrieval of sensor data (assumes external lock acquisition).
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Update of sensor data (assumes external lock acquisition).
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Terminates the device lifecycle threads.
        """
        self.master.join()
        for worker in self.workers:
            worker.join()


class DeviceThreadMaster(Thread):
    """
    @brief High-level orchestrator thread for a single device unit.
    Functional Utility: Manages timepoint progression and task distribution.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device [%d] Thread Master" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main orchestration loop: discovery -> queue population -> synchronization.
        """
        while True:
            # Topology Discovery: Fetches current neighbors.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                # Termination: Signals shutdown to worker pool.
                self.device.simulation_ended = True
                self.device.tasks_ready.set()
                break

            # Block until assignment is complete.
            self.device.timepoint_done.wait()

            # Task Distribution: populates the internal queue for worker consumption.
            for task in self.device.scripts:
                self.device.tasks.put(task)

            self.device.tasks_ready.set()

            # Sync: Ensures all workers have finalized their assigned tasks.
            self.device.tasks.join()

            # Reset state for the next timepoint.
            self.device.tasks_ready.clear()
            self.device.timepoint_done.clear()

            # Global Sync: Aligns all devices before proceeding.
            self.device.barrier.wait()


class DeviceThreadWorker(Thread):
    """
    @brief Low-level worker thread that executes data aggregation tasks.
    """

    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device [%d] Worker [%d]" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        @brief Continuous consumer loop for the device's task queue.
        """
        while not self.device.simulation_ended:
            # Wait for tasks to be available in the current cycle.
            self.device.tasks_ready.wait()

            try:
                # Non-blocking acquisition: Workers compete for tasks in the local queue.
                script, location = self.device.tasks.get(block=False)

                /**
                 * Block Logic: Critical section for distributed state update.
                 * Invariant: Exclusive access to the location's data across the cluster.
                 */
                self.device.locks[location].acquire()

                script_data = []
                # Map Phase: Aggregates data from neighborhood.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Inclusion: local state.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if len(script_data) > 0:
                    # Computation: Execution of aggregation logic.
                    result = script.run(script_data)

                    # Reduce/Update: Propagates the state change.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                self.device.locks[location].release()
                
                # Signal task completion to the Master.
                self.device.tasks.task_done()
            except Empty:
                # Logic: No more tasks in queue; loop back to wait for next timepoint.
                pass
