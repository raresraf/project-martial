"""
@c8ef764d-3142-4cb8-867b-cc582c467232/device.py
@brief Concurrent sensor simulation using dynamic thread spawning with semaphore-controlled concurrency limits.
* Algorithm: Event-driven worker thread management with multi-phase semaphore barriers and per-location mutual exclusion.
* Functional Utility: Orchestrates simulation timepoints across a device cluster, managing neighbor data aggregation and synchronized state updates.
"""

from threading import Semaphore, Event, Lock, Thread

class ReusableBarrier(object):
    """
    @brief Implementation of a two-phase synchronization barrier using counting semaphores.
    * Algorithm: Dual-phase arrival/release pattern to ensure consistent thread alignment across repeated cycles.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the barrier state and its internal phase semaphores.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Synchronizes the calling thread through both stages of the barrier.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Stage 1: Collects all threads and releases them simultaneously.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Logic: Final thread arrival triggers the release of the entire group.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Stage 2: Secondary synchronization to prevent thread overruns in tight loops.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    """
    @brief Core device node that manages sensor readings and coordinate processing threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device state and bootstraps the main management thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.none_script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.timepoint_end = 0
        self.barrier = None
        self.lock_hash = None # Intent: Map of location-specific locks shared across all devices.

    def __str__(self):
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """
        @brief Links the device to the shared cluster synchronization point.
        """
        self.barrier = barrier

    def set_locks(self, lock_hash):
        """
        @brief Assigns the shared global lock set for location-level synchronization.
        """
        self.lock_hash = lock_hash

    def setup_devices(self, devices):
        """
        @brief Collective resource initialization for the simulation cluster.
        Invariant: The device with the minimum ID acts as the coordinator, creating and distributing the barrier and locks.
        """
        ids_list = []
        for dev in devices:
            ids_list.append(dev.device_id)

        if self.device_id == min(ids_list):
            self.barrier = ReusableBarrier(len(devices))
            self.lock_hash = {}

            # Logic: Aggregate all unique sensor locations across all devices to create specific locks.
            for dev in devices:
                for location in dev.sensor_data:
                    if location not in self.lock_hash:
                        self.lock_hash[location] = Lock()

            # Logic: Broadcast the shared primitives to all participating nodes.
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.set_barrier(self.barrier)
                    dev.set_locks(self.lock_hash)

    def assign_script(self, script, location):
        """
        @brief Buffers an incoming task or signals completion of the current timepoint batch.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.none_script_received.set()

    def get_data(self, location):
        """
        @brief Standard data retrieval interface for local sensor readings.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Standard data update interface for local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Terminates the primary device management thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Management thread coordinating the lifecycle of a device node and its transient workers.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator with a concurrency-limiting semaphore (8 workers).
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.semaphore = Semaphore(value=8)

    def run(self):
        """
        @brief Main execution lifecycle for the coordinator thread.
        Algorithm: Dynamic thread spawning for each script, constrained by a global worker semaphore.
        """
        while True:
            # Logic: Fetch current neighbor configuration.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Ensures all script assignments for the timepoint are received.
            self.device.none_script_received.wait()
            self.device.none_script_received.clear()

            thread_list = []

            # Dispatch Phase: Spawns an individual thread for each assigned script task.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, neighbours, script, location,
                    self.semaphore)
                thread.start()
                thread_list.append(thread)

            # Logic: Joins all local workers before proceeding to the global synchronization barrier.
            for i in xrange(len(thread_list)):
                thread_list[i].join()

            # Synchronization Phase: Align all devices across the cluster before the next timepoint.
            self.device.barrier.wait()

class MyThread(Thread):
    """
    @brief worker thread implementing the execution of a single sensor script.
    """

    def __init__(self, device, neighbours, script, location, semaphore):
        """
        @brief Initializes worker with task parameters and the concurrency control semaphore.
        """
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.semaphore = semaphore

    def run(self):
        """
        @brief Main execution for a single script unit.
        Algorithm: Resource-constrained execution with location-based mutual exclusion.
        """
        # Logic: Respects the device-level concurrency limit.
        self.semaphore.acquire()

        # Logic: Acquires global location lock to ensure atomic distributed state update.
        self.device.lock_hash[self.location].acquire()

        script_data = []

        # Distributed Data Aggregation: Accumulates readings from neighbors and local node.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Functional Utility: Executes processing logic and broadcasts results to the network.
        if script_data != []:
            result = self.script.run(script_data)

            for device in self.neighbours:
                device.set_data(self.location, result)

            self.device.set_data(self.location, result)

        # Post-condition: Cleanup and release acquired synchronization primitives.
        self.device.lock_hash[self.location].release()
        self.semaphore.release()
