"""
@file device.py
@brief Distributed sensor unit simulation with load-balanced parallel processing.
@details Implements a peer-to-peer network of sensing units that perform synchronized 
data aggregation. Uses a static partitioning strategy to distribute scripts across 
multiple worker threads and coordinates temporal alignment via a cyclic barrier.
"""

from threading import Event, Thread, Lock, Condition

class Device(object):
    """
    @brief Logic controller for an autonomous sensing entity.
    Functional Utility: Manages local data buffers, organizes task execution, and 
    participates in cluster-wide synchronization via shared barriers and locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of local sensor location readings.
        @param supervisor entity providing topology discovery services.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        # Lifecycle: Spawns the main management thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.data_locks = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collaborative initialization of shared synchronization resources.
        Logic: Designates device 0 as the allocator for the cluster barrier and 
        the global lock registry.
        """
        if self.device_id == 0:
            new_bar = Barrier(len(devices))

            # Discovery: Identifies the maximum location index to size the lock array.
            locations = []
            for device in devices:
                for location in device.sensor_data.keys():
                    locations.append(location)

            max_loc = max(locations) if locations else 0
            # Allocation: Creates a dedicated lock for every unique sensor location.
            data_locks = [Lock() for _ in range(max_loc + 1)]
            
            # Propagation: Distributes shared primitives to all peers.
            for device in devices:
                device.set_barrier_locks(new_bar, data_locks)

    def set_barrier_locks(self, barrier, data_locks):
        """
        @brief Sets references to shared synchronization resources.
        """
        self.barrier = barrier
        self.data_locks = data_locks

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing script for the current unit of time.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Protocol: Signals end of the script assignment phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieval of local sensor readings.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        @brief Updates local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device lifecycle thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Orchestrator thread managing discrete units of execution (timepoints).
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main loop: discovery -> partitioning -> parallel execution -> sync.
        """
        while True:
            # Discovery: Fetches topological neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Sync: Wait for the assignment phase to complete.
            self.device.timepoint_done.wait()

            /**
             * Block Logic: Load-balanced script partitioning.
             * Logic: Distributes scripts into 8 buckets using a round-robin strategy 
             * to balance work across worker threads.
             */
            buckets = [[] for _ in range(8)]
            for count, task in enumerate(self.device.scripts):
                buckets[count % 8].append(task)

            # Execution: Spawns one thread per non-empty bucket.
            threads = []
            for bucket in buckets:
                if bucket:
                    thread = ScriptThread(self.device, bucket, neighbours)
                    thread.start()
                    threads.append(thread)

            # Sync: Wait for all local worker threads to finalize.
            for thread in threads:
                thread.join()

            # Global Sync: Ensure all cluster members align at the barrier.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()


class ScriptThread(Thread):
    """
    @brief Worker thread that executes a batch of assigned scripts sequentially.
    Architecture: Implements distributed Map-Reduce for each script in its partition.
    """

    def __init__(self, device, scripts, neighbours):
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):
        """
        @brief Execution logic for script batch.
        Critical Section: Uses location-specific shared locks to ensure atomicity.
        """
        for (script, location) in self.scripts:
            # Lock: Acquires exclusive access to this location across the cluster.
            self.device.data_locks[location].acquire()

            script_data = []
            # Map Phase: Aggregates state from neighbors and self.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Computation: Executes aggregation logic.
                result = script.run(script_data)
                
                # Reduce Phase: Propagates result to all participants.
                for device in self.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)

            self.device.data_locks[location].release()


class Barrier():
    """
    @brief implementation of a reusable cyclic barrier using Condition variables.
    Functional Utility: Synchronizes a fixed group of threads across recurring cycles.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        @brief blocks the calling thread until all expected threads arrive.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Release all waiting threads and reset for next cycle.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()
