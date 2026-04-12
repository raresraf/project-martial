"""
@file device.py
@brief Distributed sensor processing system with fine-grained internal thread synchronization.
@details Implements a peer-to-peer network of sensing units that execute data-aggregation 
scripts. Leverages multiple internal barriers to coordinate local worker threads and 
global barriers for cluster-wide alignment across timepoints.
"""

from threading import Event, Thread, Lock
from Queue import Queue, Empty
from barrier import Barrier

class Device(object):
    """
    @brief Logic controller for an autonomous sensor unit in a distributed cluster.
    Functional Utility: Manages local data state, coordinates an internal pool of 
    worker threads, and shares synchronization primitives with peer devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of local sensor location readings.
        @param supervisor Entity providing topological neighbor discovery services.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Thread Pool: Fixed capacity of 8 workers per device.
        self.num_threads = 8
        self.scripts = []
        
        # Task Distribution: Synchronized queue for worker thread consumption.
        self.jobs_queue = Queue()
        self.neighbours = []

        # Internal Synchronization: Barriers to align local worker threads across lifecycle phases.
        self.scripts_received = Event()
        self.scripts_received_barrier = Barrier(self.num_threads)
        self.scripts_processed_barrier = Barrier(self.num_threads)
        self.neighbours_received_barrier = Barrier(self.num_threads)

        # Global Resources: Shared across the cluster during setup.
        self.location_locks = {}
        self.timepoint_barrier = None

        # Lifecycle: Spawns the worker thread pool.
        self.threads = [DeviceThread(self, i) for i in xrange(self.num_threads)]
        for i in xrange(self.num_threads):
            self.threads[i].start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collaborative initialization of shared synchronization resources.
        Logic: Designates the device with the lowest ID as the cluster-wide resource allocator.
        """
        leader_id = min([device.device_id for device in devices])

        if self.device_id == leader_id:
            # Discovery: Identifies all unique sensor locations across the peer network.
            locations_set = set()
            for device in devices:
                locations_set.update(device.sensor_data.keys())
            
            # Allocation: Creates a dedicated Lock for every unique location.
            self.location_locks = {loc: Lock() for loc in locations_set}
            
            # Allocation: Creates the global barrier for all devices in the cluster.
            self.timepoint_barrier = Barrier(len(devices))

            # Propagation: Shares allocated resources with all peer devices.
            for device in devices:
                device.location_locks = self.location_locks
                device.timepoint_barrier = self.timepoint_barrier

    def assign_script(self, script, location):
        """
        @brief Schedules a processing task for the current unit of time.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.jobs_queue.put((script, location))
        else:
            # Protocol: Signals end of the script submission phase.
            self.scripts_received.set()

    def get_data(self, location):
        """
        @brief retrieval of local sensor readings.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        @brief update of local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the worker thread pool.
        """
        for i in xrange(self.num_threads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    @brief Worker thread responsible for executing data aggregation scripts.
    Architecture: Implements a distributed Map-Reduce operation.
    """

    def __init__(self, device, id_thread):
        Thread.__init__(self, name="Device-%d-Worker-%d" % (device.device_id, id_thread))
        self.device = device
        self.id_thread = id_thread

    def run(self):
        """
        @brief main loop coordinating timepoint cycles and task execution.
        """
        worker_leader = 0
        while True:
            # Phase 1: Discovery.
            if self.id_thread == worker_leader:
                # Designated worker leader fetches current network topology.
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            # Local Sync: Ensures all workers see the updated neighbors.
            self.device.neighbours_received_barrier.wait()

            if self.device.neighbours is None:
                # Termination: Signals shutdown to the thread pool.
                break

            # Phase 2: Task Acquisition.
            self.device.scripts_received.wait()
            self.device.scripts_received_barrier.wait()
            
            # State Management: Worker leader clears the event for the next cycle.
            if self.id_thread == worker_leader:
                self.device.scripts_received.clear()

            /**
             * Block Logic: Parallel task execution.
             * Logic: Workers pull tasks from the shared device queue until exhausted.
             */
            while True:
                try:
                    (script, location) = self.device.jobs_queue.get_nowait()
                    self.run_script(script, location)
                except Empty:
                    break
            
            # Local Sync: Ensures all assigned work for the device is complete.
            self.device.scripts_processed_barrier.wait()

            # Phase 3: Cluster-wide Synchronization.
            if self.id_thread == worker_leader:
                # Reset Logic: Re-enqueues assigned scripts for the next temporal unit.
                for script_task in self.device.scripts:
                    self.device.jobs_queue.put(script_task)
                
                # Global Sync: Designated worker leader participates in the cluster barrier.
                self.device.timepoint_barrier.wait()

    def run_script(self, script, location):
        """
        @brief execution logic for a single aggregation task.
        Critical Section: Uses shared location locks to ensure atomicity across the cluster.
        """
        with self.device.location_locks[location]:
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

            if script_data:
                # Computation: execution of the aggregation logic.
                result = script.run(script_data)
                
                # Reduce/Update: Propagates results back to participants.
                for device in self.device.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
