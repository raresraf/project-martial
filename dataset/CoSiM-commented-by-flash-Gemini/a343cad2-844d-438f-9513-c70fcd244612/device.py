"""
@file device.py
@brief Distributed sensor data processing system using a worker-pool architecture.
@details Orchestrates a network of virtual devices that execute data-aggregation scripts 
concurrently. Utilizes synchronized queues for task distribution and barriers for temporal alignment.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

from barrier import ReusableBarrierCond

class Device(object):
    """
    @brief Controller for a network-connected sensor unit.
    Functional Utility: Manages local data, task scheduling via a work queue, and 
    cluster-wide synchronization resources.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier for the device.
        @param sensor_data Dictionary of local sensor readings.
        @param supervisor Reference to the central network coordinator.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Task Management: Queue for distributing work to local worker threads.
        self.scripts = [] 
        self.work_queue = Queue() 
        self.neighbours = [] 

        # Synchronization: Events and primitives for coordinating processing phases.
        self.timepoint_done = Event() 
        self.setup_ready = Event() 
        self.neighbours_set = Event() 
        self.scripts_mutex = Semaphore(1) 
        self.location_locks_mutex = None 
        self.location_locks = {} 
        self.timepoint_barrier = None 

        # Background Execution: Main management thread and worker pool.
        self.thread = DeviceThread(self)
        self.workers = [DeviceWorker(self) for _ in range(8)]

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization of shared synchronization resources.
        Logic: Elects device 0 as the master to initialize global barriers and locks.
        """
        if self.device_id == 0:
            self.timepoint_barrier = ReusableBarrierCond(len(devices))
            self.location_locks_mutex = Lock()
            self.setup_ready.set() 
        else:
            # Sync: Peer devices wait for the master to complete resource allocation.
            device = next(device for device in devices if device.device_id == 0)
            device.setup_ready.wait() 
            self.timepoint_barrier = device.timepoint_barrier
            self.location_locks = device.location_locks
            self.location_locks_mutex = device.location_locks_mutex

        /**
         * Block Logic: Dynamic lock allocation for unique sensor locations.
         * Critical Section: Protected by location_locks_mutex to ensure global consistency.
         */
        with self.location_locks_mutex:
            for location in self.sensor_data.keys():
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()

        # Lifecycle: Initiates the management and worker threads.
        self.thread.start()
        for worker in self.workers:
            worker.start()

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing script for execution on specific location data.
        Functional Utility: Asynchronous task submission via the internal work queue.
        """
        self.neighbours_set.wait()

        if script is not None:
            with self.scripts_mutex:
                self.scripts.append((script, location))
            self.work_queue.put((script, location))
        else:
            # Termination: Signals the end of the current timepoint.
            self.neighbours_set.clear() 
            self.timepoint_done.set() 

    def get_data(self, location):
        """
        @brief Retrieval of local sensor state (assumes external locking).
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        @brief Updates local sensor state (assumes external locking).
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Graceful teardown of the device and its worker pool.
        Protocol: Sends 'poison pills' (None, None) to the queue to terminate workers.
        """
        self.thread.join()
        for worker in self.workers:
            self.work_queue.put((None, None))

        for worker in self.workers:
            worker.join()


class DeviceThread(Thread):
    """
    @brief Management thread responsible for orchestrating temporal cycles (timepoints).
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main loop coordinating neighbors discovery and timepoint synchronization.
        """
        while True:
            # Discovery: Fetches the current topological neighbors from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                break

            # Task Injection: Reschedules cached scripts for the new timepoint.
            for (script, location) in self.device.scripts:
                self.device.work_queue.put((script, location))

            self.device.neighbours_set.set()

            # Barrier: Wait for local processing completion.
            self.device.timepoint_done.wait()

            # Sync: Block until all tasks in the current work queue are acknowledged.
            self.device.work_queue.join()

            self.device.timepoint_done.clear()

            # Global Sync: Ensure all devices in the cluster are aligned before the next cycle.
            self.device.timepoint_barrier.wait()


class DeviceWorker(Thread):
    """
    @brief Specialized thread for executing data aggregation scripts.
    Functional Utility: Implements a distributed Map-Reduce operation on a shared location.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device %d Worker" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Continuous consumer loop for the device's work queue.
        """
        while True:
            # Task Acquisition: Retrieves the next (script, location) pair.
            (script, location) = self.device.work_queue.get(block=True)

            # Protocol: Poison pill check for graceful shutdown.
            if script is None and location is None:
                self.device.work_queue.task_done()
                break

            /**
             * Block Logic: Critical section for distributed data processing.
             * Invariant: Exclusive access to the location's data across the entire cluster.
             */
            with self.device.location_locks[location]:
                script_data = []

                # Collection: Aggregates data from neighbors and self.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Execution: Runs the aggregation logic.
                    result = script.run(script_data)

                    # Dissemination: Writes the result back to all participating entities.
                    for device in self.device.neighbours:
                        device.set_data(location, result)

                    self.device.set_data(location, result)

            # Signal task completion to the queue manager.
            self.device.work_queue.task_done()
