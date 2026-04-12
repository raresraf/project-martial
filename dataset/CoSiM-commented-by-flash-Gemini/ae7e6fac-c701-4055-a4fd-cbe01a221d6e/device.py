"""
@file device.py
@brief Distributed sensor processing system using a centralized thread-pool and barrier synchronization.
@details Implements a peer-to-peer network of sensing units that execute data-aggregation 
scripts concurrently. Leverages a worker thread pool for local parallelism and a lead-device 
coordinated barrier for temporal alignment across the cluster.
"""

from threading import Event, Thread, Lock, Condition, Semaphore

class Device(object):
    """
    @brief Core logic controller for an autonomous sensing entity in a distributed network.
    Functional Utility: Manages local data buffers, stages tasks, and coordinates cluster-wide 
    synchronization through shared locks and barriers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of sensor location-value pairs.
        @param supervisor Entity providing topological neighbor discovery services.
        """
        self.devices = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        
        # Lifecycle: Spawns the main device orchestrator thread.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Sync State: Identifies the master coordinator for cluster-wide primitives.
        self.lead_device_index = -1
        self.location_locks = []

        if device_id == 0:
            # Shared synchronization resources managed by the lead device.
            self.threads_finished_count = 0
            self.next_timepoint_cond = Condition()
            self.resource_ready = Event()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization of shared synchronization resources.
        Logic: Designates device 0 as the allocator for the cluster-wide lock registry.
        """
        self.devices = devices

        for i in xrange(len(self.devices)):
            if devices[i].device_id == 0:
                self.lead_device_index = i
                break

        if self.device_id == 0:
            self.resource_ready.clear()

            # Discovery: Finds the maximal location index to size the lock array.
            max_location = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > max_location:
                        max_location = location

            # Allocation: Creates a dedicated Lock for every unique sensor location.
            for _ in range(max_location + 1):
                self.location_locks.append(Lock())

            # Propagation: Shares the lock registry with all peers.
            for device in devices:
                device.location_locks = self.location_locks

            self.resource_ready.set()
        else:
            # Sync: Peer devices wait for the lead device to finalize setup.
            self.devices[self.lead_device_index].resource_ready.wait()

    def assign_script(self, script, location):
        """
        @brief Schedules a processing task for the current unit of time.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Protocol: Signals end of the script submission phase.
            self.timepoint_done.set()

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
        @brief Gracefully terminates the device orchestrator thread.
        """
        self.thread.join()

    def notify_finish(self):
        """
        @brief Barrier synchronization: blocks until all devices in the cluster arrive.
        Functional Utility: ensures temporal alignment before the next timepoint.
        """
        leader = self.devices[self.lead_device_index]
        leader.next_timepoint_cond.acquire()
        leader.threads_finished_count += 1

        if leader.threads_finished_count == len(self.devices):
            # Reset and Release: Last device to arrive wakes all peers.
            leader.threads_finished_count = 0
            leader.next_timepoint_cond.notifyAll()
        else:
            # Wait: Suspends execution until the arrival threshold is met.
            leader.next_timepoint_cond.wait()

        leader.next_timepoint_cond.release()


class DeviceThread(Thread):
    """
    @brief Orchestrator thread managing discrete execution units (timepoints).
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Architecture: Dedicated worker pool for task execution.
        self.thread_pool = WorkerThreadPool(device)

    def run(self):
        """
        @brief Main coordination loop: discovery -> parallel processing -> barrier.
        """
        while True:
            # Discovery: Queries the supervisor for network topology.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                self.thread_pool.shutdown()
                break

            # Sync: Wait for local task assignments to complete.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Task Submission: Offloads scripts to the worker pool.
            for (script, location) in self.device.scripts:
                self.thread_pool.do_work(script, location, neighbours)

            # Sync: Ensures all local scripts are finalized before synchronization.
            self.thread_pool.wait_to_finish_work()

            # Global Sync: Blocks at the cluster barrier.
            self.device.notify_finish()


class WorkerThreadPool(object):
    """
    @brief Resource-bounded execution environment for device tasks.
    Functional Utility: Manages a pool of 8 reusable worker threads.
    """
    
    def __init__(self, device):
        self.device = device
        self.work_finished_event = Event()
        self.work_finished_event.set()
        self.worker_pool = []
        # Queue state: identifies workers currently available for tasks.
        self.ready_for_work_queue = []
        self.worker_semaphore = Semaphore(8)
        self.queue_lock = Lock()

        # Initialization: Spawns persistent worker threads.
        for _ in xrange(8):
            thread = SimpleWorker(self, self.device)
            self.worker_pool.append(thread)
            self.ready_for_work_queue.append(thread)
            thread.start()

    def do_work(self, script, location, neighbours):
        """
        @brief Dispatches a task to an available worker.
        """
        if self.work_finished_event.isSet():
            self.work_finished_event.clear()
        
        # Throttling: Blocks if all 8 workers are occupied.
        self.worker_semaphore.acquire()
        
        with self.queue_lock:
            worker = self.ready_for_work_queue.pop(0)
        
        worker.assign_task(script, location, neighbours)

    def shutdown(self):
        """
        @brief Gracefully terminates all pool workers.
        """
        for worker in self.worker_pool:
            worker.stop_signal = True
            worker.data_ready.release()
        for worker in self.worker_pool:
            worker.join()

    def worker_finished(self, worker):
        """
        @brief Call-back used by workers to return themselves to the available pool.
        """
        with self.queue_lock:
            self.ready_for_work_queue.append(worker)
            # Logic: Signals pool-wide idle state if all workers have returned.
            if len(self.ready_for_work_queue) == 8:
                self.work_finished_event.set()
        
        self.worker_semaphore.release()

    def wait_to_finish_work(self):
        """
        @brief Blocks until the current timepoint's tasks are completed.
        """
        self.work_finished_event.wait()


class SimpleWorker(Thread):
    """
    @brief Individual execution thread in the device pool.
    Architecture: Implements a distributed Map-Reduce operation.
    """
    
    def __init__(self, worker_pool, device):
        Thread.__init__(self, name="Worker-%d" % device.device_id)
        self.worker_pool = worker_pool
        self.stop_signal = False
        self.data_ready = Semaphore(0)
        self.device = device
        self.script = None
        self.location = None
        self.neighbours = None

    def assign_task(self, script, location, neighbours):
        """
        @brief Configuration phase before activation.
        """
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.data_ready.release()

    def run(self):
        """
        @brief Persistent consumer loop.
        """
        while True:
            # Wait for task assignment or shutdown signal.
            self.data_ready.acquire()
            if self.stop_signal:
                break

            /**
             * Block Logic: Synchronized aggregation logic.
             * Critical Section: Uses shared location locks to ensure atomic state updates.
             */
            with self.device.location_locks[self.location]:
                script_data = []
                
                # Map Phase: Aggregates data from neighbors.
                for device in self.neighbours:
                    data = device.get_data(self.location)
                    if data is not None:
                        script_data.append(data)
                
                # Inclusion: local state.
                data = self.device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Computation: execution of the aggregation script.
                    result = self.script.run(script_data)
                    
                    # Reduce/Update Phase: writes back the state change to all participants.
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                    self.device.set_data(self.location, result)
            
            # Protocol: Signals availability to the pool manager.
            self.worker_pool.worker_finished(self)
