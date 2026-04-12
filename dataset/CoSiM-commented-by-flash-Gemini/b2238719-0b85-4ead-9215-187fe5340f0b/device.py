"""
@file device.py
@brief Distributed sensor unit simulation with a coordinated worker-pool and double-barrier synchronization.
@details Implements a peer-to-peer network of devices that perform synchronized data 
aggregation. Leverages a local pool of worker threads, controlled via events and 
reusable barriers, to process tasks in discrete temporal timepoints.
"""

from threading import Event, Thread, Semaphore
from barrier import ReusableBarrierSem

class Device(object):
    """
    @brief Logic controller for an autonomous sensing entity.
    Functional Utility: Manages local sensor data, coordinates a pool of internal 
    worker threads, and shares cluster-wide synchronization primitives.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of sensor location readings.
        @param supervisor Entity providing topological neighbor discovery.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Synchronization Events: Manage cross-device and cross-thread alignment.
        self.barrier_set = Event()
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        # Architecture: High-level management thread.
        self.thread = DeviceThread(self)
        
        self.barrier = None
        self.neighbours = []
        self.data_locks = []
        self.thread_list = []
        
        # Worker Pool: Fixed capacity of 8 threads for local task execution.
        self.worker_number = 8
        self.worker_barrier = ReusableBarrierSem(self.worker_number)
        
        # Task Staging: Queue for Distributing scripts to the pool.
        self.script_queue = []
        self.script_lock = Semaphore(1)
        
        # Control Flags.
        self.exit_flag = Event()
        self.tasks_finished = Event()
        self.start_tasks = Event()

    def set_flag(self):
        """
        @brief Signals that the global barrier has been successfully propagated.
        """
        self.barrier_set.set()

    def set_barrier(self, barrier):
        """
        @brief Sets the reference to the shared cyclic barrier.
        """
        self.barrier = barrier

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collaborative initialization of cluster-wide synchronization primitives.
        Logic: Designates device 0 as the allocator for global barriers and the lock registry.
        """
        if self.device_id == 0:
            # Allocation: Creates the shared barrier for all devices in the cluster.
            self.barrier = ReusableBarrierSem(len(devices))
            
            # Discovery: Finds the maximum location index to size the global lock array.
            location_index = -1
            for dev in devices:
                for k in dev.sensor_data:
                    if k > location_index:
                        location_index = k

            # Propagation: Distributes shared primitives to all peers.
            for dev in devices:
                dev.set_barrier(self.barrier)
                dev.set_flag()

            # Global Resource: creates a binary semaphore (Lock) for every sensor location.
            self.data_locks = {loc : Semaphore(1) for loc in range(location_index+1)}
            for dev in devices:
                dev.data_locks = self.data_locks
        else:
            # Sync: Peers wait for the master device to complete resource allocation.
            self.barrier_set.wait()
        
        # Lifecycle: Initiates the management and worker threads.
        self.thread.start()
        for tid in range(self.worker_number):
            thread = WorkerThread(self, tid)
            self.thread_list.append(thread)
            thread.start()

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing task for the current unit of time.
        """
        if script is not None:
            self.scripts.append((script, location))
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
        @brief updates local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates all device threads.
        """
        for thread in self.thread_list:
            thread.join()
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Management thread coordinating the temporal timepoint cycles.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main cycle loop: discovery -> synchronization -> task activation -> wait.
        """
        while True:
            # Global Sync: Aligns all devices at the start of the timepoint.
            self.device.barrier.wait()

            # Discovery: Queries supervisor for current neighbors.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                # Termination: Signals shutdown to the worker pool.
                self.device.exit_flag.set()
                self.device.start_tasks.set()
                break

            # Sync: Wait for local task assignments to finalize.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Task Submission: Populates the internal queue for worker consumption.
            self.device.script_queue = list(self.device.scripts)

            # Execution Trigger: Wakes the worker pool to begin processing.
            self.device.start_tasks.set()

            # Sync: Wait for the local pool to finish the current workload.
            self.device.tasks_finished.wait()
            self.device.tasks_finished.clear()


class WorkerThread(Thread):
    """
    @brief Reusable worker thread that executes data aggregation scripts.
    Functional Utility: Implements distributed Map-Reduce on shared locations.
    """
    
    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Worker-%d-%d" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        @brief Persistent consumer loop with double-barrier coordination.
        """
        iterations = 0
        while True:
            # Arrival Barrier: Workers wait here for the start of a cycle.
            self.device.worker_barrier.wait()

            # Callback: Designates thread 0 to signal the Master that work is complete.
            if self.thread_id == 0 and iterations != 0:
                self.device.tasks_finished.set()

            # Activation Sync: Blocks until the DeviceThread master triggers execution.
            self.device.start_tasks.wait()

            # Departure Barrier: Ensures all workers are active before clearing the trigger.
            self.device.worker_barrier.wait()
            if self.thread_id == 0:
                self.device.start_tasks.clear()
            
            iterations += 1
            if self.device.exit_flag.is_set():
                break

            /**
             * Block Logic: Script consumption and execution.
             * Logic: Atomically pulls a task from the shared device queue.
             */
            self.device.script_lock.acquire()
            if self.device.script_queue:
                (script, location) = self.device.script_queue.pop(0)
                self.device.script_lock.release()
            else:
                # Logic: No tasks remaining; loop back to barriers.
                self.device.script_lock.release()
                continue

            /**
             * Block Logic: Distributed data processing.
             * Critical Section: Uses shared location-specific locks to ensure atomicity.
             */
            self.device.data_locks[location].acquire()
            
            script_data = []
            # Map Phase: Aggregates state from neighbors.
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
                
                # Reduce/Update Phase: updates state across all participants.
                for device in self.device.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
            
            # Release: Free the location for other participating devices.
            self.device.data_locks[location].release()
