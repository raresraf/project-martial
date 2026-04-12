"""
@file device.py
@brief Distributed sensor processing system with resource-managed thread pooling and cyclic synchronization.
@details Implements a peer-to-peer network of sensing units that execute data-aggregation 
scripts. Uses a task-based 'canal' for worker distribution, location-specific locking 
for consistency, and a custom two-phase barrier for cluster coordination.
"""

from threading import Event, Thread, Semaphore, Lock

class Device(object):
    """
    @brief Logic controller for an autonomous sensor unit in a distributed cluster.
    Functional Utility: Manages local data buffers, stages tasks via a synchronized canal, 
    and maintains synchronization resources shared with peer devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of sensor location-reading pairs.
        @param supervisor entity providing topology discovery and cluster coordination.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        
        # Lock Registry: dedicated lock for every local sensor location to ensure atomicity.
        self.locks = {loc: Lock() for loc in sensor_data.keys()}
        
        self.supervisor = supervisor
        # Sync: Semaphore to notify workers of newly available tasks in the canal.
        self.script_received = Semaphore(0)
        self.scripts = []
        self.scripts_number = 0
        # Task Staging: Shared canal used to distribute work to the thread pool.
        self.canal = []
        self.timepoint_done = Event()
        
        # Lifecycle: Spawns the main device management thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.time = 0

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collaborative initialization of shared synchronization resources.
        Logic: Designates device 0 as the allocator for the cluster-wide cyclic barrier.
        """
        self.devices = devices
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in self.devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Schedules a processing script for the current temporal timepoint.
        Logic: Pushes the task into the canal and signals available workers.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Transition: signals end of the assignment phase and activates workers.
            for task in self.scripts:
                self.scripts_number += 1
                self.canal.insert(0, task)
                self.script_received.release()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieval of local sensor readings.
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


class DeviceThread(Thread):
    """
    @brief Management thread coordinating cycle execution and local worker pool.
    Architecture: Manages a pool of 16 worker threads for task execution.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads_number = 16
        # Callback Sync: Semaphore to track completion of individual tasks.
        self.tasks_done = Semaphore(0)
        self.stop = False

        # Initialization: Persistent pool of workers.
        self.threads = [Thread(target=self.script_work, args=(i,))
                        for i in range(self.threads_number)]

    def start_threads(self):
        """
        @brief Activates the worker thread pool.
        """
        for thread in self.threads:
            thread.start()

    def stop_threads(self):
        """
        @brief Blocks until all worker threads in the pool have terminated.
        """
        for thread in self.threads:
            thread.join()

    def script_work(self, id):
        """
        @brief execution logic for pool workers.
        Architecture: Implements distributed Map-Reduce on shared locations.
        """
        while True:
            # Acquisition: Blocks until a script is received via the canal.
            self.device.script_received.acquire()
            script, location = self.device.canal.pop()
            
            # Protocol: Poison pill check for worker termination.
            if script is None:
                break

            /**
             * Block Logic: Data collection and aggregation.
             * Logic: Queries self and neighbors for location-specific data.
             */
            all_data = [(device.get_data(location), device)
                        for device in self.device.neighbours
                        if device.get_data(location)]

            data = self.device.get_data(location)
            if data is not None:
                all_data.append((data, self.device))
            
            script_data = [x for x, _ in all_data]
            participants = [x for _, x in all_data]

            if len(script_data) > 1:
                # Computation Phase.
                result = script.run(script_data)
                
                # Reduce/Sync Phase: Propagates result to all participants.
                # Invariant: Uses atomic locking per neighbor to ensure consistency.
                for participant in participants:
                        with participant.locks[location]:
                            current_state = participant.get_data(location)
                            # Logic: monotonic update (greedy maximization of state).
                            if current_state < result:
                                participant.set_data(location, result)

            # Signal task finalization to the management thread.
            self.tasks_done.release()

    def run(self):
        """
        @brief Main cycle loop: discovery -> parallel processing -> barrier synchronization.
        """
        # Discovery: initial neighborhood acquisition.
        self.device.neighbours = self.device.supervisor.get_neighbours()
        self.start_threads()
        
        # Initial alignment.
        self.device.barrier.wait()

        while True:
            if self.device.neighbours is None:
                # Shutdown: injects poison pills for every worker in the pool.
                self.stop = True
                for i in range(self.threads_number):
                    self.device.canal.insert(0, (None, None))
                    self.device.script_received.release()
                break

            # Sync: Wait for local task assignments.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            
            # Synchronization: Block until all dispatched scripts for this timepoint are finalized.
            for i in range(self.device.scripts_number):
                self.tasks_done.acquire()
            
            self.device.scripts_number = 0
            
            # Global Sync: Aligns all devices in the cluster before starting the next timepoint.
            self.device.barrier.wait()
            self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.time += 1

        self.stop_threads()


class ReusableBarrier():
    """
    @brief Implementation of a two-phase cyclic barrier using semaphores.
    Functional Utility: Provides reliable synchronization points for multi-threaded cycles.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Executes the two-turnstile protocol.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Turnstile logic: blocks arrival until threshold is met, then releases all.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Wakes all threads blocked on this turnstile.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Resets the phase counter for the next cycle.
                count_threads[0] = self.num_threads
        threads_sem.acquire()
