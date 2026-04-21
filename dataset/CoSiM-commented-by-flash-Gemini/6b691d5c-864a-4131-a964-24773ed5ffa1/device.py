
"""
@file device.py
@brief Distributed device simulation framework with dynamic worker orchestration.

Functional Intent: Implements a parallel execution environment for autonomous 
devices in a simulated network. Features a hierarchical coordination model where 
each device dynamically spawns ScriptWorker threads for localized data processing. 
Coordinates neighbor discovery, ensures eventual consistency via distributed 
locking, and synchronizes time-discrete epochs through a custom reusable barrier.

Domain: Production Systems, Distributed Computing, Parallel Processing.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    @brief Custom reusable synchronization barrier implemented using semaphores.
    
    Algorithm: Two-phase (turnstile) synchronization.
    Logic: Employs a double-gating mechanism to ensure that all threads rendezvous 
    before release and that the barrier is fully cleared before re-entry for the 
    subsequent epoch.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the barrier with fixed participation quotas.
        """
        self.num_threads = num_threads
        # Logic: Using lists for mutable counter references across synchronization calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until the full group reaches the rendezvous point.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements an atomic arrival and release sequence for a group of threads.
        """
        # Synchronization: Critical section for protecting the group counter.
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Logic: The final thread releases the entire group and resets the state.
                n_threads = self.num_threads
                while n_threads > 0:
                    threads_sem.release()
                    n_threads -= 1
                count_threads[0] = self.num_threads
        
        # Block Logic: Arrival gate.
        threads_sem.acquire()


class Device(object):
    """
    @brief Represents a computational entity in the simulated distributed swarm.
    
    Functional Utility: Manages local data residency and coordinates task 
    processing across neighborhood boundaries. Features a dedicated manager 
    thread and supports dynamic shared-lock discovery for consistent data 
    manipulation across the network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device and triggers its management thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Synchronization: Events for task delivery and epoch boundary signaling.
        self.script_received = Event()
        self.timepoint_done = Event()
        
        # Control: Hierarchical management thread.
        self.thread = DeviceThread(self)
        self.devices = []
        self.barrier = None
        self.workers = []
        
        # Synchronization: Dynamic pool of mutexes for protecting specific data addresses.
        keys = range(60)
        self.loc_barrier = {key: None for key in keys}
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Bootstraps the distributed synchronization plane across all nodes.
        
        Algorithm: Peer-to-peer barrier distribution.
        Logic: Ensures that all nodes share a single global ReusableBarrier 
        instance to align their discrete execution time-steps.
        """
        # Block Logic: Global barrier propagation.
        # Pre-condition: Barrier is initialized by the first calling device.
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        @brief Enqueues a script and resolves its synchronization context.
        """
        if script is not None:
            self.scripts.append((script, location))
            
            # Logic: Lazy discovery/initialization of distributed locks.
            # Invariant: All devices must use the same Lock object for a given 'location'.
            if self.loc_barrier[location] is None:
                found_lock = False
                for device in self.devices:
                    if device.loc_barrier[location] is not None:
                        self.loc_barrier[location] = device.loc_barrier[location]
                        found_lock = True
                        break
                if not found_lock:
                    self.loc_barrier[location] = Lock()
            
            self.script_received.set()
        else:
            # Logic: Signaling None terminates task assignment for the current epoch.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves local data (requires external locking for consistency).
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates local data state.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the orchestration thread.
        """
        self.thread.join()


class ScriptWorker(Thread):
    """
    @brief Execution thread responsible for neighborhood-aware data transformations.
    
    Algorithm: Distributed state reduction and broadcast.
    """
    
    def __init__(self, device, neighbours, script, location):
        """
        @brief Initializes the worker with its specific task and neighborhood context.
        """
        Thread.__init__(self, name="Script Worker for Device %d, Loc %d" % (device.device_id, location))
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief Worker execution loop.
        
        Logic: 
        1. Acquires the global lock for the specific data location.
        2. Aggregates data from neighbors.
        3. Executes script and broadcasts results back to peers.
        4. Releases the lock.
        """
        # Synchronization: Mutual exclusion across the network for the target data address.
        self.device.loc_barrier[self.location].acquire()
        
        script_data = []
        # Block Logic: Neighborhood state aggregation.
        for device_neigh in self.neighbours:
            data = device_neigh.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Functional Intent: Executes user-defined processing logic.
            result = self.script.run(script_data)
            
            # Block Logic: Update dissemination.
            for device_neigh in self.neighbours:
                device_neigh.set_data(self.location, result)
            
            self.device.set_data(self.location, result)
        
        self.device.loc_barrier[self.location].release()


class DeviceThread(Thread):
    """
    @brief Orchestration thread responsible for epoch transitions and worker lifecycle.
    
    Logic: Manages discrete time-steps by discovering topology, spawning workers, 
    and participating in global consensus barriers.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core device operational cycle.
        
        Algorithm: Dynamic parallel task orchestration.
        Logic: 
        1. Identifies neighborhood configuration via supervisor.
        2. Waits for all scripts for the current time-step to be assigned.
        3. Spawns ScriptWorker threads for concurrent execution.
        4. Joins all local workers and synchronizes at the global barrier.
        """
        while True:
            # Block Logic: Neighbor discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Poison Pill: Shutdown signal from supervisor.
                break

            # Synchronization: Wait for epoch-based script delivery.
            self.device.timepoint_done.wait()

            # Task Dispatch: Creates dedicated worker threads for each assigned script.
            for (script, location) in self.device.scripts:
                worker = ScriptWorker(self.device, neighbours, script, location)
                self.device.workers.append(worker)

            for worker in self.device.workers:
                worker.start()

            # Synchronization: Ensures all local computations are complete before the global rendezvous.
            for worker in self.device.workers:
                worker.join()

            # Finalization: State reset for the next epoch.
            self.device.workers = []
            self.device.scripts = []
            self.device.timepoint_done.clear()
            
            # Synchronization: Global rendezvous ensuring all devices are ready to advance.
            self.device.barrier.wait()
