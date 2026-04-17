"""
@f015479c-699b-4648-98c5-4fa5406028c3/device.py
@brief Distributed sensor network simulation with monotonic state convergence.
This module implements a parallel processing framework where transient worker threads 
(WorkerThread) execute computational scripts. It utilizes a monotonic update 
strategy, ensuring that sensor values only increase over time (max(result, current)), 
which facilitates system-wide convergence. Consistency is guaranteed through global 
neighbor locks and a two-phase semaphore-based synchronization barrier.

Domain: Monotonic State Updates, Distributed Convergence, Parallel Fork-Join.
"""

from threading import Thread, Lock, Semaphore, Event

class ReusableBarrierSem(object):
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Provides a robust synchronization point for a fixed group 
    of threads using a double-gate mechanism with semaphores to prevent phase 
    bleeding.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: participants count.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Orchestrates the two-phase arrival and exit protocol."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """Arrival gate: blocks threads until the group threshold is met."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Release all threads for the current phase.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Exit gate: ensures total group clearance before allowing reuse."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local data state, coordinates the distribution 
    of global synchronization resources, and provides the interface for 
    monotonic data updates.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # Main lifecycle management thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        # Global lock shared across peers to protect neighborhood updates.
        self.lock_neigh = None
        # Local mutex for node-specific state protection.
        self.lock_mine = Lock()

    def __str__(self):
        return "Device %d" % self.device_id



    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node (ID 0) initializes the shared barrier and a 
        single mutex used to protect all neighbor updates across the network.
        """
        no_devices = len(devices)
        lock_neigh = Lock()
        barrier = ReusableBarrierSem(no_devices)

        if self.device_id == 0:
            # Propagation: Distribute shared resources to peer devices.
            for i in range(no_devices):
                devices[i].barrier = barrier
                devices[i].lock_neigh = lock_neigh


    def assign_script(self, script, location):
        """Registers a task and signals the orchestration thread."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal end of workload assignment phase.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None


    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully joins the orchestration thread."""
        self.thread.join()


class WorkerThread(Thread):
    """
    Transient computational worker.
    Functional Utility: Executes a script while maintaining system-wide 
    monotonic state convergence.
    """

    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self, name="Worker Thread")
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def collect_data(self, location_data):
        """Aggregates sensor state from self and the entire neighborhood."""
        location_data.append(self.device.get_data(self.location))
        for i in range(len(self.neighbours)):
            data = self.neighbours[i].get_data(self.location)
            location_data.append(data)

    def update_neighbours(self, result):
        """
        Atomic Monotonic Propagation to neighbors.
        Logic: Uses the global 'lock_neigh' and max() to ensure forward state flow.
        """
        no_neigh = len(self.neighbours)
        for i in range(no_neigh):
            self.device.lock_neigh.acquire()
            value = self.neighbours[i].get_data(self.location)
            # Monotonic convergence rule.
            self.neighbours[i].set_data(self.location, max(result, value))
            self.device.lock_neigh.release()

    def update_self(self, result):
        """Atomic Monotonic update of local node state."""
        self.device.lock_mine.acquire()
        value = self.device.get_data(self.location)
        self.device.set_data(self.location, max(result, value))
        self.device.lock_mine.release()

    def run(self):
        """
        Execution Logic.
        Algorithm: Aggregate -> Compute -> Monotonic Propagate.
        """
        location_data = []
        self.collect_data(location_data)

        if len(location_data) > 0:
            # Process pre-aggregated neighborhood state.
            result = self.script.run(location_data)
            self.update_neighbours(result)
            self.update_self(result)


class DeviceThread(Thread):
    """
    Node-level simulation manager.
    Functional Utility: Orchestrates simulation phases and dispatches scripts 
    using a transient fork-join parallel processing model.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        """
        Main execution loop.
        Algorithm: Iterative Fork -> Join -> Consensus cycle.
        """
        # Static task buffer sized for up to 200 concurrent scripts.
        threads = [None] * 200
        while True:
            # Topology refresh.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block until assignment phase is complete.
            self.device.script_received.wait()
            self.device.script_received.clear()
            
            # Phase 1: Fork.
            # Spawns a transient worker for every script in the current batch.
            for i in range(len(self.device.scripts)):
                (script, location) = self.device.scripts[i]
                threads[i] = WorkerThread(self.device, script, \
                    location, neighbours)
                threads[i].start()

            # Phase 2: Join.
            # wait for all local parallel processing to finalize.
            for i in range(len(self.device.scripts)):
                threads[i].join()

            # Phase 3: Consensus.
            # Global rendezvous followed by local simulation step synchronization.
            self.device.barrier.wait()
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
