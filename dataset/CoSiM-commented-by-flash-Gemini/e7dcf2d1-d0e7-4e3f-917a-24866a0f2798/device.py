"""
@e7dcf2d1-d0e7-4e3f-917a-24866a0f2798/device.py
@brief Distributed sensor network simulation with workload partitioning and worker pools.
This module implements a coordinated processing framework where node managers 
(DeviceThread) partition assigned computational tasks among a local pool of 8 worker 
threads (Worker). Consistency is guaranteed through a network-wide pool of spatial 
locks and global barriers for temporal synchronization. Local thread alignment is 
managed via an internal double-gate rendezvous point.

Domain: Parallel Systems, Workload Partitioning, Two-Phase Synchronization.
"""

from threading import Event, Thread, Lock, Semaphore


class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local data state and coordinates the distribution 
    of global synchronization resources (barriers and spatial locks).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # Primary node management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Shared resources populated during setup.
        self.global_barrier = None
        self.locks = None


    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource factory.
        Logic: Coordinator node (ID 0) initializes the shared network-wide barrier 
        and discovers all unique spatial locations to pre-allocate a mutex for each.
        """
        if self.device_id == 0:
            # Atomic setup of the global temporal rendezvous point.
            self.global_barrier = ReusableBarrier(len(devices))

            for device in devices:
                device.global_barrier = self.global_barrier

            # Block Logic: Spatial Lock discovery.
            self.locks = []
            locations = devices[0].sensor_data.keys()
            for index in range(1, len(devices)):
                aux = devices[index].sensor_data.keys()
                # Aggregate unique location identifiers across the network group.
                locations = list(set(locations).union(aux))

            # Allocation: Create a dedicated lock for every spatial location.
            for _ in range(len(locations)):
                self.locks.append(Lock())

            # Propagation: distribute resources to all peer nodes.
            for device in devices:
                device.locks = self.locks


    def assign_script(self, script, location):
        """Registers a task and signals completion of the simulation step assignment."""
        if script is not None:
            self.scripts.append((script, location))
        else:
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


class DeviceThread(Thread):
    """
    Node-level manager.
    Functional Utility: Orchestrates simulation phases and partitions computational 
    workloads across the local parallel worker pool.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        # Local Barrier: coordinates the 8 workers + 1 orchestrator thread.
        self.barrier_parent = ReusableBarrier(9)

        # Spawns persistent worker pool.
        self.threads = []
        for _ in range(8):
            self.threads.append(Worker(self.device, None, None, self.barrier_parent))

        for thread in self.threads:
            thread.start()


    def run(self):
        """
        Main simulation loop for the orchestrator.
        Algorithm: Iterative topology refresh, task partitioning, and consensus.
        """
        while True:
            # Refresh topology.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block until assignment phase is complete.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Block Logic: Workload Partitioning.
            # Algorithm: Divides assigned scripts into 8 chunks for the worker pool.
            if len(self.device.scripts) <= 8:
                # Direct assignment for small workloads.
                for index in range(len(self.device.scripts)):
                    self.threads[index].script = self.device.scripts[index]
                    self.threads[index].neighbours = neighbours
            else:
                # Chunked assignment for large workloads.
                # Note: current implementation overwrites worker's script in loop.
                aux = len(self.device.scripts)/8
                inf = 0
                sup = aux
                for index in range(8):
                    if index == 7:
                        sup = len(self.device.scripts)
                    self.threads[index].neighbours = neighbours
                    for index2 in range(inf, sup):
                        self.threads[index].script = self.device.scripts[index2]
                    inf += aux
                    sup += aux


            # Barrier Point 1: Trigger workers to start processing.
            self.barrier_parent.wait()

            # Wait for supervisor to signal simulation step end.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Barrier Point 2: Wait for workers to complete local tasks.
            self.barrier_parent.wait()
            
            # Global Consensus: wait for all devices in the network to align.
            self.device.global_barrier.wait()

        # Shutdown Logic: signal termination to all workers.
        for thread in self.threads:
            thread.out = 1
        self.barrier_parent.wait()

        # Reclaim worker resources.
        for thread in self.threads:
            thread.join()


class ReusableBarrier(object):
    """
    Two-phase reusable barrier implementation using semaphores.
    Functional Utility: Implements a double-gate rendezvous mechanism to ensure 
    consistent group transitions across simulation cycles.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: participants count.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Executes the arrival and exit phases of the rendezvous."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Internal gate logic using atomic counter decrement and semaphore release."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Worker(Thread):
    """
    Persistent worker thread implementation.
    Functional Utility: Executes a computational script while maintaining 
    network-wide spatial mutual exclusion for the target location.
    """

    def __init__(self, device, script, neighbours, barrier_parent):
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.neighbours = neighbours
        self.workers_barrier = barrier_parent
        self.out = 0

    def run(self):
        """
        Worker execution loop.
        Algorithm: Iterative sequence: Rendezvous -> Process -> Rendezvous.
        """
        while True:
            # Wait for orchestrator signal to begin a phase.
            self.workers_barrier.wait()

            # Check for termination signal.
            if self.out == 1:
                break

            if self.neighbours != None and self.script is not None:
                script_data = []

                # Critical Section: Spatial mutual exclusion across the entire network.
                self.device.locks[self.script[1]].acquire()
                
                # Aggregate neighborhood data.
                for device in self.neighbours:
                    data = device.get_data(self.script[1])
                    if data is not None:
                        script_data.append(data)
                
                # Include local state.
                data = self.device.get_data(self.script[1])
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Compute result and propagate to peers.
                    result = self.script[0].run(script_data)
                    for device in self.neighbours:
                        device.set_data(self.script[1], result)
                    self.device.set_data(self.script[1], result)
                
                # Release spatial mutex.
                self.device.locks[self.script[1]].release()
            
            # Rendezvous point: signal completion of local processing.
            self.workers_barrier.wait()
