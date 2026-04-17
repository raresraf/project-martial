"""
@e7dcf2d1-d0e7-4e3f-917a-24866a0f2798/device.py
@brief Distributed sensor processing simulation using a persistent worker pool and tiered barrier synchronization.
* Algorithm: Static task partitioning across 8 persistent `Worker` threads with internal (local) and global (cluster) multi-phase semaphore barriers.
* Functional Utility: Orchestrates simulation timepoints across a network of devices by managing distributed data acquisition and synchronized state updates using a shared global lock registry for sensor locations.
"""

from threading import Event, Thread, Lock, Semaphore


class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings, internal worker pool, and shared synchronization infrastructure.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and bootstraps the coordination thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event() # Intent: Signals completion of task delivery for the current phase.
        self.thread = DeviceThread(self)
        self.thread.start()

        self.global_barrier = None # Intent: Global barrier for cluster-wide alignment.
        self.locks = None          # Intent: Registry for shared locks indexed by sensor location.


    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization and distribution of shared simulation resources.
        Invariant: Root node (ID 0) initializes the collective barrier and discovers all global sensor locations to create granular locks.
        """
        if self.device_id == 0:
            self.global_barrier = ReusableBarrier(len(devices))

            for device in devices:
                device.global_barrier = self.global_barrier

            # Logic: Discovers and aggregates unique sensor locations across the entire cluster.
            self.locks = []
            locations = devices[0].sensor_data.keys()
            for index in range(1, len(devices)):
                aux = devices[index].sensor_data.keys()
                locations = list(set(locations).union(aux))

            # Logic: Pre-allocates a lock for every discovered global location.
            for _ in range(len(locations)):
                self.locks.append(Lock())

            for device in devices:
                device.locks = self.locks


    def assign_script(self, script, location):
        """
        @brief Receives a processing task for the current simulation cycle.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Signals completion of script arrival for this node.
            self.script_received.set()
            self.timepoint_done.set()

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
        @brief Terminates the device coordination thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief coordination thread managing temporal phases and worker task dispatching.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator and its internal worker pool (8 threads).
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        # Intent: Internal device barrier for aligning the local coordinator and its 8 workers.
        self.barrier_parent = ReusableBarrier(9)

        self.threads = []
        for _ in range(8):
            self.threads.append(Worker(self.device, None, None, self.barrier_parent))

        for thread in self.threads:
            thread.start()


    def run(self):
        """
        @brief main loop for the device node coordination.
        Algorithm: Phased synchronization managing neighbor discovery and chunked task assignment.
        """
        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits for script delivery start.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Task Partitioning Phase.
            if len(self.device.scripts) <= 8:
                # Strategy: Direct 1-to-1 mapping if task count is small.
                for index in range(len(self.device.scripts)):
                    self.threads[index].script = self.device.scripts[index]
                    self.threads[index].neighbours = neighbours
            else:
                # Strategy: Divide scripts into 8 chunks.
                # Note: Original logic had a bug where only the last script of each chunk was retained.
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


            # Sync Stage 1: Trigger workers to start processing assigned scripts.
            self.barrier_parent.wait()

            # Block Logic: Ensures completion of script arrival before final synchronization.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Sync Stage 2: Wait for all local workers to finish current timepoint tasks.
            self.barrier_parent.wait()
            
            # Sync Stage 3: Align all devices across the cluster.
            self.device.global_barrier.wait()

        # Termination Phase: Shutdown all local workers.
        for thread in self.threads:
            thread.out = 1
        self.barrier_parent.wait()

        for thread in self.threads:
            thread.join()


class ReusableBarrier(object):
    """
    @brief Implementation of a two-phase synchronization barrier using semaphores.
    * Algorithm: Dual-stage arrival pattern to ensure strict thread alignment.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the barrier with target count and phase primitives.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Intent: Shared mutable counter.
        self.count_threads2 = [self.num_threads] # Intent: Shared mutable counter.
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Synchronizes the calling thread through both stages of the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single synchronization stage.
        Invariant: The last thread to arrive releases the entire group.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Worker(Thread):
    """
    @brief Persistent worker thread implementing the computational component of the simulation.
    """

    def __init__(self, device, script, neighbours, barrier_parent):
        """
        @brief Initializes the worker with its device context and local barrier.
        """
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.neighbours = neighbours
        self.workers_barrier = barrier_parent
        self.out = 0 # Intent: Exit signal from coordinator.

    def run(self):
        """
        @brief main loop for the persistent worker thread.
        Algorithm: Iterative execution triggered by the local coordinator barrier.
        """
        while True:
            # Block Logic: Waits for coordinator to assign tasks and signal start.
            self.workers_barrier.wait()

            if self.out == 1:
                # Logic: Shutdown signal.
                break

            if self.neighbours != None:
                script_data = []

                # Pre-condition: Acquire shared global location lock for atomic distributed update.
                self.device.locks[self.script[1]].acquire()
                
                # Distributed Aggregation: Collect readings from neighbors and self.
                for device in self.neighbours:
                    data = device.get_data(self.script[1])
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(self.script[1])
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Execution and Propagation Phase.
                    result = self.script[0].run(script_data)
                    for device in self.neighbours:
                        device.set_data(self.script[1], result)
                    self.device.set_data(self.script[1], result)
                
                # Post-condition: Release global location lock.
                self.device.locks[self.script[1]].release()
            
            # Sync Logic: Signals completion to coordinator.
            self.workers_barrier.wait()
