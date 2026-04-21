
"""
@file device.py
@brief Distributed device simulation framework with multi-level barrier synchronization.

Functional Intent: Implements a parallel execution environment for autonomous devices 
in a simulated distributed network. Features a custom reusable barrier for 
synchronizing both device-level and thread-level execution steps. Coordinates 
neighborhood data reduction and broadcast across multiple worker threads to ensure 
eventual consistency in discrete time-steps.

Domain: Production Systems, Distributed Computing, Parallel Processing.
"""


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    @brief Custom reusable synchronization barrier implemented using semaphores.
    
    Algorithm: Two-phase (turnstile) synchronization.
    Logic: Uses double-gating to ensure all participating threads arrive before 
    any are released, and all threads depart before the barrier is reset for 
    the next epoch.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the barrier with a fixed participant count.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads


        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until the collective rendezvous is reached.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Entrance phase: Accumulates arriving threads and releases the group.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Logic: The final thread to arrive triggers the group release.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Exit phase: Ensures all threads have cleared phase 1 before resetting state.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    """
    @brief Represents an independent computational node in the distributed simulation.
    
    Functional Utility: Manages device-local sensor state and coordinates task 
    distribution across a pool of parallel workers. Synchronizes with global 
    peers via a shared barrier and utilizes local locks for shared data consistency.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device and starts its management and worker threads.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.lock = {} # Logic: Shared lock pool for location-specific data protection.

        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.setup_done = Event()
        self.terminate = Event()
        self.neighbours = []

        self.barrier = None
        # Optimization: Internal barrier for local thread-pool synchronization.
        self.threads_barrier = ReusableBarrierSem(9)
        self.master = DeviceThread(self, self.terminate, self.barrier, self.threads_barrier, 
                                    self.setup_done)
        self.master.start()

        self.threads = []
        # Optimization: Fixed worker pool size to match common multi-core architectures.
        for _ in range(8):
            thread = Worker(self.master, self.terminate, self.threads_barrier)
            self.threads.append(thread)
            thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Orchestrates the cluster-wide setup of synchronization primitives.
        
        Algorithm: Designated master initialization (Device 0).
        Logic: The master device generates a pool of mutexes and a global barrier, 
        then propagates them to all participants to ensure a unified execution timeline.
        """
        # Pre-condition: Only the first device performs initialization to prevent state fragmentation.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                self.lock[dev] = Lock()
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier
                    dev.lock = self.lock
                    dev.setup_done.set()

            self.setup_done.set()

    def assign_script(self, script, location):
        """
        @brief Binds a transformation script to a specific data location for the current step.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Logic: Signaling None indicates the end of task assignment for the current epoch.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Thread-safe retrieval of sensor data.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Thread-safe update of local sensor state.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates orchestration and worker thread pools.
        """
        self.terminate.set()
        for i in range(8):
            self.threads[i].script_received.set()
            self.threads[i].join()
        self.master.join()


class DeviceThread(Thread):
    """
    @brief Orchestration thread responsible for task scheduling and cross-device synchronization.
    
    Logic: Manages the high-level workflow transitions between neighbor discovery, 
    local work distribution, and global consensus.
    """

    def __init__(self, device, terminate, barrier, threads_barrier, setup_done):
        """
        @brief Initializes the manager with parent context and local barriers.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = []
        self.terminate = terminate
        self.barrier = barrier
        self.threads_barrier = threads_barrier
        self.setup_done = setup_done

    def run(self):
        """
        @brief Core operational loop for the device manager.
        
        Algorithm: Step-wise distributed task execution.
        Logic: 
        1. Rendezvous at the global barrier before starting.
        2. Discovers neighborhood topology.
        3. Distributes tasks to worker threads using round-robin scheduling.
        4. Synchronizes with local workers before committing results.
        """
        self.setup_done.wait()
        self.device.barrier.wait()

        while True:
            # Synchronization: Global rendezvous ensures all devices are ready for the next step.
            self.device.barrier.wait()

            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                break

            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

            # Task Distribution: Prepares work batches for the parallel pool.
            scripts = []
            for i in range(8):
                scripts.append([])

            for i in range(len(self.device.scripts)):
                # Logic: Evenly partitions the script load across available CPU-bound workers.
                scripts[i%8].append(self.device.scripts[i])

            for i in range(8):
                self.device.threads[i].scripts = scripts[i]
                self.device.threads[i].script_received.set()

            # Synchronization: Blocks until all local worker threads reach the local barrier.
            if not self.terminate.is_set():
                self.threads_barrier.wait()


class Worker(Thread):
    """
    @brief Worker thread responsible for executing a subset of data-processing scripts.
    
    Algorithm: Neighborhood state aggregation and broadcast.
    """

    def __init__(self, master, terminate, barrier):
        Thread.__init__(self)
        self.master = master
        self.script_received = Event()
        self.terminate = terminate
        self.scripts = []
        self.barrier = barrier

    @staticmethod
    def append_data(device, location, script_data):
        """
        @brief Atomically gathers sensor data from a target device.
        """
        # Synchronization: Ensures consistent view of the peer's data plane.
        device.lock[device].acquire()
        data = device.get_data(location)
        device.lock[device].release()
        if data is not None:
            script_data.append(data)

    @staticmethod
    def set_data(device, location, result):
        """
        @brief Atomically propagates results back to a target device.
        """
        device.lock[device].acquire()
        device.set_data(location, result)
        device.lock[device].release()

    def run(self):
        """
        @brief Worker execution cycle.
        
        Logic: 
        1. Waits for tasks from the manager.
        2. Collects neighborhood data for each assigned script.
        3. Executes the transformation and broadcasts results to all neighbors.
        4. Reaches local barrier to signal batch completion.
        """
        while True:
            self.script_received.wait()
            self.script_received.clear()

            if self.terminate.is_set():
                break
                
            if self.scripts is not None:
                for (script, location) in self.scripts:
                    
                    script_data = []
                    if self.master.neighbours is not None:
                        # Block Logic: Distributed data gathering.
                        for device in self.master.neighbours:
                            self.append_data(device, location, script_data)

                    self.append_data(self.master.device, location, script_data)

                    if script_data != []:
                        # Functional Intent: Executes user-defined processing logic.
                        result = script.run(script_data)

                        if self.master.neighbours is not None:
                            # Block Logic: Distributed state broadcast.
                            for device in self.master.neighbours:
                                self.set_data(device, location, result)
                        
                        self.set_data(self.master.device, location, result)

            # Synchronization: Local rendezvous with the master thread.
            self.barrier.wait()
