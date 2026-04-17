"""
@dc9ff09f-5f73-42bd-831e-f563975905d2/device.py
@brief Distributed sensor processing simulation using a persistent worker pool and tiered synchronization.
* Algorithm: Atomic task consumption from a shared device queue by a pool of workers sized according to `cpu_count()`, with per-location semaphore locks and multi-stage barriers.
* Functional Utility: Orchestrates simulation timepoints across multiple devices by managing high-concurrency script execution and ensuring synchronized distributed state updates.
"""

from threading import Event, Thread, Lock, Semaphore
from multiprocessing import cpu_count
from Queue import Queue


class ReusableBarrierSem(object):
    """
    @brief Two-phase synchronization barrier implementation using counting semaphores.
    * Algorithm: Dual-stage arrival/release pattern to ensure consistent thread alignment across repeated simulation cycles.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the barrier state and its internal phase control primitives.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Synchronizes the calling thread through both stages of the barrier.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Stage 1: Collects all threads and releases them simultaneously.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Logic: Last thread arrival triggers the release of the entire group.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Stage 2: Secondary synchronization to prevent thread overruns in consecutive phases.
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
    @brief Encapsulates a sensor node with its local readings, coordination state, and internal worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and bootstraps the persistent worker pool.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event() # Intent: Signals completion of task delivery for the current phase.
        self.neighbours = []

        # Domain: Adaptive Concurrency - sizes the worker pool based on available hardware resources.
        self.num_of_threads = cpu_count()
        if self.num_of_threads < 8:
            self.num_of_threads = 8

        self.tasks = Queue() # Intent: Task queue for the local worker threads.
        self.semaphore = Semaphore(0) # Intent: Signals available tasks in the queue to worker threads.

        # Intent: Global pool of locks (as semaphores) protecting specific sensor locations.
        self.num_locations = self.supervisor.supervisor.testcase.num_locations
        self.lock_locations = []

        self.lock_queue = Lock() # Intent: Serializes access to the task queue.
        # Intent: Internal device barrier for aligning local workers.
        self.barrier = ReusableBarrierSem(self.num_of_threads)
        # Intent: Global barrier for cluster-wide coordination.
        self.global_barrier = ReusableBarrierSem(0)
        self.pool = Pool(self)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization and distribution of shared simulation resources.
        Invariant: Root node (ID 0) initializes the global barrier and location locks.
        """
        if self.device_id == 0:
            self.global_barrier = ReusableBarrierSem(len(devices))
            for _ in range(self.num_locations):
                # Logic: Uses semaphores as binary locks for sensor locations.
                self.lock_locations.append(Semaphore(1))
            for device in devices:
                device.global_barrier = self.global_barrier
                device.lock_locations = self.lock_locations

    def assign_script(self, script, location):
        """
        @brief Receives a processing task for the current simulation phase.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
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
        @brief Gracefully terminates the device's worker pool.
        """
        self.pool.shutdown()


class Pool(object):
    """
    @brief Manager for the persistent worker thread pool.
    """
    
    def __init__(self, device):
        """
        @brief Bootstraps the execution workforce.
        """
        self.device = device
        self.thread_list = []
        for i in range(self.device.num_of_threads):
            self.thread_list.append(DeviceThread(self.device, i))
        for thread in self.thread_list:
            thread.start()

    def add_task(self, task):
        """
        @brief Enqueues a task and increments the availability semaphore.
        """
        self.device.tasks.put(task)
        self.device.semaphore.release()

    def shutdown(self):
        """
        @brief Terminates all workers.
        """
        for thread in self.thread_list:
            thread.join()


class DeviceThread(Thread):
    """
    @brief Persistent worker thread implementing the computational component of simulation phases.
    """

    def __init__(self, device, thread_id):
        """
        @brief Initializes the worker with its device context and local sequence ID.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id # Intent: Designation of coordinator (ID 0).
        self.script = None
        self.location = None
        self.script_data = []
        self.data = None
        self.result = None

    def run(self):
        """
        @brief Core execution loop of the worker thread.
        Algorithm: Phased synchronization with role-based coordination and distributed aggregation.
        """
        while True:
            # Sync Phase 1: Collective alignment before state refresh.
            if self.thread_id == 0:
                self.device.timepoint_done.clear()
                # Logic: Designated coordinator thread refreshes neighbor set.
                self.device.neighbours = self.device.supervisor.get_neighbours()

            self.device.barrier.wait()
            if self.device.neighbours is None:
                # Logic: Shutdown signal received.
                break

            # Sync Phase 2: Task Dispatching.
            if self.thread_id == 0:
                # Block Logic: Ensures script batch is fully arrived.
                self.device.timepoint_done.wait()
                for (script, location) in self.device.scripts:
                    self.device.pool.add_task((script, location))

                # Logic: Release all local workers by signaling task availability.
                for _ in range(self.device.num_of_threads):
                    self.device.semaphore.release()

            while True:
                # Task Consumption Phase.
                self.device.semaphore.acquire()
                # Pre-condition: Atomic task removal from the shared queue.
                with self.device.lock_queue:
                    if not self.device.tasks.empty():
                        (self.script, self.location) = self.device.tasks.get()
                    else:
                        # Logic: Local task pool exhausted for this timepoint.
                        break

                # Distributed Aggregation Phase: Collect readings under location-specific mutual exclusion.
                self.device.lock_locations[self.location].acquire()
                self.script_data = []

                for device in self.device.neighbours:
                    self.data = device.get_data(self.location)
                    if self.data is not None:
                        self.script_data.append(self.data)
                
                self.data = self.device.get_data(self.location)
                if self.data is not None:
                    self.script_data.append(self.data)

                if self.script_data != []:
                    # Execution Phase: Computes new state.
                    self.result = self.script.run(self.script_data)
                    # Propagation Phase: Updates the entire neighborhood.
                    for device in self.device.neighbours:
                        device.set_data(self.location, self.result)
                    self.device.set_data(self.location, self.result)

                # Post-condition: Release global location lock.
                self.device.lock_locations[self.location].release()

            # Sync Phase 3: Global Cluster Alignment.
            if self.thread_id == 0:
                self.device.global_barrier.wait()

            # Sync Phase 4: Finalize local worker alignment before next timepoint.
            self.device.barrier.wait()
