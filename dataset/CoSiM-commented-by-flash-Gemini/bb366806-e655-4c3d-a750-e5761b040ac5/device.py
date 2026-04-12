"""
@file device.py
@brief Distributed sensor unit simulation with thread pool management and cyclic synchronization.
@details Implements a peer-to-peer network of sensing devices that perform synchronized data 
aggregation. Leverages a persistent ThreadPool for concurrent script execution and a two-phase 
semaphore-based barrier for coordinating execution units (timepoints) across the cluster.
"""

from threading import Event, Thread, Lock, Condition, Semaphore
from Queue import Queue

class ReusableBarrier():
    """
    @brief implementation of a two-phase cyclic barrier using semaphores.
    Functional Utility: Synchronizes a fixed group of threads across recurring cycles.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # State: Thread counters wrapped in lists for pass-by-reference mutable logic.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief executes the two-turnstile protocol (Arrival and Reset phases).
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
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Resets the counter for the next unit of time.
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Worker(Thread):
    """
    @brief persistent worker thread that processes tasks from a shared Queue.
    Architecture: Consumes (function, args, kargs) tuples and executes them sequentially.
    """
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.start()

    def run(self):
        """
        @brief consumer loop: blocks until work is available.
        Protocol: terminates if it receives None as the function.
        """
        while True:
            func, args, kargs = self.tasks.get()
            if func is None:
                self.tasks.task_done()
                break
            try:
                # Execution: Invokes the aggregation logic.
                func(*args, **kargs)
            except Exception as e:
                print(e)
            self.tasks.task_done()

class ThreadPool:
    """
    @brief Management abstraction for a collection of persistent worker threads.
    Functional Utility: Provides a centralized entry point for task submission and synchronization.
    """
    def __init__(self, num_threads):
        self.tasks = Queue()
        self.workers = [Worker(self.tasks) for _ in range(num_threads)]

    def add_task(self, func, *args, **kargs):
        """
        @brief Enqueues a new task for parallel execution.
        """
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """
        @brief Blocks until all tasks in the queue have been fully processed.
        """
        self.tasks.join()

    def terminate_workers(self):
        """
        @brief Signals all workers in the pool to shut down.
        """
        for _ in self.workers:
            self.tasks.put((None, None, None))

    def join_all(self):
        """
        @brief Blocks until all worker threads in the pool have exited.
        """
        for worker in self.workers:
            worker.join()

class Device(object):
    """
    @brief Controller for an autonomous sensing entity in a distributed cluster.
    Functional Utility: Manages local data state, coordinates task submission to 
    the local pool, and participates in cluster-wide synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of sensor location-value pairs.
        @param supervisor entity providing topology discovery.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Synchronization Primitives.
        self.script_received = Event()
        self.wait_neighbours = Event()
        self.scripts = []
        self.neighbours = []
        self.all_devices = []
        
        # Lock Registry: One lock per sensor location to ensure cluster-wide atomic access.
        self.locks = [Lock() for _ in range(50)]
        
        # Architecture: Persistent thread pool for concurrent script execution.
        self.pool = ThreadPool(8)
        
        # Lifecycle: Spawns the main device orchestrator thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collaborative initialization of shared synchronization resources.
        Logic: Designates device 0 as the allocator for the global cyclic barrier.
        """
        self.all_devices = devices
        self.barrier = ReusableBarrier(len(devices))

    def assign_script(self, script, location):
        """
        @brief Schedules a processing script for the current unit of time.
        Logic: Buffers scripts locally and also submits them to the worker pool.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.pool.add_task(self.execute_script, script, location)
        else:
            # Protocol: Signals end of the script submission phase.
            self.script_received.set()

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

    def execute_script(self, script, location):
        """
        @brief Logic for a single data aggregation task.
        Architecture: Implements a distributed Map-Reduce operation.
        """
        # Sync: Wait for the master thread to finalize neighbor discovery.
        self.wait_neighbours.wait()
        script_data = []

        if self.neighbours:
            for peer in self.neighbours:
                # Critical Section: Exclusive access to peer sensor state.
                peer.locks[location].acquire()
                data = peer.get_data(location)
                peer.locks[location].release()
                if data is not None:
                    script_data.append(data)

        # Inclusion: Local state.
        self.locks[location].acquire()
        data = self.get_data(location)
        self.locks[location].release()
        if data is not None:
            script_data.append(data)

        if script_data:
            # Computation: executes aggregation logic.
            result = script.run(script_data)

            # Reduce Phase: propagates result back to all participants.
            if self.neighbours:
                for peer in self.neighbours:
                    peer.locks[location].acquire()
                    peer.set_data(location, result)
                    peer.locks[location].release()

            self.locks[location].acquire()
            self.set_data(location, result)
            self.locks[location].release()


class DeviceThread(Thread):
    """
    @brief orchestrator thread managing discrete execution units (timepoints).
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main coordination loop: discovery -> batch submission -> synchronization.
        """
        while True:
            # State reset for the next timepoint.
            self.device.script_received.clear()
            self.device.wait_neighbours.clear()

            # Phase 1: Discovery.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.wait_neighbours.set()

            if self.device.neighbours is None:
                # Termination: shuts down the local worker pool and thread.
                self.device.pool.wait_completion()
                self.device.pool.terminate_workers()
                self.device.pool.join_all()
                break

            # Phase 2: Task Submission.
            # Logic: Enqueues all previously assigned scripts for the new timepoint.
            for (script, location) in self.device.scripts:
                self.device.pool.add_task(self.device.execute_script, script, location)

            # Sync: Blocks until both assignment phase ends and all workers finalize tasks.
            self.device.script_received.wait()
            self.device.pool.wait_completion()

            # Global Sync: align cluster devices at the barrier.
            for dev in self.device.all_devices:
                dev.barrier.wait()
