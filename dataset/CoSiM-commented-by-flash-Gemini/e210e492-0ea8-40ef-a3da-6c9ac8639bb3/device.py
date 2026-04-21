
/**
 * @file device.py
 * @brief Concurrent simulation framework for distributed devices with centralized state orchestration.
 * 
 * Functional Intent: Provides a platform for simulating network nodes (Devices) 
 * that execute scripts across a shared data space. It employs a thread pool 
 * pattern to parallelize script execution within each device and a cyclic 
 * barrier for global synchronization between simulation time-steps. Shared 
 * state is protected via fine-grained per-location locking managed by a 
 * central coordinator.
 * 
 * Domain: Production Systems, Distributed Simulation, Concurrency, Thread Management.
 */

from threading import Thread, Lock, Condition, Semaphore, Event
from Queue import Queue

class Device(object):
    /**
     * @class Device
     * @brief High-level manager for a single simulation node.
     * 
     * Logic: Delegates script processing to a dedicated control thread 
     * (`DeviceThread`) and provides thread-safe ingestion of scripts 
     * via a queue.
     */

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        # Optimization: Fixed hardware core count for thread pool scaling.
        self.num_cores = 8  
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Invariant: Asynchronous ingestion of simulation tasks for the current step.
        self.new_scripts = Queue()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        /**
         * setup_devices - Bootstraps the global synchronization context.
         * 
         * Logic: Executed by a leader node (ID 0) to initialize the 
         * `SharedDeviceData` container and distribute it to all participants.
         */
        if self.device_id == 0:
            shared_data = SharedDeviceData(len(devices))
            
            # Synchronization: Pre-populates locks for known data locations 
            # to reduce runtime overhead during the first pass.
            for data in self.sensor_data:
                if data not in shared_data.location_locks:
                    shared_data.location_locks[data] = Lock()

            for dev in devices:
                dev.shared_data = shared_data

    def assign_script(self, script, location):
        /**
         * assign_script - Queues a task for the current simulation window.
         */
        self.new_scripts.put((script, location))

    def get_data(self, location):
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    /**
     * @class DeviceThread
     * @brief Orchestration loop for simulation time-steps and local task execution.
     * 
     * Logic: Manages a reusable `ThreadPool` to process scripts in parallel. 
     * It iteratively pulls tasks from the device queue until a sentinel (None) 
     * is reached, then synchronizes globally with all other device nodes.
     */

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        /**
         * Block Logic: Simulation lifecycle loop.
         * Invariant: Each iteration corresponds to one globally synchronized time-step.
         */
        thread_pool = ThreadPool(self.device.num_cores)
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Logic: Shutdown signal from supervisor.
                break

            # Block Logic: Re-execution of persistent scripts from previous cycles.
            for (script, location) in self.device.scripts:
                thread_pool.submit(RunScript(script, location, neighbours,
                                             self.device))
            
            # Block Logic: Processing of newly assigned scripts for this window.
            while True:
                (script, location) = self.device.new_scripts.get()
                if script is None: 
                    # Sentinel Logic: Marks the completion of task assignments for this step.
                    break

                # Synchronization: Ensures thread-safe registration of new data locations.
                self.device.shared_data.ll_lock.acquire()
                if location not in self.device.shared_data.location_locks:
                    self.device.shared_data.location_locks[location] = Lock()
                self.device.shared_data.ll_lock.release()

                thread_pool.submit(RunScript(script, location, neighbours,
                                             self.device))
                self.device.scripts.append((script, location))

            # Functional Utility: Flush and wait for local task completion before global barrier.
            thread_pool.shutdown() 
            thread_pool.wait_termination(False) 

            # Synchronization: Global phase barrier for simulation parity.
            self.device.shared_data.timepoint_barrier.wait()

        thread_pool.wait_termination() 

class RunScript(object):
    /**
     * @class RunScript
     * @brief Encapsulates the algorithmic logic for a single data-processing task.
     * 
     * Logic: Implements an aggregate-process-propagate pattern. It gaters 
     * neighborhood data, executes the transformation, and commits results 
     * using per-location mutexes to ensure atomicity.
     */

    def __init__(self, script, location, neighbours, device):
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        # Logic: Atomic lookup of the shared mutex for the specific data location.
        self.device.shared_data.ll_lock.acquire()
        lock = self.device.shared_data.location_locks[self.location]
        self.device.shared_data.ll_lock.release()

        script_data = []

        # Synchronization: Exclusive access to the data location across the entire cluster.
        lock.acquire()  

        # Block Logic: Distributed data aggregation.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Logic: Execution of the user-provided script logic.
            result = self.script.run(script_data)

            # Block Logic: Result propagation to the neighborhood.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)

        lock.release() 

class CyclicBarrier(object):
    /**
     * @class CyclicBarrier
     * @brief Multi-use thread synchronization point.
     * 
     * Algorithm: Conditional wait with automatic reset. 
     * Parties wait until the count reaches the threshold, then all are 
     * released and the counter is reset for the next simulation step.
     */

    def __init__(self, parties):
        self.parties = parties
        self.count = 0
        self.condition = Condition()

    def wait(self):
        self.condition.acquire()
        self.count += 1
        if self.count == self.parties:
            # Case: Final party has arrived; signal all waiting threads.
            self.condition.notifyAll() 
            self.count = 0  
        else:
            # Case: Threshold not met; block the thread.
            self.condition.wait()
        self.condition.release()

class ThreadPool(object):
    /**
     * @class ThreadPool
     * @brief Manager for a fixed set of persistent worker threads.
     * 
     * Logic: Uses a producer-consumer pattern via a synchronized task queue 
     * and semaphores to dispatch work to background threads.
     */

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.task_queue = Queue() 
        self.num_tasks = Semaphore(0)
        self.stop_signal = Event() 
        self.shutdown_signal = Event()

        self.threads = []
        for i in xrange(0, num_threads):
            self.threads.append(Worker(self.task_queue,
                                       self.num_tasks,
                                       self.stop_signal))
        
        for i in xrange(0, num_threads):
            self.threads[i].start()

    def submit(self, task):
        # Guard: Reject new work during pool teardown.
        if self.shutdown_signal.is_set():
            return 
        self.task_queue.put(task)
        self.num_tasks.release()

    def shutdown(self):
        self.shutdown_signal.set()

    def wait_termination(self, end=True):
        /**
         * wait_termination - Orchestrates pool draining and worker lifecycle.
         * 
         * Logic: Blocks until the task queue is empty. If 'end' is true, 
         * it poisons the queue with None values to terminate all workers.
         */
        self.task_queue.join()
        if end is True:
            self.stop_signal.set() 
            for i in xrange(0, self.num_threads):
                self.task_queue.put(None) 
                self.num_tasks.release()

            for i in xrange(0, self.num_threads):
                self.threads[i].join()
        else:
            self.shutdown_signal.clear()


class Worker(Thread):
    /**
     * @class Worker
     * @brief Persistent background thread for executing simulation tasks.
     */

    def __init__(self, task_queue, num_tasks, stop_signal):
        Thread.__init__(self)
        self.task_queue = task_queue
        self.num_tasks = num_tasks
        self.stop_signal = stop_signal

    def run(self):
        while True:
            # Synchronization: Blocks until a task is available or shutdown is triggered.
            self.num_tasks.acquire()
            if self.stop_signal.is_set():
                break
            
            task = self.task_queue.get()
            if task is None:
                break
            
            task.run()
            self.task_queue.task_done()

class SharedDeviceData(object):
    /**
     * @class SharedDeviceData
     * @brief Centralized repository for cluster-wide synchronization primitives.
     */

    def __init__(self, num_devices):
        self.num_devices = num_devices
        # Invariant: One barrier shared by all DeviceThreads for step synchronization.
        self.timepoint_barrier = CyclicBarrier(num_devices)
        
        # Invariant: Dynamically managed map of per-location mutexes.
        self.location_locks = {}

        # Synchronization: Mutex for the location_locks map itself.
        self.ll_lock = Lock()
