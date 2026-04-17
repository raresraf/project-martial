"""
@e0289284-1dd2-4a36-9748-9aa340f4df30/device.py
@brief Distributed sensor processing simulation using a persistent thread pool and on-demand shared locking.
* Algorithm: Producer-consumer task execution via a persistent 8-thread pool with dual-phase semaphore barriers and lazy global lock distribution.
* Functional Utility: Orchestrates simulation timepoints by managing concurrent script execution across a cluster, ensuring atomic state updates through fine-grained location locks shared between nodes.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

class ThreadPool(object):
    """
    @brief Manager for a pool of persistent worker threads that execute sensor scripts.
    """
    
    def __init__(self, thread_number, device):
        """
        @brief Initializes the pool and bootstraps the worker threads.
        """
        self.thread_number = thread_number
        self.queue = Queue(self.thread_number)
        self.threads = []
        self.device = device

        # Logic: Spawns exactly thread_number persistent workers.
        for _ in xrange(thread_number):
            self.threads.append(Thread(target=self.execute))

        for thread in self.threads:
            thread.start()

    def execute(self):
        """
        @brief Main execution loop for each worker thread in the pool.
        Algorithm: Iterative task consumption from the shared queue; terminates on receiving (None, None, None).
        """
        neighbours, script, location = self.queue.get()

        while neighbours is not None \
              and script is not None \
              and location is not None:

            self.run(neighbours, script, location)
            self.queue.task_done()
            neighbours, script, location = self.queue.get()

        self.queue.task_done()

    def run(self, neighbours, script, location):
        """
        @brief Executes a single script unit with synchronized data access.
        Logic: Aggregates neighbor data and propagates results under a location-specific lock.
        """
        script_data = []
        # Pre-condition: Must acquire shared global location lock for cluster-wide consistency.
        self.device.location_lock[location].acquire()
        
        # Distributed Aggregation Phase: Collect readings from neighborhood peers.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Logic: Include local data in the processing batch.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Execution and Propagation Phase.
            result = script.run(script_data)
            for device in neighbours:
                device.set_data(location, result)
            self.device.set_data(location, result)

        # Post-condition: Release global location lock.
        self.device.location_lock[location].release()

    def submit(self, neighbours, script, location):
        """
        @brief Enqueues a new script task for the pool workers.
        """
        self.queue.put((neighbours, script, location))

    def wait(self):
        """
        @brief Blocks until all enqueued tasks in the pool are finished.
        """
        self.queue.join()

    def end(self):
        """
        @brief Orchestrates a clean shutdown of all worker threads in the pool.
        """
        self.wait()
        # Logic: Dispatches one termination signal per worker thread.
        for _ in xrange(self.thread_number):
            self.submit(None, None, None)

        for thread in self.threads:
            thread.join()


class Barrier(object):
    """
    @brief Implementation of a two-phase synchronization barrier using semaphores.
    * Algorithm: Dual-stage arrival/release logic to prevent thread overruns between phases.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the barrier state with target count and phase primitives.
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
        Invariant: The last thread to arrive releases the entire group and resets the counter.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    @brief Represents a sensor node that manages its local readings and coordinates worker pool activity.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and prepares the main coordination thread.
        """
        self.location_lock = [None] * 100 # Intent: Registry for shared locks indexed by sensor location ID.
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.all_devices = None
        self.recived_flag = False # Intent: Coordination signal for the start of a simulation phase.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global synchronization setup and node registration.
        Invariant: Root node (or the first to reach this point) initializes the collective barrier.
        """
        self.all_devices = devices
        if self.barrier is None:
            self.barrier = Barrier(len(devices))
            for device in devices:
                device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Buffers an incoming processing task and handles on-demand shared lock distribution.
        """
        if script is not None:
            # Logic: Lazy, on-demand lock initialization for new sensor locations.
            if self.location_lock[location] is None:
                new_lock = Lock()
                self.location_lock[location] = new_lock
                self.recived_flag = True

                # Invariant: Propagates the new lock to all participating devices for cluster-wide consistency.
                for device_number in xrange(len(self.all_devices)):
                    self.all_devices[device_number].location_lock[location] = new_lock

            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Logic: Signals completion of the assignment phase for this timepoint.
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
    @brief Main management thread coordinating timepoint progression and task offloading.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator and its local worker pool (8 threads).
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(8, self.device)

    def run(self):
        """
        @brief Core lifecycle loop of the device node.
        Algorithm: Phased execution loop with event-driven task submission and barrier alignment.
        """
        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Internal phase loop to handle multi-stage script arrival.
            while True:
                self.device.timepoint_done.wait()

                # Logic: Uses recived_flag to distinguish between task arrival and phase completion.
                if self.device.recived_flag:
                    # Dispatch Phase: Submits currently assigned scripts to the worker pool.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit(neighbours, script, location)
                    self.device.recived_flag = False
                else:
                    # Logic: End of timepoint phase reached.
                    self.device.timepoint_done.clear()
                    self.device.recived_flag = True
                    break

            # Synchronization Phase: Wait for all local workers to finish.
            self.thread_pool.wait()

            # Global Synchronization Phase: Align all devices across the cluster.
            self.device.barrier.wait()

        # Termination Phase: Shutdown worker threads.
        self.thread_pool.end()
