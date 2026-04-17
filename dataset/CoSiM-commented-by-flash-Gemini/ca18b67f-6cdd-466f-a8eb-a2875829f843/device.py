"""
@ca18b67f-6cdd-466f-a8eb-a2875829f843/device.py
@brief Distributed sensor processing simulation using location-aware worker threads and global barriers.
* Algorithm: Greedy task scheduling that prioritizes location affinity to minimize synchronization overhead, followed by multi-phase barrier coordination.
* Functional Utility: Orchestrates simulation timepoints across a network of devices by managing a pool of worker threads that perform distributed data aggregation and updates.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue

class ReusableBarrier(object):
    """
    @brief Two-phase synchronization barrier implementation using counting semaphores.
    * Algorithm: Dual-stage arrival/release logic to prevent thread overruns between consecutive simulation steps.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the barrier with a target thread threshold.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Intent: Mutable shared counter for phase 1.
        self.count_threads2 = [self.num_threads] # Intent: Mutable shared counter for phase 2.
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread through both stages of the barrier.
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
                for i in range(self.num_threads):
                    threads_sem.release()
                
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    @brief Encapsulates a sensor node with its local data and management thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device state and prepares the main control thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.result_queue = Queue.Queue()
        self.set_lock = Lock() # Intent: Protects local data updates from concurrent worker threads.
        self.neighbours_lock = None
        self.neighbours_barrier = None

        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization and distribution of shared synchronization primitives.
        Invariant: The first device in the list acts as the master coordinator.
        """
        if self.device_id == devices[0].device_id:
            self.neighbours_lock = Lock()
            self.neighbours_barrier = ReusableBarrier(len(devices))
        else:
            self.neighbours_lock = devices[0].neighbours_lock
            self.neighbours_barrier = devices[0].neighbours_barrier

        self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Appends a task to the local queue or signals end-of-timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Logic: All scripts for the phase have been received.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Standard data retrieval for local sensor readings.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Synchronized update for local sensor readings.
        """
        self.set_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.set_lock.release()

    def shutdown(self):
        """
        @brief Terminates the device's management thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Coordination thread that dispatches scripts to a local pool of worker threads.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator thread.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []

    def run(self):
        """
        @brief Main coordination loop for simulation phases.
        Algorithm: Location-based task partitioning followed by parallel execution.
        """
        while True:
            # Logic: Thread-safe neighbor discovery.
            self.device.neighbours_lock.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.neighbours_lock.release()

            if neighbours is None:
                # Logic: Shutdown signal.
                break

            # Block Logic: Waits for script delivery start.
            self.device.script_received.wait()

            # Dispatch Phase: Spawns exactly 8 workers and distributes tasks using a greedy affinity strategy.
            self.workers = []
            for i in range(8):
                self.workers.append(DeviceWorker(self.device, i, neighbours))

            for (script, location) in self.device.scripts:
                # Logic: Task Assignment Strategy.
                # 1. Attempt to group scripts by sensor location to leverage cache/lock locality.
                added = False
                for worker in self.workers:
                    if location in worker.locations:
                        worker.add_script(script, location)
                        added = True

                # 2. Fallback: Assign to the least-burdened worker to balance load.
                if added == False:
                    minimum = len(self.workers[0].locations)
                    chosen_worker = self.workers[0]
                    for worker in self.workers:
                        if minimum > len(worker.locations):
                            minimum = len(worker.locations)
                            chosen_worker = worker

                    chosen_worker.add_script(script, location)

            # Execution Phase: Start all local workers.
            for worker in self.workers:
                worker.start()

            # Logic: Block until all local workers complete the current timepoint tasks.
            for worker in self.workers:
                worker.join()

            # Synchronization Phase: Align with all devices across the cluster.
            self.device.neighbours_barrier.wait()
            self.device.script_received.clear()


class DeviceWorker(Thread):
    """
    @brief Worker thread implementing the execution of partitioned sensor scripts.
    """

    def __init__(self, device, worker_id, neighbours):
        """
        @brief Initializes worker with task lists and neighbor context.
        """
        Thread.__init__(self)
        self.device = device
        self.worker_id = worker_id
        self.scripts = []
        self.locations = []
        self.neighbours = neighbours

    def add_script(self, script, location):
        """
        @brief Appends a script/location pair to the worker's private task list.
        """
        self.scripts.append(script)
        self.locations.append(location)

    def run_scripts(self):
        """
        @brief Main execution logic for the worker's assigned tasks.
        Algorithm: Iterative data aggregation from peers followed by local state update.
        """
        for (script, location) in zip(self.scripts, self.locations):
            script_data = []
            
            # Distributed Aggregation Phase: Collect readings from neighbors.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Logic: Include local data.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Execution and Propagation: Processes data and updates state across the neighborhood.
            if script_data != []:
                res = script.run(script_data)

                for device in self.neighbours:
                    device.set_data(location, res)
                self.device.set_data(location, res)

    def run(self):
        """
        @brief Entry point for the worker thread execution.
        """
        self.run_scripts()
