"""
@e0ddb65b-ea24-496c-918d-4c2f0c7ae701/device.py
@brief Distributed sensor processing simulation using a persistent thread pool and atomic cross-node state transitions.
* Algorithm: Producer-consumer task consumption via persistent worker threads with phased barrier synchronization and asymmetrical locking.
* Functional Utility: Orchestrates simulation timepoints across multiple devices by managing a local pool of 8 worker threads that perform distributed data aggregation and propagation using location-specific mutual exclusion.
"""

from threading import Event, Thread, Lock
from barrier import Barrier
from thread_pool import ThreadPool


class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings, internal worker pool, and shared synchronization state.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and bootstraps the coordination thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.barrier = None
        self.scripts = []
        self.timepoint_done = Event() # Intent: Signals completion of task delivery for the current phase.

        # Logic: Initializes a unique lock for every local sensor location.
        self.locks = {}
        for location in sensor_data:
            self.locks[location] = Lock()

        self.scripts_available = False # Intent: Phase coordination flag.

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global synchronization setup.
        Invariant: Root node (ID 0) initializes the cluster-wide barrier.
        """
        if self.device_id == 0:
            barrier = Barrier(len(devices))
            self.barrier = barrier
            self.send_barrier(devices, barrier)

    @staticmethod
    def send_barrier(devices, barrier):
        """
        @brief Distributes the shared barrier instance to all peer nodes.
        """
        for device in devices:
            if device.device_id != 0:
                device.set_barrier(barrier)

    def set_barrier(self, barrier):
        """
        @brief Links the device to the shared cluster synchronization point.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Receives a processing task for the current simulation phase.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Synchronized acquisition of sensor data.
        Pre-condition: Acquisition of location-specific lock ensures data consistency.
        Note: Lock is expected to be released by a subsequent set_data call for the same location.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]

        return None

    def set_data(self, location, data):
        """
        @brief Synchronized update of sensor data.
        Post-condition: Release of location-specific lock completes the atomic operation.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """
        @brief Terminates the device coordination thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief coordination thread managing temporal phases and worker pool task submission.
    """
    # Domain: Resource Scaling - fixed worker pool size.
    NR_THREADS = 8

    def __init__(self, device):
        """
        @brief Initializes the coordinator and its local worker pool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(self.NR_THREADS)

    def run(self):
        """
        @brief Main execution lifecycle for the coordinator thread.
        Algorithm: Phased execution loop with event-driven task submission and global barrier alignment.
        """
        self.thread_pool.set_device(self.device)

        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Internal phase loop to handle continuous task delivery.
            while True:
                self.device.timepoint_done.wait()
                if self.device.scripts_available:
                    self.device.scripts_available = False

                    # Dispatch Phase: Submits currently assigned scripts to the worker pool.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task((neighbours, location, script))
                else:
                    # Logic: End of current timepoint phase reached.
                    self.device.timepoint_done.clear()
                    self.device.scripts_available = True
                    break

            # Synchronization Phase: Wait for all local workers to finish.
            self.thread_pool.wait()

            # Global Synchronization Phase: Align all devices across the cluster.
            self.device.barrier.wait()

        # Termination Phase: Shutdown worker threads.
        self.thread_pool.finish()


from threading import Thread
from Queue import Queue


class ThreadPool(object):
    """
    @brief Thread pool implementation for parallelizing sensor script execution.
    """
    
    def __init__(self, nr_threads):
        """
        @brief Initializes the pool and bootstraps the persistent worker threads.
        """
        self.device = None
        self.queue = Queue(nr_threads)
        self.thread_list = []
        self.create_threads(nr_threads)
        self.start_threads()

    def create_threads(self, nr_threads):
        """
        @brief Instantiates worker threads targeting the execute_task loop.
        """
        for _ in xrange(nr_threads):
            thread = Thread(target=self.execute_task)
            self.thread_list.append(thread)

    def start_threads(self):
        """
        @brief Signals all workers to begin polling for tasks.
        """
        for i in xrange(len(self.thread_list)):
            self.thread_list[i].start()

    def set_device(self, device):
        """
        @brief Affiliates the pool with a specific device instance for data context.
        """
        self.device = device

    def submit_task(self, task):
        """
        @brief Enqueues a new script task for the pool workers.
        """
        self.queue.put(task)

    def execute_task(self):
        """
        @brief Main loop for each worker thread in the pool.
        Algorithm: Producer-Consumer consumption with termination handling.
        """
        while True:
            task = self.queue.get()
            neighbours = task[0]
            script = task[2]

            # Logic: Poison pill handling (None, None, None) for graceful shutdown.
            if script is None and neighbours is None:
                self.queue.task_done()
                break

            self.run_script(task)
            self.queue.task_done()

    def run_script(self, task):
        """
        @brief Orchestrates the execution of a single script unit.
        Logic: Aggregates data from neighborhood and updates shared state under location locks.
        """
        neighbours, location, script = task
        script_data = []

        # Distributed Data Aggregation: Collects readings from neighborhood nodes.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        # Logic: Includes local reading in the processing batch.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Execution and Propagation Phase.
            result = script.run(script_data)
            
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    # Logic: Updates peer state.
                    device.set_data(location, result)

            # Logic: Updates local state.
            self.device.set_data(location, result)

    def wait(self):
        """
        @brief Blocks until all enqueued tasks in the pool are finished.
        """
        self.queue.join()

    def finish(self):
        """
        @brief Orchestrates a clean shutdown of all worker threads in the pool.
        """
        self.wait()
        # Logic: Dispatches one termination signal per worker.
        for _ in xrange(len(self.thread_list)):
            self.submit_task((None, None, None))

        for i in xrange(len(self.thread_list)):
            self.thread_list[i].join()
