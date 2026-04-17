"""
@e6582f0d-403a-470e-900a-69c2d0765b1b/device.py
@brief Distributed sensor processing simulation using a persistent worker pool and shared global location locks.
* Algorithm: Producer-consumer task distribution via `Queue` with persistent worker threads and condition-variable barriers.
* Functional Utility: Orchestrates simulation timepoints across a network of devices by managing a local pool of worker threads that perform distributed data aggregation and propagation under fine-grained location locks.
"""

from threading import Event, Thread, Lock
from Queue import Queue
from reusable_barrier_condition import ReusableBarrier


class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings, coordination state, and internal worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and bootstraps the main coordination thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event() # Intent: Signals completion of task delivery for the current phase.
        self.thread = DeviceThread(self)
        self.location_locks = {}      # Intent: Map of shared locks for specific sensor locations.
        self.barrier = None
        self.num_threads = 8 # Domain: Concurrency Scaling - fixed worker pool size.
        self.queue = Queue(self.num_threads) # Intent: Task queue for the persistent worker threads.
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global synchronization setup and resource distribution.
        Invariant: Root node (or the first to arrive) initializes the shared barrier and global location locks.
        """
        if self.barrier is None:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = self.barrier
                # Logic: Discovers and initializes unique locks for all sensor locations in the cluster.
                for location in device.sensor_data:
                    if location not in self.location_locks:
                        self.location_locks[location] = Lock()
            
            # Logic: Distributes the shared lock registry to all participating devices.
            for device in devices:
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing task for the current simulation cycle.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Signals completion of script arrival for this timepoint.
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

class WorkerThread(Thread):
    """
    @brief Persistent worker thread that executes sensor scripts from the shared device queue.
    """

    def __init__(self, queue, device):
        """
        @brief Initializes the worker with its task queue and device context.
        """
        Thread.__init__(self)
        self.queue = queue
        self.device = device

    def run(self):
        """
        @brief main loop for the persistent worker thread.
        Algorithm: Producer-Consumer consumption; terminates on receiving (None, None, None).
        """
        while True:
            data_tuple = self.queue.get()

            # Logic: Poison pill handling for graceful thread termination.
            if data_tuple == (None, None, None):
                break

            # Pre-condition: Acquire shared global location lock for atomic distributed update.
            self.device.location_locks[data_tuple[1]].acquire()
            script_data = []
            
            # Distributed Aggregation Phase: Collect readings from neighbors and self.
            for device in data_tuple[2]:
                data = device.get_data(data_tuple[1])
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(data_tuple[1])
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Execution and Propagation Phase.
                result = data_tuple[0].run(script_data)

                for device in data_tuple[2]:
                    device.set_data(data_tuple[1], result)
                
                self.device.set_data(data_tuple[1], result)
            
            # Post-condition: Release global location lock.
            self.device.location_locks[data_tuple[1]].release()


class DeviceThread(Thread):
    """
    @brief Main management thread coordinating temporal phases and worker pool activity.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator for a specific device node.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core lifecycle loop of the device node coordination.
        Algorithm: Phased execution loop with event-driven task submission and barrier alignment.
        """
        threads = []

        # Logic: Spawns the persistent worker workforce.
        for i in range(self.device.num_threads):
            thread = WorkerThread(self.device.queue, self.device)
            threads.append(thread)
            threads[i].start()

        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Logic: Shutdown signal.
                break

            # Block Logic: Ensures all script assignments for the timepoint are received.
            self.device.timepoint_done.wait()

            # Dispatch Phase: Submits currently assigned scripts to the local worker pool.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location, neighbours))

            # Synchronization Phase: Align all devices across the cluster before the next timepoint.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()

        # Termination Phase: Shutdown local worker threads.
        for i in range(self.device.num_threads):
            self.device.queue.put((None, None, None))

        for i in range(self.device.num_threads):
            threads[i].join()


from threading import Condition

class ReusableBarrier(object):
    """
    @brief Implementation of a reusable synchronization barrier using condition variables.
    * Algorithm: Monitor pattern with auto-reset functionality.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()


    def wait(self):
        """
        @brief Blocks calling thread until the barrier threshold is met.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Logic: Final thread resets count and notifies all waiting peers.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()
