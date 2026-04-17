"""
@da2ef9f9-56a9-4257-bd49-dcc942094b68/device.py
@brief Distributed sensor processing simulation using a persistent thread pool and tiered barrier synchronization.
* Algorithm: Dynamic task offloading to a pool of 8 persistent worker threads via a thread-safe management queue, with multi-stage local and global barriers.
* Functional Utility: Orchestrates simulation timepoints across multiple devices by managing concurrent script execution, ensuring consistent data updates through fine-grained location locks and coarse-grained device locks.
"""

from threading import Event, Thread, Lock, Condition
from barrier import ReusableBarrierCond
from Queue import Queue

# Domain: Resource Scaling - Defines the persistent worker pool size per device.
THREAD_NR = 8

class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings, coordination state, and internal worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and bootstraps the persistent worker thread pool.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.setup_finished = Event() # Intent: Signals completion of shared resource distribution.
        self.dataLock = Lock()         # Intent: Serializes global updates to this device's sensor readings.
        self.shared_lock = Lock()      # Intent: Synchronizes access to shared device state.
        
        self.thread_queue = Queue(0)   # Intent: Management queue for available worker threads.
        # Intent: Internal device barrier for aligning local workers during neighbor discovery.
        self.wait_get_neighbours = ReusableBarrierCond(THREAD_NR)
        
        self.thread_pool = []
        self.neighbours = []

        # Logic: Spawns the persistent execution workforce and registers them in the management queue.
        for i in range(0, THREAD_NR):
            thread = DeviceThread(self, i)
            self.thread_pool.append(thread)
            self.thread_queue.put(thread)
            thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization and distribution of shared simulation resources.
        Invariant: Root node (ID 0) initializes the cluster-wide barrier and a shared location-lock map.
        """
        if self.device_id == 0:
            # Logic: Cluster-wide barrier shared across ALL threads of ALL devices.
            self.reusable_barrier = ReusableBarrierCond(len(devices) * THREAD_NR)
            self.location_locks = {}
            for device in devices:
                if device.device_id != self.device_id:
                    device.set_location_locks(self.location_locks)
                    device.set_barrier(self.reusable_barrier)

            self.setup_finished.set()

    def set_barrier(self, reusable_barrier):
        """
        @brief Links the device to the shared cluster synchronization point.
        """
        self.reusable_barrier = reusable_barrier
        self.setup_finished.set()

    def set_location_locks(self, location_locks):
        """
        @brief Assigns the shared global lock set for location-level synchronization.
        """
        self.location_locks = location_locks

    def assign_script(self, script, location):
        """
        @brief Top-level interface for script arrival.
        Algorithm: Dynamic task dispatch to the next available worker thread from the pool.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Logic: Lazy lock initialization for the target location.
            if location not in self.location_locks:
                self.location_locks[location] = Lock()

            # Dispatch Phase: Pops an idle worker and assigns the task.
            thread = self.thread_queue.get()
            thread.give_script(script, location)
        else:
            # Termination or Phase-End Logic: Injects "poison pills" for workers to signal end-of-batch.
            for (s, l) in self.scripts:
                thread = self.thread_queue.get()
                thread.give_script(s, l)

            for thread in self.thread_pool:
                thread.give_script(None, None)

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
        @brief Gracefully terminates the device's persistent worker threads.
        """
        for i in range(THREAD_NR):
            self.thread_pool[i].join()


class DeviceThread(Thread):
    """
    @brief Persistent worker thread that consumes tasks from a private script queue.
    """

    def __init__(self, device, ID):
        """
        @brief Initializes the worker with its device context and local sequence ID.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id = ID # Intent: Used to designate a coordinator (ID 0) for neighbor discovery.
        self.script_queue = Queue(0)

    def give_script(self, script, location):
        """
        @brief Enqueues a single task unit for this worker.
        """
        self.script_queue.put((script, location))

    def run(self):
        """
        @brief main loop for the persistent worker thread.
        Algorithm: Multi-stage synchronization lifecycle (Setup -> Discovery -> Execution -> Barrier).
        """
        while True:
            # Logic: Ensures global resource distribution is complete.
            self.device.setup_finished.wait()

            # Discovery Phase: Role-based neighborhood refresh.
            if self.id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # Internal Synchronization: Ensure all local threads have seen the new neighborhood.
            self.device.wait_get_neighbours.wait()

            if self.device.neighbours is None:
                # Logic: Shutdown signal received.
                break

            while True:
                # Task Consumption Phase.
                (script, location) = self.script_queue.get()

                if script is None:
                    # Logic: Current timepoint batch processed.
                    break

                # Distributed Aggregation: Accumulate data from neighbors under location and device locks.
                self.device.location_locks[location].acquire()
                script_data = []

                for device in self.device.neighbours:
                    device.dataLock.acquire()
                    data = device.get_data(location)
                    device.dataLock.release()

                    if data is not None:
                        script_data.append(data)
                
                self.device.dataLock.acquire()
                data = self.device.get_data(location)
                self.device.dataLock.release()
                
                if data is not None:
                   script_data.append(data)

                self.device.location_locks[location].release()

                if script_data != []:
                    # Execution Phase: Computes new state.
                    result = script.run(script_data)
                    
                    # Propagation Phase: Broadcasts results to the neighborhood under multi-lock coordination.
                    self.device.location_locks[location].acquire()
                    for device in self.device.neighbours:
                        device.dataLock.acquire()
                        device.set_data(location, result)
                        device.dataLock.release()

                    self.device.dataLock.acquire()
                    self.device.set_data(location, result)
                    self.device.dataLock.release()
                    self.device.location_locks[location].release()

                # Post-condition: Returns self to the pool of available workers.
                self.device.thread_queue.put(self)

            # Synchronization Phase: Align all devices and threads globally before the next timepoint.
            self.device.reusable_barrier.wait()
