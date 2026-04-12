"""
@file device.py
@brief Distributed sensor unit simulation with queue-driven parallel processing and cyclic synchronization.
@details Implements a peer-to-peer network of sensing units that perform synchronized 
data aggregation. Uses a master coordination thread and a pool of worker threads 
synchronized via a shared work queue and a global cyclic barrier.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
from Queue import Queue

class Device(object):
    """
    @brief Controller for an autonomous sensing entity in a distributed cluster.
    Functional Utility: Manages local data buffers, organizes a worker pool for task 
    execution, and shares synchronization primitives with network peers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of local sensor readings.
        @param supervisor Entity providing topological neighbor discovery services.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        
        # Task Distribution: Synchronized queue for worker pool consumption.
        self.queue = Queue()
        self.worker_threads = []
        self.worker_threads_no = 8
        self.timepoint_done = Event()

        # Architecture: Spawns persistent worker threads.
        for _ in range(0, self.worker_threads_no):
            worker = WorkerThread(self, self.queue)
            worker.start()
            self.worker_threads.append(worker)

        # Resource Initialization: Designates device 0 as the cluster coordinator.
        if device_id == 0:
            devices_no = len(supervisor.supervisor.testcase.devices)
            self.barrier = ReusableBarrierCond(devices_no)
            self.dict_location_lock = {}
        else:
            # Sync: Peer devices initialize these references during setup_devices().
            self.barrier = None
            self.dict_location_lock = None

        self.all_devs = None

        # Lifecycle: Spawns the main device orchestrator thread.
        self.master_thread = DeviceThread(self)
        self.master_thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization of shared synchronization resources.
        Logic: Propagation of shared barrier and lock registry from the coordinator (device 0).
        """
        if self.device_id != 0:
            for dev in devices:
                if dev.device_id == 0:
                    self.barrier = dev.barrier
                    self.dict_location_lock = dev.dict_location_lock
                    break

        self.all_devs = devices

    def assign_script(self, script, location):
        """
        @brief Schedules a processing task for the current unit of time.
        Logic: Ensures a shared lock exists for the target location.
        """
        if script is not None:
            if location not in self.dict_location_lock:
                self.dict_location_lock[location] = Lock()
            self.scripts.append((script, location))
        else:
            # Protocol: Signals end of the script submission phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieval of local sensor readings.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        @brief Update of local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device lifecycle threads.
        """
        for worker in self.worker_threads:
            worker.join()
        self.master_thread.join()


class WorkerThread(Thread):
    """
    @brief Reusable execution unit that processes tasks from the shared queue.
    Architecture: Implements a distributed Map-Reduce operation.
    """

    def __init__(self, device, queue):
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device
        self.queue = queue

    def run(self):
        """
        @brief Continuous consumer loop for the device task queue.
        """
        while True:
            # Acquisition: Blocks until a task or shutdown signal (neighbours is None) is received.
            (scr_loc, neighbours) = self.queue.get()
            if neighbours is None:
                return

            (script, location) = scr_loc
            script_data = []

            /**
             * Block Logic: Critical section for distributed state aggregation.
             * Invariant: Uses the cluster-wide lock for the specific sensor location.
             */
            with self.device.dict_location_lock[location]:
                # Map Phase: Collects readings from neighborhood peers.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Local Data Collection.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Computation Phase.
                    result = script.run(script_data)
                    
                    # Reduce Phase: Propagates results back to participants.
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

            # Signal task completion to the queue manager.
            self.queue.task_done()


class DeviceThread(Thread):
    """
    @brief Orchestration thread managing timepoint cycles and task injection.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main coordination loop: discovery -> synchronization -> execution -> wait.
        """
        while True:
            # Discovery: Fetches topological neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            
            if neighbours is None:
                # Termination: Sends 'poison pills' to all workers in the pool.
                for _ in range(0, self.device.worker_threads_no):
                    self.device.queue.put((None, None))
                break

            # Sync: Wait for the local script assignment phase to conclude.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Task Submission: Enqueues all assigned scripts for parallel execution.
            for src_loc in self.device.scripts:
                self.device.queue.put((src_loc, neighbours))

            # Synchronization: Wait for all local workers to finalize tasks for the timepoint.
            self.device.queue.join()
            
            # Global Sync: Ensure all devices align at the cluster barrier.
            self.device.barrier.wait()
