"""
@ea4cd275-0107-4996-8cab-8c7676f0a9ea/device.py
@brief Distributed sensor network simulation with persistent worker pool and sticky locking.
This module implements a highly parallel processing framework using a pool of 16 
persistent worker threads. It utilizes a stateful 'sticky lock' protocol where 
data acquisition and update operations are coupled to ensure atomic access to shared 
sensor locations across the neighborhood. Simulation phases are coordinated via 
a monitor-based barrier and a dual-queue task distribution system.

Domain: High-Parallelism worker pools, Stateful Locking, Distributed Coordination.
"""

import threading
from threading import Thread
from Queue import Queue
from cond_barrier import ReusableBarrier


class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local data state and provides a transactional 
    locking interface for spatial mutual exclusion.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        # Pre-allocate locks for local sensor data consistency.
        self.location_locks = {loc: threading.Lock() for loc in self.sensor_data}
        self.supervisor = supervisor
        self.scripts = []

        # Primary orchestration thread.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Input buffer for scripts assigned by the supervisor.
        self.scripts_queue = Queue()
        # Internal task distribution queue for workers.
        self.workers_queue = Queue()

        self.barrier = None


    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node (ID 0) initializes the shared barrier for the group.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = barrier

    def assign_script(self, script, location):
        """Registers a computational task into the node's input buffer."""
        self.scripts_queue.put((script, location))

    def get_data(self, location):
        """Safe retrieval of local sensor data (unsynchronized)."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def get_data_synchronize(self, location):
        """
        Stateful Data Retrieval.
        Functional Utility: Part 1 of a transactional update. Atomically 
        acquires the spatial lock for the location.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """Updates local sensor state (unsynchronized)."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def set_data_synchronize(self, location, data):
        """
        Stateful Data Update.
        Functional Utility: Part 2 of a transactional update. Replaces sensor 
        value and releases the spatial lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()

    def shutdown(self):
        """Gracefully joins the node's orchestration thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Main node manager.
    Functional Utility: Manages a pool of 16 worker threads and orchestrates 
    the flow of tasks from the supervisor to the execution layer.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.stop = False

    def run(self):
        """
        Main execution loop for the node manager.
        Algorithm: Iterative sequence of topology discovery, task dispatch 
        via workers_queue, and consensus.
        """
        num_workers = 16
        workers = []
        workers_queue = Queue()

        # Spawns a pool of high-concurrency worker threads.
        for i in range(num_workers):
            workers.append(WorkerThread(self.device, i, workers_queue))
        for worker in workers:
            worker.start()

        while True:
            # Refresh topology.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Filtering: identify peers excluding self for neighborhood operations.
            neighbours = [x for x in neighbours if x != self.device]
            for worker in workers:
                worker.neighbours = neighbours

            # Re-enqueue persistent scripts from previous timepoints.
            for script in self.device.scripts:
                workers_queue.put(script)

            # Block Logic: Dynamic task acquisition from input buffer.
            while True:
                script, location = self.device.scripts_queue.get()
                if script is None:
                    # 'Poison pill' indicates end of step workload.
                    break
                
                # Cache script for next timepoint and dispatch to worker pool.
                self.device.scripts.append((script, location))
                workers_queue.put((script, location))

            # Synchronize: wait for all parallel workers to complete current workload.
            workers_queue.join()
            
            # Global Consensus Point.
            self.device.barrier.wait()

        # Shutdown sequence for the worker pool.
        for worker in workers:
            workers_queue.put((None, None))
        for worker in workers:
            worker.join()


class WorkerThread(Thread):
    """
    Persistent worker thread implementation.
    Functional Utility: Consumes individual script tasks and implements the 
    core processing logic using sticky locks.
    """

    def __init__(self, device, worker_id, queue):
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.device = device
        self.neighbours = []
        self.worker_id = worker_id
        self.queue = queue

    def run(self):
        """
        Worker execution loop.
        Logic: Continuous 'pull-process' cycle from the shared device queue.
        """
        while True:
            # Block on task arrival.
            script, location = self.queue.get()
            # Termination check.
            if script is None:
                self.queue.task_done()
                break

            script_data = []
            # Aggregate neighborhood state using stateful locking protocol.
            for device in self.neighbours:
                # Part 1: Acquire spatial lock via get_data_synchronize.
                data = device.get_data_synchronize(location)
                if data is not None:
                    script_data.append(data)
            
            # Include local node state.
            data = self.device.get_data_synchronize(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Apply computational logic.
                result = script.run(script_data)

                # Part 2: Propagation and lock release via set_data_synchronize.
                for device in self.neighbours:
                    device.set_data_synchronize(location, result)
                
                self.device.set_data_synchronize(location, result)
            
            # Signal task completion to support queue.join().
            self.queue.task_done()
