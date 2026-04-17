"""
@cea16033-9733-46d3-be52-7e5d2ffbf731/device.py
@brief Hierarchical distributed sensor processing simulation using a delegated thread pool and multi-phase barriers.
* Algorithm: Multi-tier worker orchestration (Control -> Scripter -> Executors) with location-level locking and two-phase semaphore synchronization.
* Functional Utility: Manages high-concurrency script execution across a device cluster, ensuring atomic data updates and synchronized timepoint transitions.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue


class Device(object):
    """
    @brief Encapsulates a sensor node with its local data, synchronization state, and management hierarchy.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device and prepares its internal thread-safe infrastructure.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None
        
        self.script_running = Lock() # Intent: Serializes access to active script execution state.
        self.timepoint_done = Event()
        
        self.data_locks = dict() # Intent: Maps sensor locations to their respective synchronization locks.
        
        self.queue = Queue() # Intent: Work queue for script execution tasks.
        
        # Domain: Resource Scaling - Defines the worker pool capacity (14 threads).
        self.available_threads = 14

        # Logic: Initializes a unique lock for every sensor location known to this device.
        for loc in sensor_data:
            self.data_locks.__setitem__(loc, Lock())

        self.can_get_data = Lock() # Intent: Coarse-grained lock for device metadata synchronization.

        self.master = None
        self.script_over = False
        self.alive = True
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global synchronization setup for the device cluster.
        Invariant: Establishes a shared barrier and designates the first device as master.
        """
        self.barrier = ReusableBarrier(len(devices))
        self.master = devices[0]

    def assign_script(self, script, location):
        """
        @brief Queues a script for execution and signals the arrival of work to the Scripter.
        """
        if script is not None:
            self.script_running.acquire()
            self.scripts.append((script, location))
            self.queue.put_nowait((script, location))
            self.script_received.set()
        else:
            # Logic: Signals that all scripts for the current phase have been assigned.
            self.script_running.acquire()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Coarse-grained data retrieval interface.
        """
        self.can_get_data.acquire()
        return_value = self.sensor_data[location] if location in self.sensor_data else None
        self.can_get_data.release()
        return return_value

    def get_device_data(self, location):
        """
        @brief Fine-grained, location-locked data retrieval interface.
        Pre-condition: Acquisition of the specific location lock ensures data consistency during concurrent updates.
        """
        if location not in self.sensor_data:
            return None

        self.data_locks.get(location).acquire()
        new_data = self.sensor_data[location]
        self.data_locks.get(location).release()
        return new_data

    def set_data(self, location, data):
        """
        @brief Synchronized data update for a specific sensor location.
        """
        if location in self.sensor_data:
            self.data_locks.get(location).acquire()
            self.sensor_data[location] = data
            self.data_locks.get(location).release()

    def shutdown(self):
        """
        @brief Gracefully terminates the device's management thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Top-level coordinator thread for simulation timepoints.
    Algorithm: Phased execution lifecycle managing the transition between task submission and global synchronization.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main coordination loop for the device node.
        """
        while True:
            self.device.can_get_data.acquire()
            
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                # Logic: Shutdown path - ensure all devices arrive at the exit barrier.
                self.device.master.barrier.wait()
                self.device.can_get_data.release()
                return

            # Phase Transition: Spawns a dedicated Scripter to manage worker pool lifecycle for this timepoint.
            script_instance = Scripter(self.device, neighbours)
            script_instance.start()

            # Block Logic: Waits for the supervisor to finish script assignment.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Signal worker termination for the current phase.
            self.device.script_over = True
            self.device.script_received.set()

            script_instance.join()

            # Post-condition Preparation: Re-queues scripts for the next cycle (state carry-over).
            for (script, location) in self.device.scripts:
                self.device.queue.put_nowait((script, location))

            self.device.script_over = False

            # Synchronization Phase: Align all devices across the cluster.
            self.device.master.barrier.wait()

            self.device.can_get_data.release()
            self.device.script_running.release()


class Scripter(Thread):
    """
    @brief Mid-tier coordinator that manages a pool of ScriptExecutor worker threads.
    """

    def __init__(self, device, neighbours):
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        @brief Main execution lifecycle for the scripter.
        Algorithm: Bootstraps executors and coordinates their shutdown via "poison pill" tasks.
        """
        list_executors = []

        # Logic: Spawns the execution workforce.
        for iterator in range(1, self.device.available_threads):
            executor = ScriptExecutor(self.device, self.device.queue, self.neighbours, iterator)
            list_executors.append(executor)
            executor.start()

        while True:
            # Logic: Waits for signals from DeviceThread regarding script assignment progress.
            self.device.script_received.wait()
            self.device.script_received.clear()

            if self.device.script_over:
                # Logic: Phase completion signal received - terminates all workers.
                for iterator in range(1, self.device.available_threads):
                    self.device.queue.put((None, None))

                for executor in list_executors:
                    executor.join()

                # Cleanup: Re-initializes the work queue for the next simulation phase.
                self.device.queue = Queue(-1)
                return

            self.device.script_running.release()


class ScriptExecutor(Thread):
    """
    @brief Bottom-tier worker thread implementing the execution of individual scripts.
    """

    def __init__(self, device, queue, neighbours, identifier):
        Thread.__init__(self, name="Script Executor %d" % identifier)
        self.device = device
        self.queue = queue
        self.neighbours = neighbours

    def run(self):
        """
        @brief main loop for script task consumption and processing.
        Algorithm: Producer-Consumer pattern with distributed data aggregation and propagation.
        """
        while True:
            # Logic: Consumes the next script task or exits if termination signal (None) received.
            (script, location) = self.queue.get()
            if script is None:
                return

            script_data = []
            
            # Distributed Aggregation Phase: Collect readings from neighbors using fine-grained locks.
            for device in self.neighbours:
                data = device.get_device_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Logic: Include local data in processing.
            data = self.device.get_device_data(location)
            if data is not None:
                script_data.append(data)

            # Execution and Propagation Phase: Computes new state and broadcasts to peers.
            if script_data != []:
                result = script.run(script_data)
                
                for device in self.neighbours:
                    device.set_data(location, result)
                
                self.device.set_data(location, result)


class ReusableBarrier:
    """
    @brief Implementation of a two-phase synchronization barrier using semaphores.
    * Algorithm: Dual-stage arrival pattern to ensure strict thread alignment across simulation cycles.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the barrier with thread count and phase-specific control primitives.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Intent: Mutable shared counter for phase 1.
        self.count_threads2 = [self.num_threads] # Intent: Mutable shared counter for phase 2.
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Synchronizes the calling thread through both phases of the barrier.
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
                for iterator in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
