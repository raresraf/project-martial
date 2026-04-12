"""
@file device.py
@brief Distributed sensor unit simulation with a managed thread pool and cyclic synchronization.
@details Implements a peer-to-peer network of sensing devices that perform synchronized data 
aggregation. Uses a persistent ThreadPool for parallel task execution and coordinates 
temporal timepoints across the cluster via a two-phase semaphore-based barrier.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

class ThreadPool(object):
    """
    @brief Resource-bounded execution environment for device tasks.
    Functional Utility: Manages a pool of reusable worker threads synchronized via 
     a shared task queue.
    """
    
    def __init__(self, num_threads, device):
        self.__device = device
        self.__queue = Queue(num_threads)
        self.__threads = [Thread(target=self.work) for _ in range(num_threads)]

        # Initialization: Activates persistent workers.
        for thread in self.__threads:
            thread.start()

    def work(self):
        """
        @brief Continuous worker loop: acquisition -> aggregation -> propagation.
        Architecture: Implements a distributed Map-Reduce operation.
        """
        while True:
            # Acquisition: Blocks until a task or shutdown signal is received.
            script, location, neighbours = self.__queue.get()

            # Protocol: Poison pill check for termination.
            if not script and not neighbours:
                self.__queue.task_done()
                break

            script_data = []
            
            # Map Phase: Aggregates state from topological neighbors.
            for peer in neighbours:
                if self.__device.device_id != peer.device_id:
                    data = peer.get_data(location)
                    if data is not None:
                        script_data.append(data)

            # Inclusion: Local state.
            data = self.__device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Computation: Executes the aggregation logic.
                result = script.run(script_data)

                # Reduce/Update Phase: Propagates the result to all participants.
                for peer in neighbours:
                    if self.__device.device_id != peer.device_id:
                        peer.set_data(location, result)
                self.__device.set_data(location, result)
            
            # Signal task completion to the pool manager.
            self.__queue.task_done()

    def add_tasks(self, scripts, neighbours):
        """
        @brief Submits a batch of tasks to the pool.
        """
        for script, location in scripts:
            self.__queue.put((script, location, neighbours))

    def wait_threads(self):
        """
        @brief Blocks until the current work queue is fully processed.
        """
        self.__queue.join()

    def stop_threads(self):
        """
        @brief Gracefully terminates all pool workers.
        """
        self.__queue.join()
        for _ in self.__threads:
            self.__queue.put((None, None, None))
        for thread in self.__threads:
            thread.join()


class ReusableBarrierSem(object):
    """
    @brief implementation of a two-phase cyclic barrier using semaphores.
    Functional Utility: Synchronizes a fixed group of threads across recurring execution cycles.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """
        @brief executes the two-turnstile protocol.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief First turnstile: Arrival coordination.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Release all blocked threads.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Second turnstile: Reset/Departure coordination.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Device(object):
    """
    @brief Logic controller for an autonomous sensor entity.
    Functional Utility: Manages local data buffers, organizes task execution cycles, 
    and shares synchronization resources with peers.
    """
    
    num_threads = 8

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of local readings.
        @param supervisor entity providing topology discovery.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        
        # Lifecycle: Spawns the main orchestrator thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        
        self.barrier = None
        # Lock Registry: dedicated locks for each sensor location to ensure cluster-wide atomicity.
        self.locks = {loc: Lock() for loc in sensor_data}

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collaborative initialization of shared synchronization resources.
        Logic: Designates device 0 as the allocator for the global cyclic barrier.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Schedules a processing script for the current temporal cycle.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Protocol: Signals end of the script submission phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Thread-safe retrieval of local sensor state.
        Functional Utility: Acquires the local location lock before returning the reading.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        @brief Thread-safe update of local sensor state.
        Functional Utility: Commits the update and releases the local location lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """
        @brief Gracefully terminates the device orchestrator thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Orchestration thread managing execution cycles and the worker pool.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Master %d" % device.device_id)
        self.device = device
        # Architecture: Dedicated ThreadPool for parallel task execution.
        self.pool = ThreadPool(Device.num_threads, device)

    def run(self):
        """
        @brief Main coordination loop: discovery -> batch injection -> synchronization.
        """
        while True:
            # Discovery: Queries network supervisor for current neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Task Injection Loop.
            while True:
                if self.device.script_received.is_set():
                    # Logic: Offloads assigned scripts to the worker pool.
                    self.pool.add_tasks(self.device.scripts, neighbours)
                    self.device.script_received.clear()

                if self.device.timepoint_done.is_set():
                    # Protocol: Terminates injection phase for current timepoint.
                    self.device.timepoint_done.clear()
                    self.device.script_received.set()
                    break

            # Sync: Ensures all local workers finalize tasks.
            self.pool.wait_threads()
            
            # Global Sync: align cluster devices at the barrier.
            self.device.barrier.wait()

        # Termination Protocol.
        self.pool.stop_threads()
