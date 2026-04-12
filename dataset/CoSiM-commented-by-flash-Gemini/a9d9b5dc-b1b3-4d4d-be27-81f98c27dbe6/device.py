"""
@file device.py
@brief Distributed sensor unit simulation with thread-pool script execution and cyclic synchronization.
@details Implements a peer-to-peer architecture where devices perform synchronized 
data aggregation. Uses a task queue with worker threads and a custom two-phase 
semaphore-based barrier for coordination.
"""

from threading import Event, Semaphore, Lock, Thread
from Queue import Queue

class Device(object):
    """
    @brief Logic controller for an autonomous sensing entity.
    Functional Utility: Manages local data state, schedules processing tasks, and 
    participates in cluster-wide synchronization via shared barriers and locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of sensor readings.
        @param supervisor Entity providing topological neighbor discovery.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # Lifecycle: Spawns the main management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

        self.barrier = None
        self.location_lock = None
        # Capacity: Fixed number of workers per device.
        self.NUM_THREADS = 8

    def __str__(self):
        return "Device %d" % self.device_id

    def is_master_thread(self, devices):
        """
        @brief Election logic to designate a master device for resource allocation.
        @return 1 if self has the minimal ID in the cluster, 0 otherwise.
        """
        for device in devices:
            if device.device_id < self.device_id:
                return 0
        return 1

    def setup_devices(self, devices):
        """
        @brief Collaborative initialization of cluster-wide synchronization primitives.
        Logic: The elected master allocates the shared barrier and the global lock registry.
        """
        if self.is_master_thread(devices) == 1:
            barrier = ReusableBarrier(len(devices))
            location_lock = {}
            self.set_barrier_lock(devices, barrier, location_lock)

    def set_barrier_lock(self, devices, barrier, location_lock):
        """
        @brief Propagates shared resources to all peer devices.
        """
        for device in devices:
            device.barrier = barrier
            # Allocation: Ensures a mutual exclusion lock exists for every sensor location.
            for location in device.sensor_data:
                if location not in location_lock:
                    location_lock[location] = Lock()
            device.location_lock = location_lock

    def assign_script(self, script, location):
        """
        @brief Enqueues a processing script for the current temporal window.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Protocol: Signals end of script assignment phase for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief retrieval of local sensor readings.
        """
        data = None
        if location in self.sensor_data:
            data = self.sensor_data[location]
        return data

    def set_data(self, location, data, source=None):
        """
        @brief Update of local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device lifecycle.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Orchestrator thread managing execution cycles (discovery -> execution -> sync).
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_scripts(self, queue, neighbours):
        """
        @brief Internal worker logic for executing aggregation tasks from the queue.
        Architecture: Implements a distributed Map-Reduce operation.
        """
        try:
            # Non-blocking acquisition of the next task.
            (script, location) = queue.get_nowait()
            
            # Lock: Ensures atomic access to the specific sensor location across the cluster.
            lock_location = self.device.location_lock.get(location)
            lock_location.acquire()
            
            script_data = []
            # Map Phase: Aggregates data from neighbors.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
            # Inclusion: Self data.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
            
            if script_data != []:
                # Computation: Executes the aggregation script.
                result = script.run(script_data)

                # Reduce Phase: Updates all participants with the result.
                for device in neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
            
            lock_location.release()
            queue.task_done()
        except:
            # Logic: queue empty or lock error; terminate worker.
            pass

    def start_threads(self, threadlist):
        for thread in threadlist:
            thread.start()

    def join_threads(self, threadlist):
        for thread in threadlist:
            thread.join()

    def run(self):
        """
        @brief Main coordination loop for the device.
        """
        while True:
            # Discovery: Fetches topological neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Sync: Wait for all scripts to be assigned for this timepoint.
            self.device.timepoint_done.wait()

            # Task Distribution: populates a queue for the local worker pool.
            queue = Queue()
            for (script, location) in self.device.scripts:
                queue.put((script, location))

            /**
             * Block Logic: Parallel execution via worker pool.
             * Logic: Spawns NUM_THREADS to consume tasks from the shared queue.
             */
            threadlist = []
            for _ in range(self.device.NUM_THREADS):
                thread = Thread(target=self.run_scripts, args=(queue, neighbours))
                threadlist.append(thread)
            
            self.start_threads(threadlist)
            self.join_threads(threadlist)
            queue.join()

            # State Reset and Global Synchronization.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

class ReusableBarrier():
    """
    @brief Implementation of a two-phase cyclic barrier using semaphores.
    Functional Utility: Provides reliable synchronization points for multi-threaded cycles.
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
        @brief Two-turnstile protocol to ensure all threads align before release.
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
                # Release arrival turnstile.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Second turnstile: Departure coordination.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Reset barrier for next cycle.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()
