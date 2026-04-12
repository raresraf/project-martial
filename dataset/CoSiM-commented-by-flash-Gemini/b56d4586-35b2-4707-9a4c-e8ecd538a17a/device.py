"""
@file device.py
@brief Distributed sensor processing system with global synchronization and parallel execution.
@details Implements a peer-to-peer network of sensing units that perform synchronized data 
aggregation. Uses a global cyclic barrier for temporal alignment across timepoints 
and dynamic, location-specific locks to ensure cluster-wide data consistency.
"""

from threading import Event, Thread, Lock, Semaphore

# Global Shared Resources: Primitives shared across all Device instances in the cluster.
L_LOCKS = {}
LOCK = Lock()
BARRIER = None

class ReusableBarrier():
    """
    @brief Implementation of a two-phase cyclic barrier using semaphores.
    Functional Utility: Synchronizes a fixed group of threads across recurring execution windows.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # State: Thread counters wrapped in lists for pass-by-reference mutation in phase logic.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Executes the two-turnstile protocol (Phase 1 then Phase 2).
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Turnstile logic: blocks arrivals until threshold is met, then releases all.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Release all blocked threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for reuse in next cycle.
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    @brief Controller for an autonomous sensing entity in a distributed network.
    Functional Utility: Manages local data buffers and participates in cluster-wide 
    synchronization via global barriers and locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @param device_id Unique identifier.
        @param sensor_data Initial dictionary of sensor location-value pairs.
        @param supervisor Entity providing topological neighborhood discovery.
        """
        self.event = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # Lifecycle: the management thread is initialized but not yet started.
        self.thread = DeviceThread(self)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization of shared cluster resources.
        Logic: Designates device 0 as the allocator for the global cyclic barrier.
        """
        if self.device_id > 0:
            # Sync: peer devices wait for the master to initialize resources.
            self.event.wait()
            self.thread.start()
        else:
            global BARRIER
            BARRIER = ReusableBarrier(len(devices))
            # Propagation: wakes all peer devices once the barrier is ready.
            for device in devices:
                if device.device_id > 0:
                    device.event.set()
            self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Schedules a processing task for the current unit of time.
        """
        if script is None:
            # Protocol: signals end of task assignment phase.
            self.timepoint_done.set()
        else:
            self.scripts.append((script, location))

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
        @brief Gracefully terminates the device orchestrator thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Orchestrator thread managing timepoint cycles and script execution.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main coordination loop: synchronization -> execution -> state reset.
        """
        while True:
            # Global Sync: align all devices at the start of the timepoint cycle.
            global BARRIER
            BARRIER.wait()
            
            # Discovery: fetch current topological neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Termination: supervisor indicates end of simulation.
                break

            # Sync: wait for local task assignment to conclude.
            self.device.timepoint_done.wait()

            /**
             * Block Logic: Script execution engine.
             * Logic: Iteratively processes each assigned (script, location) pair.
             */
            for (script, location) in self.device.scripts:
                # Lock Management: ensures thread-safe allocation of location-specific locks.
                global LOCK, L_LOCKS
                LOCK.acquire()
                if location not in L_LOCKS:
                    L_LOCKS[location] = Lock()
                
                # Critical Section: exclusive access to the location across the cluster.
                with L_LOCKS[location]:
                    LOCK.release() # Unlock global registry once location lock is held.

                    script_data = []
                    # Map Phase: aggregate state from neighborhood.
                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    # Inclusion: local state.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data:
                        # Computation: execution of the aggregation script.
                        result = script.run(script_data)
                        
                        # Reduce Phase: propagates result back to all participants.
                        for device in neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)
                
                # Potential deadlock prevention: ensure global LOCK is not held if loop continues.
                if not LOCK.locked():
                    pass # LOCK already released in with-block context logic above.

            # Cycle reset for the next unit of time.
            self.device.timepoint_done.clear()
