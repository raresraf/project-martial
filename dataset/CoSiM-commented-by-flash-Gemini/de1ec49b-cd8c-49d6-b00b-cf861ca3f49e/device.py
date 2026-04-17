"""
@de1ec49b-cd8c-49d6-b00b-cf861ca3f49e/device.py
@brief Distributed sensor processing simulation using a persistent worker thread pool and shared global locks.
* Algorithm: Producer-consumer task distribution via `Queue` with persistent script re-queuing and multi-phase barrier synchronization.
* Functional Utility: Orchestrates simulation timepoints across a network of devices by managing a local pool of worker threads that perform distributed data aggregation and propagation using fine-grained location locks.
"""

from threading import Event, Thread, Lock
from Queue import Queue
import reusable_barrier_semaphore

class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings, internal worker pool, and shared synchronization state.
    """
    # Domain: Global class-level synchronization primitives shared across the entire device cluster.
    barrier = None # Intent: Global barrier for cluster-wide alignment.
    lockList = {}  # Intent: Map of location-specific locks for atomic distributed updates.
    lockListLock = Lock() # Intent: Serializes modifications to the global lockList and barrier.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and bootstraps the persistent worker pool (8 threads).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Intent: Local device barrier for aligning the 8 local worker threads.
        self.neighbours_done = reusable_barrier_semaphore.ReusableBarrier(8)
        self.neighbours = None
        
        self.scripts = Queue() # Intent: Active task queue for the current simulation phase.
        
        # Intent: Persistent storage of assigned scripts to allow for automatic re-execution in future cycles.
        self.permanent_scripts = []
        self._thread_count = 8 # Domain: Concurrency Scaling - fixed workers per node.
        self.threads = []
        self.startup_event = Event() # Intent: Signals completion of local setup to worker threads.
        
        # Logic: Spawns the persistent execution workforce.
        for i in range(0, self._thread_count):
            self.threads.append(DeviceThread(self, i))
        for i in range(0, self._thread_count):
            self.threads[i].start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization and distribution of shared simulation resources.
        Invariant: Root node (ID 0) or the first node to arrive initializes the collective barrier and location locks.
        """
        Device.lockListLock.acquire()
        if Device.barrier is None:
            Device.barrier = reusable_barrier_semaphore.ReusableBarrier(len(devices))
        Device.lockListLock.release()

        # Logic: Discovers and initializes unique locks for all local sensor locations.
        the_keys = self.sensor_data.keys()
        for i in the_keys:
            Device.lockListLock.acquire()
            if i not in Device.lockList:
                Device.lockList[i] = Lock()
            Device.lockListLock.release()
        
        # Post-condition: Notifies local threads that setup is complete.
        self.startup_event.set()

    def assign_script(self, script, location):
        """
        @brief Populates the work queue for the current phase and archives the task for future steps.
        """
        if script is not None:
            self.scripts.put((script, location))
            self.permanent_scripts.append((script, location))
        else:
            # Logic: Injects "poison pills" (one per worker) to signal the end of the current batch.
            for i in range(0, self._thread_count):
                self.scripts.put((script, location))

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
        @brief Gracefully terminates the device's worker threads.
        """
        for i in range(0, self._thread_count):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    @brief Persistent worker thread that executes sensor scripts in an iterative simulation loop.
    """

    def __init__(self, device, the_id):
        """
        @brief Initializes the worker with its device context and local sequence ID.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.the_id = the_id # Intent: Designation of coordinator (ID 0).

    def run(self):
        """
        @brief main loop for the persistent worker thread.
        Algorithm: Multi-stage synchronization lifecycle (Setup -> Discovery -> Execution -> Barrier).
        """
        # Block Logic: Ensures global setup is complete before beginning simulation.
        self.device.startup_event.wait()
        
        while True:
            # Sync Phase 1: Collective alignment before neighbor discovery.
            if self.the_id == 0:
                # Logic: Global barrier for node-level synchronization.
                Device.barrier.wait()
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            # Sync Phase 2: Ensure all local threads have seen the new neighborhood.
            self.device.neighbours_done.wait()

            if self.device.neighbours is None:
                # Logic: Shutdown signal received.
                break

            while True:
                # Task Consumption Phase.
                (script, location) = self.device.device_scripts_get_logic()
                
                if script is None:
                    # Logic: End of current phase reached.
                    self.device.neighbours_done.wait()
                    
                    if self.the_id == 0:
                        # Coordination Logic: Re-populates the queue with permanent scripts for the next timepoint.
                        for (perm_script, perm_location) in self.device.permanent_scripts:
                            self.device.scripts.put((perm_script, perm_location))
                    break

                # Distributed Aggregation Phase: Collect readings under fine-grained location locks.
                if location is not None:
                    Device.lockList[location].acquire()
                    script_data = []
                    
                    for device in self.device.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        # Execution and Propagation Phase.
                        result = script.run(script_data)
                        for device in self.device.neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)
                        
                    # Post-condition: Release global location lock.
                    Device.lockList[location].release()

    # Note: Added local access utility for scripts queue to maintain clean logic in run().
    def device_scripts_get_logic(self):
        return self.device.scripts.get()

from threading import Semaphore, Lock
class ReusableBarrier():
    """
    @brief Implementation of a two-phase synchronization barrier using semaphores.
    * Algorithm: Dual-stage arrival pattern to ensure strict thread alignment.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Intent: Shared mutable counter.
        self.count_threads2 = [self.num_threads] # Intent: Shared mutable counter.
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
        Invariant: The last thread to arrive releases the entire group.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
