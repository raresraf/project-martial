"""
@cf589e3b-9095-4448-a0ab-1cef55de8ec3/device.py
@brief Distributed sensor processing simulation using static work partitioning and multi-tier synchronization.
* Algorithm: Round-robin script assignment to a fixed set of persistent worker threads with phased multi-barrier coordination.
* Functional Utility: Manages concurrent data processing across a device cluster by coordinating local thread pools and global shared resources.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    @brief Two-phase synchronization barrier implementation using counting semaphores.
    * Algorithm: Dual-stage arrival/release logic to prevent thread overruns between simulation timepoints.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the barrier with a target thread threshold.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread through both stages of the barrier.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Stage 1: Collects all threads and releases them simultaneously.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Logic: Collective release of all threads at the synchronization point.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Stage 2: Secondary synchronization to ensure consistent state across repeated cycles.
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
    @brief Encapsulates a sensor node with its local data and management thread pool.
    * Domain: Shared class-level synchronization primitives for global cluster coordination.
    """
    
    bar1 = ReusableBarrier(1) # Intent: Global barrier for cross-device alignment.
    event1 = Event()           # Intent: Global signal for system initialization.
    locck = []                 # Intent: Shared list of locks for protecting sensor locations.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device state and bootstraps its worker thread pool (8 threads).
        """
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.devices = []

        # Logic: Event list to signal phase starts to individual worker threads.
        self.event = []
        for _ in xrange(11): # Domain: Bounds the number of simulation timepoints.
            self.event.append(Event())

        self.nr_threads_device = 8 # Domain: Concurrency Scaling - persistent worker count.
        self.nr_thread_atribuire = 0 # Intent: Index for round-robin script assignment.
        
        # Intent: Internal device barrier for coordinating workers and the coordinator thread.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device + 1)

        # Logic: Spawns the main coordinator thread.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Logic: Spawns the persistent worker thread pool.
        self.threads = []
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Collective resource initialization and distribution.
        Invariant: Root node (ID 0) establishes the global locks and cluster-wide barrier.
        """
        self.devices = devices
        
        if self.device_id == 0:
            # Logic: Pre-allocates a pool of locks for possible sensor locations.
            for _ in xrange(30):
                Device.locck.append(Lock())
            Device.bar1 = ReusableBarrier(len(devices))
            Device.event1.set()

    def assign_script(self, script, location):
        """
        @brief Distributes a script to the local worker pool using a round-robin strategy.
        """
        if script is not None:
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            self.nr_thread_atribuire = (self.nr_thread_atribuire + 1) % self.nr_threads_device
        else:
            # Logic: Signals completion of assignment for the current phase.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves local sensor data for a specific location.
        """
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates local sensor data for a specific location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates all device threads.
        """
        self.thread.join()
        for threadd in self.threads:
            threadd.join()


class DeviceThread(Thread):
    """
    @brief Coordination thread managing phase transitions and global synchronization.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator thread for a specific device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None
        self.contor = 0 # Intent: Tracks the current simulation timepoint index.

    def run(self):
        """
        @brief Main execution lifecycle for the coordinator thread.
        Algorithm: Iterative phase management with multi-level barrier alignment.
        """
        # Block Logic: Waits for the cluster-wide start signal.
        Device.event1.wait()

        while True:
            # Logic: Refresh neighbor set from supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                # Logic: Shutdown signal - notifies all workers to exit.
                self.device.event[self.contor].set()
                break

            # Block Logic: Waits for script delivery completion.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Dispatch Phase: Signals all local worker threads to begin script execution for this timepoint.
            self.device.event[self.contor].set()
            self.contor += 1

            # Synchronization Stage 1: Wait for local worker pool completion.
            self.device.bar_threads_device.wait()

            # Synchronization Stage 2: Wait for cluster-wide alignment.
            Device.bar1.wait()

class ThreadAux(Thread):
    """
    @brief Persistent worker thread implementing the execution of assigned sensor scripts.
    """
    
    def __init__(self, device):
        """
        @brief Initializes the worker with its private task map.
        """
        Thread.__init__(self)
        self.device = device
        self.script_loc = {} # Intent: Map of assigned script instances to their target locations.
        self.contor = 0      # Intent: Tracks the current simulation timepoint index.

    def run(self):
        """
        @brief Main worker execution loop.
        Algorithm: Event-triggered execution followed by internal device barrier synchronization.
        """
        while True:
            # Block Logic: Waits for the coordinator to signal the start of a specific timepoint.
            self.device.event[self.contor].wait()
            self.contor += 1

            # Logic: Retrieve current neighborhood context.
            neigh = self.device.thread.neighbours
            if neigh is None:
                # Logic: Exit worker loop.
                break

            # Execution Phase: Process all assigned scripts.
            for script in self.script_loc:
                location = self.script_loc[script]
                
                # Pre-condition: Acquire global location lock for atomic distributed state update.
                Device.locck[location].acquire()
                script_data = []

                # Distributed Aggregation: Accumulate data from neighbors and local node.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execution and Propagation: Computes result and broadcasts across the neighborhood.
                if script_data != []:
                    result = script.run(script_data)
                    for device in neigh:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Post-condition: Release global lock.
                Device.locck[location].release()

            # Internal Synchronization: Signal local task completion to the coordinator.
            self.device.bar_threads_device.wait()
