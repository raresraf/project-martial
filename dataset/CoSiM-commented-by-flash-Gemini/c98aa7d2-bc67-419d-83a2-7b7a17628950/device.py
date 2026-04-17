"""
@c98aa7d2-bc67-419d-83a2-7b7a17628950/device.py
@brief Distributed sensor processing simulation with phased sub-thread execution and global synchronization.
* Algorithm: Event-driven worker spawning for every script with multi-phase semaphore barriers and per-location locking.
* Functional Utility: Orchestrates simulation timepoints across a device cluster by dynamically dispatching processing tasks and synchronizing state across nodes.
"""

from threading import Event, Thread, Semaphore, Lock

class ReusableBarrier(object):
    """
    @brief Implementation of a two-phase synchronization barrier using semaphores.
    * Algorithm: Dual-stage arrival/release to prevent thread overruns between consecutive timepoints.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the barrier with a target thread count and dual phase control.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Intent: Shared mutable counter for phase 1.
        self.count_threads2 = [self.num_threads] # Intent: Shared mutable counter for phase 2.
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
        @brief Executes a single synchronization phase.
        Invariant: The last thread to arrive releases all waiting peers.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Logic: Collective release of all threads at the synchronization point.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    @brief Represents a sensor node that manages its own readings and processing threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device state and bootstraps the main coordinator thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.lock = Lock() # Intent: General purpose lock for local data access.
        self.locationlock = [] # Intent: Shared list of locks for specific sensor locations.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization and distribution of shared synchronization primitives.
        Invariant: Root node (ID 0) establishes a global set of 100 location locks and a shared barrier.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            locationlock = []
            # Logic: Pre-allocates a pool of locks to cover a standard range of sensor locations.
            for _ in xrange(100):
                locationlock.append(Lock())
            for device in devices:
                device.locationlock = locationlock
                device.set_barrier(barrier)
        else:
            pass

    def set_barrier(self, barrier):
        """
        @brief Assigns the shared cluster barrier to this device.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Enqueues a script for the current processing cycle.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Signals completion of script arrival for the current timepoint.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves local sensor data for a specific location.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates local sensor data for a specific location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the device's management thread.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief Main coordination thread managing timepoint progression and sub-thread spawning.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator thread for a specific device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core lifecycle loop of the device node.
        Algorithm: Dynamic spawning of individual sub-threads for each script unit in a timepoint.
        """
        while True:
            # Logic: Neighbor discovery and simulation exit condition.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Block Logic: Ensures all tasks for the current cycle have arrived.
            self.device.timepoint_done.wait()
            subthreads = []

            # Dispatch Phase: Spawns one sub-thread per assigned script.
            for (script, location) in self.device.scripts:
                subthreads.append(
                    DeviceSubThread(self, neighbours, script, location))
                subthreads[len(subthreads) - 1].start()
            
            # Logic: Wait for all local script processing to complete.
            for subthread in subthreads:
                subthread.join()
            
            # Post-condition: Reset phase state and align with peers at the global barrier.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

class DeviceSubThread(Thread):
    """
    @brief Worker thread dedicated to executing a single script at a specific location.
    """
    
    def __init__(self, devicethread, neighbours, script, location):
        """
        @brief Initializes the worker with its task parameters and context.
        """
        Thread.__init__(self, name="Device SubThread %d"
            % devicethread.device.device_id)
        self.neighbours = neighbours
        self.devicethread = devicethread
        self.script = script
        self.location = location

    def run(self):
        """
        @brief Main execution logic for a single script unit.
        Algorithm: Resource-locked execution with distributed data aggregation and propagation.
        """
        # Pre-condition: Acquire location-specific lock for atomic cluster-wide update.
        self.devicethread.device.locationlock[self.location].acquire()
        script_data = []
        
        # Distributed Aggregation Phase: Accumulate data from neighborhood and self.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.devicethread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Functional Utility: Runs processing logic and broadcasts result.
        if script_data != []:
            result = self.script.run(script_data)
            for device in self.neighbours:
                # Logic: Thread-safe data update across the network.
                with device.lock:
                    device.set_data(self.location, result)
            
            with self.devicethread.device.lock:
                self.devicethread.device.set_data(self.location, result)
        
        # Post-condition: Release location lock.
        self.devicethread.device.locationlock[self.location].release()
