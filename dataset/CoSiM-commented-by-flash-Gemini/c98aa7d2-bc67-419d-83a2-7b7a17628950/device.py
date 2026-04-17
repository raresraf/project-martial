"""
@c98aa7d2-bc67-419d-83a2-7b7a17628950/device.py
@brief Distributed sensor network simulation with dynamic sub-threading and spatial locking.
This module implements a parallel execution model where each computational script 
triggers the creation of a transient 'DeviceSubThread'. Global consistency is maintained 
through a two-phase synchronization barrier and a centralized pool of mutual 
exclusion locks mapped to individual sensor locations.

Domain: Concurrent Programming, Dynamic Thread Spawning, Two-Phase Barriers.
"""

from threading import Event, Thread, Semaphore, Lock



class ReusableBarrier(object):
    """
    Two-phase reusable barrier implementation using semaphores.
    Functional Utility: Implements a double-gate rendezvous to ensure all threads 
    have reached a common execution point before allowing any to proceed to the next cycle.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier state.
        @param num_threads: Number of participants in the synchronization.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Orchestrates the two phases of the synchronization rendezvous."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Internal phase logic using a shared counter and semaphore gate.
        Logic: The last thread to arrive releases all waiting participants.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()




class Device(object):
    """
    Node entity in the sensor network simulation.
    Functional Utility: Manages local data and coordinates global resource 
    distribution for synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        # Mutex for protecting per-device state updates.
        self.lock = Lock()
        self.locationlock = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource factory.
        Logic: Node 0 initializes the shared barrier and a static pool of 
        100 spatial locks for the entire network.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            # Pre-allocates a fixed pool of locks for sensor locations.
            locationlock = []
            for _ in xrange(100):
                locationlock.append(Lock())
            
            # Propagation: Distributes resources to all network members.
            for device in devices:
                device.locationlock = locationlock
                device.set_barrier(barrier)
        else:
            pass

    def set_barrier(self, barrier):
        """Injects the shared network barrier."""
        self.barrier = barrier

    def assign_script(self, script, location):
        """Registers a task and signals completion of the assignment phase."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates local sensor value."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main management thread."""
        self.thread.join()




class DeviceThread(Thread):
    """
    Simulated node orchestration thread.
    Functional Utility: Manages simulation timepoints and spawns sub-threads 
    to handle assigned computational scripts in parallel.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main simulation execution loop.
        Algorithm: Iterative sub-thread spawning with barrier coordination.
        """
        while True:
            # Topology refresh.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Block Logic: Simulation Phase Timing.
            # Wait for supervisor to signal that work is ready.
            self.device.timepoint_done.wait()
            subthreads = []

            # Block Logic: Task Parallelization.
            # Spawns a dedicated thread for every script in the current timepoint.
            for (script, location) in self.device.scripts:
                subthreads.append(
                    DeviceSubThread(self, neighbours, script, location))
                subthreads[len(subthreads) - 1].start()
            
            # Wait for all transient sub-threads to complete their work.
            for subthread in subthreads:
                subthread.join()
            
            # Cleanup for the next processing cycle.
            self.device.timepoint_done.clear()
            
            # Global Consensus Point.
            self.device.barrier.wait()


class DeviceSubThread(Thread):
    """
    Transient worker thread for executing a single script.
    Functional Utility: Implements a double-lock pattern (spatial location lock 
    plus device-level lock) for consistent data propagation.
    """
    
    def __init__(self, devicethread, neighbours, script, location):
        Thread.__init__(self, name="Device SubThread %d"
            % devicethread.device.device_id)
        self.neighbours = neighbours
        self.devicethread = devicethread
        self.script = script
        self.location = location

    def run(self):
        """
        Execution logic.
        Logic: Aggregates data while holding the spatial mutex, then propagates 
        results while holding per-device mutexes.
        """
        # Critical Section 1: Spatial mutual exclusion across the entire network.
        self.devicethread.device.locationlock[self.location].acquire()
        script_data = []
        
        # Aggregate neighborhood state.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Include local state.
        data = self.devicethread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Apply script computation.
            result = self.script.run(script_data)
            
            # Propagation: Atomic updates to neighbor states.
            for device in self.neighbours:
                with device.lock:
                    device.set_data(self.location, result)
            
            # Update local state.
            with self.devicethread.device.lock:
                self.devicethread.device.set_data(self.location, result)
        
        # Release the spatial lock.
        self.devicethread.device.locationlock[self.location].release()
