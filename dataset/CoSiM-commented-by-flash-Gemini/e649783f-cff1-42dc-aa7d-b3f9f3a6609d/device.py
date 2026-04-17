"""
@e649783f-cff1-42dc-aa7d-b3f9f3a6609d/device.py
@brief Distributed sensor network simulation with batch-oriented transient threading.
This module implements a coordinated processing framework where a node manager 
(SupervisorThread) dispatches computational tasks in discrete batches of 8 
transient threads (Slave). Consistency is maintained through a lazily-initialized 
global spatial lock pool and a network-wide two-phase synchronization barrier.

Domain: Batch Task Dispatch, Transient Worker Threads, Lazy Spatial Locking.
"""

from threading import Event, Thread, Semaphore, Lock

class ReusableBarrier(object):
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Provides a robust synchronization point for a fixed group 
    of threads using a double-gate mechanism with semaphores.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: participants count.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Executes the two-phase rendezvous sequence."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Internal gate logic using atomic counter decrement and semaphore release."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Release the gate for the group.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local sensor data, coordinates global resource 
    discovery, and lazily initializes spatial mutual exclusion locks.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.threads = []
        self.devices = []
        self.semafor = Semaphore(0)
        self.timepoint_done = Event()
        # Primary node management thread.
        self.thread = SupervisorThread(self)
        self.thread.start()
        # concurrency boundary.
        self.num_scr = 8
        # Spatial lock pool (max 100 locations).
        self.lock = [None] * 100

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node initializes and propagates the shared barrier. 
        Aggregates peer references for lock sharing.
        """
        if self.barrier is None:
            self.barrier = ReusableBarrier(len(devices))
            for dev in devices:
                # Shared lock pool reference.
                dev.lock = self.lock
                if dev.barrier is None:
                    dev.barrier = self.barrier

        for dev in devices:
            if dev is not None:
                self.devices.append(dev)

    def assign_script(self, script, location):
        """
        Task and Lock Registration.
        Logic: Lazily initializes spatial mutexes. Searches through peer devices 
        to ensure a shared lock instance is used across the network.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Block Logic: Lazy Spatial Mutex resolution.
            if self.lock[location] is None:
                for device in self.devices:
                    if device.lock[location] is not None:
                        self.lock[location] = device.lock[location]
                        break
                # Create a new lock if not found in any peer.
                if self.lock[location] is None:
                    self.lock[location] = Lock()
            self.script_received.set()
        else:
            # Signal end of step workload.
            self.timepoint_done.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the node management thread."""
        self.thread.join()


class SupervisorThread(Thread):
    """
    Node-level task manager.
    Functional Utility: Orchestrates simulation phases and dispatches scripts 
    to transient worker threads in fixed-size batches.
    """
    
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main simulation loop.
        Algorithm: Iterative batch processing with join-based worker reclamation.
        """
        while True:
            # Topology Discovery.
            neighb = self.device.supervisor.get_neighbours()
            if neighb is None:
                break
            
            # Wait for supervisor to finalize script assignments.
            self.device.timepoint_done.wait()
            i = 0
            # Block Logic: Batch Parallelization.
            while i < len(self.device.scripts):
                # Dispatch tasks in batches of 8.
                for _ in range(0, self.device.num_scr):
                    pair = self.device.scripts[i]
                    # Fork: spawn transient worker.
                    new_thread = Slave(self.device, pair[1], neighb, pair[0])
                    self.device.threads.append(new_thread)
                    new_thread.start()
                    i = i + 1
                    if i >= len(self.device.scripts):
                        break
                
                # Join: Wait for the current batch to complete before next dispatch.
                for thread in self.device.threads:
                    thread.join()
                self.device.threads = []

            # Phase reset for next cycle.
            self.device.timepoint_done.clear()
            # Global Consensus rendezvous.
            self.device.barrier.wait()

class Slave(Thread):
    """
    Transient worker thread.
    Functional Utility: Executes a single script computation with spatial 
    mutual exclusion.
    """

    def __init__(self, device, location, neighbours, script):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script

    def run(self):
        """
        Execution logic.
        Logic: Implements atomic read-modify-write via the spatial lock pool.
        """
        # Critical Section: Spatial mutual exclusion across the entire network.
        self.device.lock[self.location].acquire()
        script_data = []
        
        # Aggregate neighborhood and local data.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Compute results and propagate.
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
        
        # Release the spatial mutex.
        self.device.lock[self.location].release()
