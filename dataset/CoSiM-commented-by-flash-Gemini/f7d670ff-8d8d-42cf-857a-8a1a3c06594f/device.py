"""
@f7d670ff-8d8d-42cf-857a-8a1a3c06594f/device.py
@brief Distributed sensor network simulation with throttled dynamic thread spawning.
This module implements a parallel processing model where individual computational 
tasks are executed by transient worker threads (DataScript). Concurrency is throttled 
by a node-local semaphore to limit system resource usage, while global consistency 
is enforced through a network-wide pool of spatial locks. A robust two-phase 
synchronization barrier ensures total temporal alignment across simulation steps.

Domain: Concurrent Programming, Throttled Parallelism, Distributed State Mutual Exclusion.
"""

from threading import Event, Thread
from threading import Lock, Semaphore

class ReusableBarrier():
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Provides a temporal rendezvous point for a fixed group 
    of threads, ensuring clean group transitions through a double-gate mechanism.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: Number of participating threads.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Orchestrates the two-phase arrival and exit protocol."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Internal gate logic using atomic counter decrement and semaphore signaling."""
        with self.count_lock:
            count_threads[0] -= 1

            if count_threads[0] == 0:
                # Arrival threshold reached: release the group.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for next phase/reuse.
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Coordinator entity for a network node.
    Functional Utility: Manages local data state, coordinates the distribution 
    of global synchronization resources, and throttles local task parallelization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        # Signaling event for network-wide initialization.
        self.done_setup = Event()
        self.device_id = device_id
        # Primary node management thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.script_received = Event()
        self.sensor_data = sensor_data
        
        # Throttling Mechanism: limits local concurrency to 8 worker threads.
        self.semaphore = Semaphore(value=8)

        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        self.nr_thread = 0
        self.lock_timepoint = Lock()
        # Transient list of worker threads for the current step.
        self.script_list = []
        # Spatial lock pool populated during setup.
        self.lock_index = []

        self.r_barrier = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource factory.
        Logic: Coordinator node (ID 0) initializes the shared barrier and a 
        pool of 24 spatial locks, distributing them to all peer nodes.
        """
        used_devices = len(devices)
        if self.device_id is 0:
            r_barrier = ReusableBarrier(used_devices)
            # Allocation: pre-allocates mutexes for expected spatial locations.
            for _ in range(0, 24):
                self.lock_index.append(Lock())

            # Propagation: distribute resources to all peers and signal completion.
            for d in range(len(devices)):
                devices[d].lock_index = self.lock_index
                devices[d].r_barrier = r_barrier
                devices[d].done_setup.set()

    def assign_script(self, script, location):
        """Registers a task and signals completion of the simulation step assignment."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def close_scripts(self):
        """
        Worker reclamation phase.
        Logic: Gracefully joins all threads spawned during the current step 
        and flushes the transient worker list.
        """
        nrThreads = len(self.script_list)
        for i in range(0, nrThreads):
            self.script_list[i].join()

        for i in range(0, nrThreads):
            self.script_list.pop()

        self.nr_thread = 0

    def shutdown(self):
        """Gracefully joins the orchestration thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    Main orchestration thread for the node simulation.
    Functional Utility: Manages simulation steps and throttles the dispatch of 
    computational tasks to transient parallel workers.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_script(self, neighbours):
        """
        Task Dispatch Phase.
        Logic: Iteratively spawns worker threads, observing the local semaphore boundary.
        """
        for (script, location) in self.device.scripts:
            # Phase 1: Throttling.
            self.device.semaphore.acquire()
            
            # Phase 2: Spawning (Fork).
            self.device.script_list.append(DataScript\
            (neighbours, location, script, self.device))

            self.device.nr_thread = self.device.nr_thread + 1
            self.device.script_list[len(self.device.script_list)-1].start()

    def run(self):
        """
        Main simulation execution loop.
        Algorithm: Multi-node setup sync -> Iterative Step Coordination -> 
        Parallel Worker Execution -> Join -> Consensus.
        """
        # Block until global network initialization is complete.
        self.device.done_setup.wait()

        while True:
            # Refresh topology.
            with self.device.lock_timepoint:
                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    break

            # Block until assignment phase is complete.
            self.device.timepoint_done.wait()
            
            # Execute tasks in parallel.
            self.run_script(neighbours)

            # Global Consensus barrier (Arrival).
            self.device.r_barrier.wait()
            
            # Reclamation and Phase Reset.
            self.device.timepoint_done.clear()
            self.device.close_scripts()
            
            # Global Consensus barrier (Exit).
            self.device.r_barrier.wait()


class DataScript(Thread):
    """
    Transient worker thread implementation.
    Functional Utility: Executes a computational script while maintaining 
    network-wide spatial mutual exclusion for the target location.
    """
    
    def __init__(self, neighbours, location, script, scr_device):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.scr_device = scr_device


    def getdata(self, script_data):
        """Aggregates local sensor state."""
        data = self.scr_device.get_data(self.location)
        if data is not None:
            script_data.append(data)

    def scriptdata(self, script_data):
        """
        Computational logic and result propagation.
        Algorithm: Compute -> Propagate -> Local Signal.
        """
        if script_data != []:
            # Apply domain logic.
            result = self.script.run(script_data)
            
            # Propagate results back to neighbors and self.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.scr_device.set_data(self.location, result)
        
        # Ordered release of the throttling semaphore.
        self.scr_device.semaphore.release()

    def run(self):
        """
        Execution Logic.
        Logic: Implements atomic read-modify-write within a global spatial lock.
        """
        # Critical Section: Spatial mutual exclusion across the entire network.
        with self.scr_device.lock_index[self.location]:
            script_data = []

            # Aggregate neighborhood data.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            self.getdata(script_data)
            self.scriptdata(script_data)
