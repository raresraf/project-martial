"""
@d321d653-7f33-4e70-9af8-0842de7fbd18/device.py
@brief Distributed sensor network simulation with on-demand thread spawning and dynamic locking.
This module implements a parallel processing framework where computational scripts 
trigger the creation of transient worker threads. It utilizes an on-demand spatial 
locking strategy, where mutexes for sensor locations are initialized as needed 
during runtime. Temporal synchronization is enforced via a robust two-phase 
semaphore-based barrier.

Domain: Dynamic Concurrency, On-Demand Synchronization, Distributed Simulation.
"""

from threading import Lock, Thread, Event
from reusable_barrier_semaphore import ReusableBarrier


class Device(object):
    """
    Core network node representation.
    Functional Utility: Manages local data state and coordinates the allocation 
    of shared synchronization resources across the group.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # Primary orchestration thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.locks = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource distribution.
        Logic: Elects a leader (node with minimum ID) to initialize the 
        ReusableBarrier and the shared spatial lock repository.
        """
        leader = devices[0]
        for device in devices:
            if device.device_id < leader.device_id:
                leader = device

        if self.device_id == leader.device_id:
            # Atomic resource creation by the coordinator.
            self.barrier = ReusableBarrier(len(devices))
            self.locks = {}
            # Propagation: Distribute shared resources to all peer nodes.
            for device in devices:
                device.barrier = self.barrier
                device.locks = self.locks

    def assign_script(self, script, location):
        """Registers a computational task and signals the local manager."""
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

    def shutdown(self):
        """Gracefully joins the orchestration thread."""
        self.thread.join()



class DeviceThread(Thread):
    """
    Node-level lifecycle coordinator.
    Functional Utility: Orchestrates the parallel execution of scripts for each 
    simulation timepoint by spawning dedicated sub-threads.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main simulation loop.
        Algorithm: Iterative sub-thread spawning with barrier coordination.
        """
        while True:
            # Topology refresh.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block until work assignment phase is complete.
            self.device.timepoint_done.wait()

            threads = []
            # Block Logic: Dynamic Parallelization.
            # Spawns a dedicated worker thread for every script in the current batch.
            for (script, location) in self.device.scripts:
                script_thread = ScriptThread(self.device, script, location, neighbours)
                threads.append(script_thread)
                threads[-1].start()

            # Wait for all local workers to complete.
            for thread in threads:
                thread.join()

            # Phase reset for the next step.
            self.device.timepoint_done.clear()

            # Global Consensus Point.
            self.device.barrier.wait()


class ScriptThread(Thread):
    """
    Transient worker thread for computational tasks.
    Functional Utility: Implements an on-demand spatial locking protocol to 
    synchronize updates for specific sensor locations.
    """

    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Execution logic for the worker.
        Logic: Atomically initializes spatial locks on-demand and coordinates 
        data exchange with neighboring nodes.
        """
        # Block Logic: Dynamic Mutex Initialization.
        # Note: Non-atomic check-then-set pattern.
        if self.location not in self.device.locks:
            self.device.locks[self.location] = Lock()

        # Critical Section: Network-wide mutual exclusion for the spatial location.
        with self.device.locks[self.location]:
            script_data = []
            
            # Aggregate neighborhood state.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Include local node state.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Apply script logic and propagate results to the neighborhood graph.
                result = self.script.run(script_data)

                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.device.set_data(self.location, result)


from threading import *

class ReusableBarrier():
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Uses a double-gate mechanism with semaphores to prevent 
    fast threads from starting a new cycle before the group has cleared the current one.
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
        """Orchestrates the two-phase synchronization rendezvous."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """Internal gate logic using atomic counter decrement and semaphore release."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Release the gate for the entire group.
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
