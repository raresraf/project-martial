"""
@fd93fded-17ed-472f-a0cc-538f5f315eb2/device.py
@brief Distributed sensor network simulation with hardware-aware worker pool and hierarchical locking.
This module implements a dynamic processing model where a node manager (DeviceThread) 
orchestrates simulation steps using a pool of persistent threads, sized relative 
to the system's CPU count. Consistency is guaranteed through a sophisticated 
hierarchical locking protocol: worker threads first acquire a global spatial lock 
for a sensor location and then nested node-level mutexes before performing 
read-modify-write operations. Temporal synchronization is enforced via multi-stage 
barrier rendezvous.

Domain: Hardware-Aware Parallelism, Hierarchical Synchronization, Distributed State.
"""

from threading import Event, Lock
from barrier import ReusableBarrierCond
from device_thread import DeviceThread

import multiprocessing

class Device(object):
    """
    Core representation of a node in the distributed simulation.
    Functional Utility: Manages local data state, coordinates leader-based resource 
    discovery, and supervises a persistent parallel worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.neighbours = []
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Persistent task buffer: recycled between simulation timepoints.
        self.scripts_aux = []
        self.timepoint_done = Event()

        # Mutex for protecting atomic task retrieval from scripts_aux.
        self.pop_script_lock = Lock()
        
        # Shared network-wide barrier populated during setup.
        self.devices_barrier = None

        # Node-Level Spatial Locks: protecting individual sensor buckets.
        self.location_locks = {}
        for location in self.sensor_data:
            self.location_locks[location] = Lock()

        # Global Spatial locks: shared network-wide mutex mapping.
        self.global_location_locks = {}

        # Worker Pool setup.
        self.threads = []
        # Optimization: sizes the pool to 4x CPU core count.
        self.number_of_threads = 4 * multiprocessing.cpu_count()

        # Internal local group barrier.
        self.threads_barrier = ReusableBarrierCond(self.number_of_threads)

        for i in range(self.number_of_threads):
            self.threads.append(DeviceThread(self, i))

        for i in range(self.number_of_threads):
            self.threads[i].start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices_barrier(self, barrier):
        """Injects the global network rendezvous point."""
        self.devices_barrier = barrier

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node (ID 0) initializes the network-wide barrier 
        and shares the global spatial lock pool with all peers.
        """
        if self.device_id == 0:
            # Singleton setup: ensures all devices in the group are aligned.
            self.devices_barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.devices_barrier = self.devices_barrier
                    # Share the same lock mapping instance.
                    device.global_location_locks = self.global_location_locks

    def assign_script(self, script, location):
        """Registers a computational task into the node's permanent and transient buffers."""
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_aux.append((script, location))
        else:
            # Signal end of step workload assignment.
            self.timepoint_done.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] \
            if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully joins all persistent worker threads."""
        for i in range(self.number_of_threads):
            self.threads[i].join()


from threading import Thread, Lock

class DeviceThread(Thread):
    """
    Persistent worker thread implementation with role-based coordination.
    Functional Utility: Executes computational scripts while adhering to a 
    complex hierarchical locking strategy to ensure network-wide data consistency.
    """

    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        Main worker execution loop.
        Algorithm: Iterative multi-phase synchronization: 
        Coordination -> Local Barrier -> Parallel Execution -> Consensus.
        """
        while True:
            # Phase 1: Coordination Logic.
            # Logic: only thread 0 per node refreshes simulation metadata.
            if self.thread_id == 0:
                self.device.timepoint_done.clear()
                # Re-population: recycle cached scripts for the new timepoint.
                for script in self.device.scripts:
                    self.device.scripts_aux.append(script)
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # Barrier Point: Ensure topology is refreshed for the entire local group.
            self.device.threads_barrier.wait()

            if self.device.neighbours is None:
                break

            # Block Logic: Workload Consumption Phase.
            # Threads pull tasks from the shared 'scripts_aux' queue until empty.
            while True:
                # Phase Exit check.
                if self.device.timepoint_done.is_set() \
                        and len(self.device.scripts_aux) == 0:
                    break

                # Atomic task acquisition from the shared list.
                self.device.pop_script_lock.acquire()
                if len(self.device.scripts_aux) > 0:
                    (script, location) = self.device.scripts_aux.pop(0)
                    self.device.pop_script_lock.release()
                else:
                    self.device.pop_script_lock.release()
                    continue

                script_data = []

                # Block Logic: Hierarchical Synchronization protocol.
                # Phase A: On-demand initialization of the shared spatial mutex.
                if location not in self.device.global_location_locks:
                    # Note: potential race condition in leaderless setup.
                    self.device.global_location_locks[location] = Lock()

                # Phase B: Global Mutual Exclusion (Network Level).
                with self.device.global_location_locks[location]:
                    
                    # Phase C: Atomic Data Acquisition (Node Level).
                    # Logic: implements a 'sticky lock' where mutexes are held across aggregation.
                    for device in self.device.neighbours:
                        if location in device.sensor_data:
                            device.location_locks[location].acquire()
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)

                    # Self state integration.
                    if self.device not in self.device.neighbours:
                        if location in self.device.sensor_data:
                            self.device.location_locks[location].acquire()
                            data = self.device.get_data(location)
                            if data is not None:
                                script_data.append(data)

                if script_data != []:
                    # Computational Step.
                    result = script.run(script_data)

                    # Phase D: Atomic Result Propagation and Mutex Release.
                    for device in self.device.neighbours:
                        if location in device.sensor_data:
                            device.set_data(location, result)
                            # Release the sticky lock acquired during phase C.
                            device.location_locks[location].release()

                    if self.device not in self.device.neighbours:
                        if location in self.device.sensor_data:
                            self.device.set_data(location, result)
                            self.device.location_locks[location].release()

            # Local Rendezvous: Ensure local group finished the step.
            self.device.threads_barrier.wait()

            # Global Consensus: Coordinator thread waits for the entire network group.
            if self.thread_id == 0:
                self.device.devices_barrier.wait()
