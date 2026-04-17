"""
@f6c21ac8-d7f4-41ea-bf6b-701babf5bc20/device.py
@brief Distributed sensor network simulation with lazy spatial locking and node-level orchestration.
This module implements a parallel processing framework where individual nodes are 
managed by dedicated orchestration threads. Consistency is guaranteed through a 
shared, lazily-initialized pool of spatial locks that ensure network-wide mutual 
exclusion for specific sensor locations. Global temporal consensus is enforced 
via a two-phase semaphore-based synchronization barrier.

Domain: Coordinated Simulation, Lazy Mutex Initialization, Two-Phase Barriers.
"""

from threading import Event, Thread, Lock
from utils import ReusableBarrier


class Device(object):
    """
    Representation of a node in the sensor network simulation.
    Functional Utility: Manages local data state, coordinates the discovery of 
    shared synchronization resources, and provides the interface for simulation setup.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        # Primary node management thread.
        self.thread = DeviceThread(self, 0)

        # Global Synchronization resources populated during setup.
        self.common_barrier = None
        # Rendezvous event for network-wide initialization.
        self.wait_initialization = Event()

        # Shared repository of spatial mutexes.
        self.locations_locks = None
        # Mutex for protecting the spatial lock dictionary itself.
        self.lock_location_dict = Lock()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Node 0 acts as a resource factory, initializing the shared barrier 
        and lock repository. Non-zero nodes wait for this initialization to finalize 
        before starting their management threads.
        """
        if not self.device_id == 0:
            # Participant node: block until coordinator finishes setup.
            self.wait_initialization.wait()
            self.thread.start()
        else:
            # Coordinator node: initialize and distribute shared resources.
            self.locations_locks = {}
            self.common_barrier = ReusableBarrier(len(devices))

            # Propagation: share references and signal completion to all peer nodes.
            for dev in devices:
                dev.common_barrier = self.common_barrier
                dev.locations_locks = self.locations_locks
            for dev in devices:
                if not dev.device_id == 0:
                    dev.wait_initialization.set()

            self.thread.start()

    def assign_script(self, script, location):
        """Registers a computational task for the current simulation step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Finalize assignment phase.
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
    Main node orchestration thread.
    Functional Utility: Manages simulation timepoints and implements a lazy-locking 
    strategy to synchronize computational script execution.
    """

    def __init__(self, device, th_id):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.th_id = th_id

    def run(self):
        """
        Main execution loop.
        Algorithm: Iterative sequence: Wait for step -> Fetch Topology -> Task Execution -> Consensus.
        """
        while True:
            # Phase 1: Global rendezvous.
            self.device.common_barrier.wait()

            # Phase 2: Role-based topology refresh.
            if self.th_id == 0:
                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    break
            else:
                # Reserved for multi-threaded nodes (not used in current config).
                pass

            # Phase 3: Wait for workload assignment.
            self.device.timepoint_done.wait()

            current_scripts = self.device.scripts

            # Phase 4: Sequential Task Execution.
            for (script, location) in current_scripts:
                # Block Logic: Lazy Spatial Mutex Initialization.
                # Logic: uses double-locking to ensure thread-safe creation of the location lock.
                self.device.lock_location_dict.acquire()

                if not self.device.locations_locks.has_key(location):
                    self.device.locations_locks[location] = Lock()

                # Critical Section: Network-wide spatial mutual exclusion.
                self.device.locations_locks[location].acquire()
                # release the dict lock to allow other locations to be initialized.
                self.device.lock_location_dict.release()

                script_data = []
                # Aggregate neighborhood data.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Include local node data.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Compute result and propagate to all nodes in the neighborhood graph.
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release spatial mutex.
                self.device.locations_locks[location].release()

            # Cleanup for next step.
            self.device.timepoint_done.clear()

from threading import Semaphore, Lock


class ReusableBarrier(object):
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Implements a double-gate mechanism with semaphores to 
    ensure total temporal alignment across simulation cycles.
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
        """Executes the two-phase arrival and exit sequence."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Internal gate logic using atomic counter decrement and semaphore signaling."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Threshold reached: release the gate.
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
