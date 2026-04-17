"""
@c8ef764d-3142-4cb8-867b-cc582c467232/device.py
@brief Distributed sensor network simulation with throttled dynamic thread spawning.
This module implements a concurrent processing model where computational scripts are 
executed by on-demand threads. Throttling is achieved via node-local semaphores to 
manage system resources, while network-wide consistency is enforced through a 
centralized map of spatial locks and a robust two-phase synchronization barrier.

Domain: Dynamic Thread Management, Resource Throttling, Spatial Mutual Exclusion.
"""

from threading import Semaphore, Event, Lock, Thread

class ReusableBarrier(object):
    """
    Two-phase reusable barrier implementation using semaphores.
    Functional Utility: Implements a double-gate rendezvous mechanism that prevents 
    fast threads from starting a new cycle before slow threads have exited the current one.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: Number of participants in the synchronization group.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Executes the full two-phase synchronization cycle."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """Arrival phase: Blocks threads until the threshold is reached."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Release all blocked threads at once.
                for _ in xrange(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Exit phase: Ensures total group clearance before allowing reuse."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in xrange(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Coordinator entity for a network node.
    Functional Utility: Manages local data state and orchestrates the distribution 
    of shared synchronization resources (barriers and spatial locks).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.none_script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.timepoint_end = 0
        self.barrier = None
        self.lock_hash = None

    def __str__(self):
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """Injects the shared network-wide barrier."""
        self.barrier = barrier

    def set_locks(self, lock_hash):
        """Injects the global mapping of sensor location locks."""
        self.lock_hash = lock_hash

    def setup_devices(self, devices):
        """
        Global synchronization initialization.
        Logic: Selects the node with the minimum ID to act as the global resource 
        factory, creating the shared barrier and a mutex for every sensor location.
        """
        ids_list = []
        for dev in devices:
            ids_list.append(dev.device_id)

        # Leader Election: ensures atomic initialization of global state.
        if self.device_id == min(ids_list):
            self.barrier = ReusableBarrier(len(devices))
            self.lock_hash = {}

            # Discover all unique spatial locations and create corresponding locks.
            for dev in devices:
                for location in dev.sensor_data:
                    if location not in self.lock_hash:
                        self.lock_hash[location] = Lock()

            # Propagation: distribute resources to all peers.
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.set_barrier(self.barrier)
                    dev.set_locks(self.lock_hash)


    def assign_script(self, script, location):
        """Queues a computational task and signals when the assignment phase is over."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.none_script_received.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor value."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main management thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Main orchestration thread for the node.
    Functional Utility: Manages simulation timepoints and throttles task parallelization 
    using a local semaphore to bound thread creation.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Throttling Mechanism: limits local concurrency to 8 worker threads.
        self.semaphore = Semaphore(value=8)

    def run(self):
        """
        Main execution loop.
        Algorithm: Iterative worker spawning with barrier-based consensus.
        """
        while True:
            # Refresh topology.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait until the supervisor indicates no more scripts are coming for this step.
            self.device.none_script_received.wait()
            self.device.none_script_received.clear()

            thread_list = []
            # Block Logic: Spawns a dedicated thread for every script in the batch.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, neighbours, script, location,
                    self.semaphore)
                thread.start()
                thread_list.append(thread)

            # Wait for all local workers to complete their tasks.
            for i in xrange(len(thread_list)):
                thread_list[i].join()

            # Global Synchronization rendezvous.
            self.device.barrier.wait()

class MyThread(Thread):
    """
    Worker thread implementation.
    Functional Utility: Executes a script while maintaining both local resource 
    bounds and global spatial consistency.
    """

    def __init__(self, device, neighbours, script, location, semaphore):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.semaphore = semaphore

    def run(self):
        """
        Execution logic.
        Logic: Implements a nested locking strategy (Local Semaphore -> Global Spatial Lock).
        """
        # Phase 1: Local throttling.
        self.semaphore.acquire()

        # Phase 2: Global spatial mutual exclusion.
        self.device.lock_hash[self.location].acquire()

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
            # Apply script domain logic and propagate updates.
            result = self.script.run(script_data)

            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)

        # Ordered release of global lock and local semaphore slot.
        self.device.lock_hash[self.location].release()
        self.semaphore.release()
