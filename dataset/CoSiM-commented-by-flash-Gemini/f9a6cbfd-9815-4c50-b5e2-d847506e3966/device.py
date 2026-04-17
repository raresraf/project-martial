"""
@f9a6cbfd-9815-4c50-b5e2-d847506e3966/device.py
@brief Distributed sensor network simulation with global spatial locking and fork-join compute.
This module implements a coordinated parallel processing framework where node managers 
(DeviceThread) utilize a 'Fork-Join' pattern to execute computational scripts via 
transient threads (ScriptThread). Consistency is guaranteed through a network-wide 
pool of spatial locks (hashset) pre-allocated for every sensor location, while 
temporal alignment across simulation steps is enforced by a robust two-phase 
semaphore barrier.

Domain: Parallel Fork-Join, Distributed State Synchronization, Spatial Mutual Exclusion.
"""

from threading import Event, Thread, Semaphore, Lock

class ReusableBarrier():
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
        """Orchestrates the two-phase arrival and exit protocol."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Internal gate logic using atomic counter decrement and semaphore signaling."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Release the gate for the group.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset counter for next phase/reuse.
                count_threads[0] = self.num_threads
        threads_sem.acquire()
        

class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local data state, coordinates the distribution 
    of global synchronization resources, and provides the interface for task assignment.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        
        # Node-level mutex for protecting local sensor state.
        self.lock = Lock()
        self.locs = []
        # Spatial lock repository: shared network-wide map of location mutexes.
        self.hashset = {}
        # Temporal rendezvous point.
        self.bariera = ReusableBarrier(1)
        
        # Primary node management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node (ID 0) discovers all unique spatial locations and 
        pre-allocates a mutex for each, then distributes the shared barrier.
        """
        if self.device_id == 0:
            self.hashset = {}
            # Discovery: aggregate all locations across the entire network.
            for device in devices:
                for location in device.sensor_data:
                    self.hashset[location] = Lock()
            
            # Atomic setup of shared rendezvous point.
            self.bariera = ReusableBarrier(len(devices))
            
            # Propagation: distribute resources to all peer nodes.
            for device in devices:
                device.bariera = self.bariera
                device.hashset = self.hashset

    def assign_script(self, script, location):
        """Registers a task and signals the orchestration thread."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Finalize assignment phase for current step.
            self.timepoint_done.set()

    def get_data(self, location):
        """Atomic retrieval of local sensor data."""
        self.lock.acquire()
        aux = self.sensor_data[location] if location in self.sensor_data else None
        self.lock.release()
        return aux

    def set_data(self, location, data):
        """Atomic update of local sensor state."""
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

    def shutdown(self):
        """Gracefully joins the node manager thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Node-level simulation manager.
    Functional Utility: Orchestrates simulation phases and implements a fork-join 
    parallel processing model for computational scripts.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        """
        Main execution loop.
        Algorithm: Iterative sequence: 
        Wait -> Topology Refresh -> Fork Workers -> Join Workers -> Consensus.
        """
        while True:
            # Refresh topology.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block until workload assignment for the current step is complete.
            self.device.timepoint_done.wait()
            
            # Phase 1: Fork.
            # Spawns a dedicated thread for every script in the batch.
            list_threads = []
            for (script, location) in self.device.scripts:
                list_threads.append(ScriptThread(self.device, script,
                location, neighbours))
            
            for i in xrange(len(list_threads)):
                list_threads[i].start()
            
            # Phase 2: Join.
            # wait for all local parallel tasks to finalize.
            for i in xrange(len(list_threads)):
                list_threads[i].join()
            
            # Phase Reset.
            self.device.timepoint_done.clear()
            # Global Consensus rendezvous.
            self.device.bariera.wait()

class ScriptThread(Thread):
    """
    Transient computational worker.
    Functional Utility: Executes a computational script while maintaining 
    network-wide spatial mutual exclusion for the target location.
    """

    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Execution Logic.
        Algorithm: Atomic read-modify-write within the global spatial lock.
        """
        # Critical Section: Spatial mutual exclusion across the entire network.
        self.device.hashset[self.location].acquire()
        script_data = []
        
        # Aggregate neighborhood data.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Include local sensor data.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Apply computational logic.
            result = self.script.run(script_data)

            # Propagation: distribute results to all nodes in the neighborhood graph.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)

        # Release spatial mutex.
        self.device.hashset[self.location].release()
