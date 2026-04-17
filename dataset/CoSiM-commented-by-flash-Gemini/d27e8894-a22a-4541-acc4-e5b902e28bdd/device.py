"""
@d27e8894-a22a-4541-acc4-e5b902e28bdd/device.py
@brief Distributed sensor network simulation with fork-join parallel computation.
This module implements a hybrid execution model where data orchestration (aggregation 
and propagation) is handled sequentially by a node manager, while the core 
computational logic is offloaded to a parallel pool of transient threads (Node). 
The system utilizes a 'Fork-Join' pattern to maximize CPU utilization during the 
processing phase and maintains network-wide consistency through a two-phase 
semaphore-based barrier.

Domain: Fork-Join Parallelism, Hybrid Orchestration, Two-Phase Barriers.
"""

from threading import enumerate, Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """
    Two-phase reusable barrier implementation using semaphores.
    Functional Utility: Provides a robust synchronization point for a fixed group 
    of threads, ensuring total network alignment before simulation steps proceed.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: Number of participating threads.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()       
        self.threads_sem1 = Semaphore(0) 
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        """Orchestrates the two-phase rendezvous protocol."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """Arrival gate logic: Blocks until all participants reach the threshold."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Release the gate for the group.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            # Reset secondary counter for exit phase.
            self.count_threads2 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Exit gate logic: Ensures clean group transition and barrier reset."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            # Reset primary counter for future reuse.
            self.count_threads1 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local sensor state and coordinates the 
    distribution of the shared synchronization barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.script_received = Event()
        self.scripts = []
        # Main lifecycle management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global resource factory.
        Logic: Node 0 initializes the shared barrier which is then 
        propagated to all other members of the group.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            # Participant Discovery: find the shared barrier from Node 0.
            for device in devices:
                if device.device_id == 0:
                    self.barrier = device.barrier

    def assign_script(self, script, location):
        """Registers a computational task and signals the orchestration thread."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully terminates the orchestration thread."""
        self.thread.join()

class Node(Thread):
    """
    Transient computational worker.
    Functional Utility: Implements the 'fork' part of the pattern, executing 
    a single script in an isolated thread context.
    """

    def __init__(self, script, script_data):
        Thread.__init__(self)
        self.script = script
        self.script_data = script_data
        self.result = None
         
    def run(self):
        """Executes the core domain logic of the script."""
        self.result = self.script.run(self.script_data)

    def join(self):
        """
        Implements the 'join' part of the pattern.
        Functional Utility: Finalizes execution and returns the computation result.
        """
        Thread.join(self)
        return (self.script, self.result)


class DeviceThread(Thread):
    """
    Main node orchestration thread.
    Functional Utility: Manages the sequence of simulation timepoints, 
    coordinating between sequential data handling and parallel computation.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main simulation execution loop.
        Algorithm: Iterative sequence: 
        Sequential Aggregate -> Parallel Fork -> Join -> Sequential Propagate.
        """
        while True:
            # Topology Discovery Phase.
            neighbours = self.device.supervisor.get_neighbours()
            thread_list=[]
            scripts_result = {}
            scripts_data = {}
            if neighbours is None:
                break

            # Block until work assignment phase is complete.
            self.device.script_received.wait()
            self.device.script_received.clear()
            
            # Phase 1: Sequential Data Aggregation.
            # Gathers neighborhood state to prepare task inputs.
            for (script, location) in self.device.scripts:
                script_data = []
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                scripts_data[script] = script_data
                if script_data != []:
                    # Functional Utility: Prepares for parallel offloading.
                    nod = Node(script,script_data)
                    thread_list.append(nod)
            
            # Phase 2: Parallel Computation (Fork).
            for nod in thread_list:
                nod.start()
            
            # Phase 3: Consolidation (Join).
            for nod in thread_list:
                key ,value = nod.join()
                scripts_result[key] = value
            
            # Phase 4: Sequential Data Propagation.
            # Distributes results back to the neighborhood graph.
            for (script, location) in self.device.scripts:
                if scripts_data[script] != []:
                    for device in neighbours:
                        device.set_data(location, scripts_result[script])
                    self.device.set_data(location, scripts_result[script])
            
            # Global Consensus Point.
            self.device.barrier.wait()
