"""
@ff8e2e84-ed62-427a-9c7b-bb8c3497022c/device.py
@brief Distributed sensor network simulation with batch-oriented fork-join compute.
This module implements a coordinated processing framework where node managers 
(DeviceThread) execute computational tasks in parallel batches of 8 transient 
threads (ScriptThread). Results are aggregated in a local cache before being 
propagated to the neighborhood graph to ensure atomicity at the simulation step 
level. Consistency is guaranteed through shared re-entrant locks and a reusable 
synchronization barrier.

Domain: Batch Parallelism, Fork-Join Patterns, Atomic State Propagation.
"""

from threading import Condition, Event, RLock, Thread

class ReusableBarrier(object):
    """
    Monitor-based reusable barrier implementation.
    Functional Utility: Provides a temporal rendezvous point for a fixed group 
    of threads using a condition variable to signal threshold arrival.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: participants count.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def reinit(self):
        """
        Dynamic resizing of the barrier.
        Logic: Decrements the thread threshold and wait for the remaining group.
        """
        self.cond.acquire()
        self.num_threads -= 1
        self.cond.release()
        self.wait()

    def wait(self):
        """Blocks the caller until the arrival threshold is reached."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Threshold reached: release all participants and reset for next cycle.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Core network node representation.
    Functional Utility: Manages local data state, coordinates the distribution 
    of shared re-entrant locks, and provides a container for assigned parallel tasks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.start = Event()
        self.scripts = []
        # Transient list of tasks for the current simulation step.
        self.scripts_to_process = []
        self.timepoint_done = Event()
        self.nr_script_threats = 0
        
        # Primary node management thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.script_threats = []
        
        # Shared Synchronization resources.
        self.barrier_devices = None
        self.neighbours = None
        # Degree of local parallelism.
        self.cors = 8
        # Global locks shared across peer devices.
        self.lock = None
        self.lock_self = None
        
        # Intermediate result storage.
        self.results = {}
        self.results_lock = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Implements a 'first-come-first-served' pattern to initialize and 
        distribute shared re-entrant locks and barriers across the entire group.
        """
        for script in self.scripts:
            # Note: access to scripts_to_process is guarded but 'lock' might be None here.
            if self.lock:
                self.lock.acquire()
                self.scripts_to_process.append(script)
                self.lock.release()

        # Block Logic: Leaderless Shared Resource Initialization.
        # Logic: every node attempts to create and propagate the singleton locks.
        if not self.lock_self:
            lock = RLock()
            for device in devices:
                device.lock_self = lock

        self.lock_self.acquire()
        if not self.lock:
            rlock = RLock()
            for device in devices:
                device.lock = rlock
        self.lock_self.release()

        self.lock_self.acquire()
        if not self.results_lock:
            results_lock = RLock()
            for device in devices:
                device.results_lock = results_lock
        self.lock_self.release()

        self.lock_self.acquire()
        if not self.barrier_devices:
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier_devices = barrier
                device.start.set()
        self.lock_self.release()



    def assign_script(self, script, location):
        """Registers a task and signals completion of the simulation step assignment."""
        if script is not None:
            self.lock.acquire()
            self.scripts.append((script, location))
            self.scripts_to_process.append((script, location))
            self.script_received.set()
            self.lock.release()
        else:
            self.lock.acquire()
            self.timepoint_done.set()
            self.script_received.set()
            self.lock.release()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        if location in self.sensor_data:
            data = self.sensor_data[location]
        else:
            data = None
        return data

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data
        
    def shutdown(self):
        """Gracefully joins the orchestration thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Node-level simulation manager.
    Functional Utility: Orchestrates simulation phases and implements a 
    windowed distributor for transient computational threads.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.device.neighbours = None

    def run(self):
        """
        Main execution loop for node orchestration.
        Algorithm: Iterative sequence: 
        Wait -> Topology Refresh -> Windowed Task Dispatch -> Post-Execution Commit -> Consensus.
        """
        # Block until global network initialization is complete.
        self.device.start.wait()
        while True:
            # Refresh Task Stack.
            self.device.scripts_to_process = []
            for script in self.device.scripts:
                self.device.scripts_to_process.append(script)

            # Topology Discovery Phase.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                # Termination sequence: adjust the global barrier to account for exit.
                self.device.barrier_devices.reinit()
                break

            self.device.results = {}
            # Block Logic: Simulation Phase Execution.
            while True:
                # Wait for workload arrival.
                if not self.device.timepoint_done.is_set():
                    self.device.script_received.wait()
                    self.device.script_received.clear()

                # Phase Check: exit internal loop when workload is drained.
                if len(self.device.scripts_to_process) == 0:
                    if self.device.timepoint_done.is_set():
                        break

                # Block Logic: Windowed Task Parallelization.
                # Logic: Spawns computational workers in batches of 8.
                while len(self.device.scripts_to_process):
                    list_threats = []
                    self.device.script_threats = []
                    self.device.nr_script_threats = 0
                    
                    # Window Slicing.
                    while len(self.device.scripts_to_process) and self.device.nr_script_threats < self.device.cors:
                        script, location = self.device.scripts_to_process.pop(0)
                        list_threats.append((script, location))
                        self.device.nr_script_threats += 1

                    # Fork: Spawn workers for the current window.
                    for script, location in list_threats:
                        script_data = []
                        
                        neighbours = self.device.neighbours
                        # Aggregate neighborhood data under re-entrant locks.
                        for device in neighbours:
                            device.lock_self.acquire()
                            data = device.get_data(location)
                            device.lock_self.release()
                            if data is not None:
                                script_data.append(data)
                        
                        # Include local sensor data.
                        self.device.lock_self.acquire()
                        data = self.device.get_data(location)
                        self.device.lock_self.release()
                        if data is not None:
                            script_data.append(data)

                        thread_script_d = ScriptThread(self.device, script, location, script_data)

                        self.device.script_threats.append(thread_script_d)
                        thread_script_d.start()

                    # Join: Synchronize on window completion.
                    for thread in self.device.script_threats:
                        thread.join()

            # Block Logic: Commit Phase.
            # Algorithm: Atomic propagation of cached batch results to the neighborhood graph.
            for location, result in self.device.results.iteritems():
                for device in self.device.neighbours:
                    device.lock_self.acquire()
                    device.set_data(location, result)
                    device.lock_self.release()
                
                self.device.lock_self.acquire()
                self.device.set_data(location, result)
                self.device.lock_self.release()

            # Global Consensus rendezvous.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            self.device.barrier_devices.wait()

class ScriptThread(Thread):
    """
    Transient worker thread implementation.
    Functional Utility: Executes a computational script and stores results 
    in a node-level shared dictionary.
    """

    def __init__(self, device, script, location, script_data):
        Thread.__init__(self, name="Device Script Thread %d" % device.device_id)
        self.device = device
        self.location = location
        self.script = script
        self.script_data = script_data

    def run(self):
        """
        Execution logic.
        Algorithm: Compute -> Cache result.
        """
        if self.script_data != []:
            # Apply computational logic to pre-aggregated data.
            result = self.script.run(self.script_data)
            
            # Atomic result caching.
            self.device.results_lock.acquire()
            self.device.results[self.location] = result
            self.device.results_lock.release()
            
        # Signal completion to node orchestrator.
        self.device.nr_script_threats -= 1
