"""
@fa05758e-e81e-41dd-9070-0d0a2a6e7b33/device.py
@brief Distributed sensor network simulation with transient fork-join threading.
This module implements a parallel processing framework where node managers 
(DeviceThread) utilize a 'Fork-Join' model to dispatch computational tasks to 
transient worker threads (MyScriptThread). Consistency is guaranteed through 
node-level mutexes (my_lock) that protect data updates during neighborhood 
state propagation. Global temporal alignment across simulation cycles is 
enforced via a robust two-phase semaphore barrier.

Domain: Parallel Fork-Join, Node-Level Mutex, Two-Phase Barriers.
"""

from threading import Event, Semaphore, Lock, Thread



class ReusableBarrierSem(object):
    """
    Two-phase reusable barrier implementation using semaphores.
    Functional Utility: Provides a temporal rendezvous point for a fixed group 
    of threads, ensuring total network alignment before simulation steps proceed.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier.
        @param num_threads: participants count.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Orchestrates the two-phase arrival and exit protocol."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """Arrival gate logic: blocks until the threshold is reached."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Release the gate for the entire group.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Exit gate logic: ensures group clearance before reset."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Representation of a node in the sensor network simulation.
    Functional Utility: Manages local data state, coordinates global synchronization 
    resource distribution, and provides the interface for task assignment.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        
        # Node-level mutex for protecting data state updates.
        self.my_lock = Lock()
        # Barrier sized during setup_devices.
        self.barrier = ReusableBarrierSem(0)
        self.timepoint_done = Event()
        # Primary node management thread.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node (ID 0) initializes and distributes the shared barrier.
        """
        if self.device_id == 0:
            # Singleton setup: ensure all nodes share the same barrier instance.
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """Registers a task and signals the orchestration thread."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Finalize workload for current step.
            self.script_received.set()
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



class MyScriptThread(Thread):
    """
    Transient computational worker thread.
    Functional Utility: Executes a script while maintaining node-level 
    mutual exclusion during state updates.
    """
    
    def __init__(self, script, location, device, neighbours):
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        Execution logic.
        Algorithm: Aggregate -> Compute -> Atomic Propagate.
        """
        script_data = []
        # Aggregate neighborhood state.
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

            # Atomic State Propagation.
            # Logic: uses per-device mutexes to ensure memory consistency during updates.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            # Self-update.
            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()

class DeviceThread(Thread):
    """
    Node-level simulation orchestrator.
    Functional Utility: Manages simulation steps and implements a fork-join 
    parallel processing model for assigned computational tasks.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for node orchestration.
        Algorithm: Iterative sequence: Wait -> Fork -> Join -> Consensus.
        """
        while True:
            # Topology Discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break;
            
            # Phase 1: Temporal Rendezvous.
            self.device.barrier.wait()

            # Block until assignment phase for the step is complete.
            self.device.script_received.wait()
            
            # Phase 2: Fork.
            # Spawns a transient thread for every script in the current batch.
            script_threads = []
            for (script, location) in self.device.scripts:
                script_threads.append(MyScriptThread(script,
                    location, self.device, neighbours))
            for thread in script_threads:
                thread.start()
            
            # Phase 3: Join.
            # wait for all local parallel processing to finalize.
            for thread in script_threads:
                thread.join()
            
            # Phase Reset and Consensus.
            self.device.timepoint_done.wait()
            self.device.barrier.wait()
            self.device.script_received.clear()
