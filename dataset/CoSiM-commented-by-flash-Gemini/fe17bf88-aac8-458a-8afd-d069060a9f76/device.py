"""
@fe17bf88-aac8-458a-8afd-d069060a9f76/device.py
@brief Distributed sensor network simulation with transient fork-join threading.
This module implements a parallel processing framework where computational scripts 
are executed by on-demand worker threads (RunScripts) using a 'Fork-Join' pattern. 
The node manager (DeviceThread) spawns a thread for each task in the current batch 
and waits for global completion before proceeding. Consistency is maintained 
through lazily-initialized shared spatial locks and a two-phase synchronization barrier.

Domain: Fork-Join Parallelism, Dynamic Thread Spawning, Lazy Spatial Locking.
"""

from threading import Thread, Event
from threading import Lock, Semaphore

class ReusableBarrier():
    """
    Two-phase reusable barrier implementation using semaphores.
    Functional Utility: Provides a robust synchronization point for a fixed group 
    of threads, ensuring clean phase transitions through a double-gate mechanism.
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
        """Arrival gate logic: blocks until the group threshold is met."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Release all threads for the current phase.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads      
        self.threads_sem1.acquire()
    
    def phase2(self):
        """Exit gate logic: ensures total group clearance before allowing reuse."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads   
        self.threads_sem2.acquire()

class RunScripts(Thread):                                         
    """
    Transient computational worker.
    Functional Utility: Implements a single script execution with spatial 
    mutual exclusion.
    """
    
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    
    def run(self):
        """
        Execution logic.
        Logic: Atomically acquires the spatial lock for the location, gathers 
        neighborhood data, and propagates results.
        """
        # Critical Section: Spatial mutual exclusion across the entire network.
        self.device.location_lock[self.location].acquire()

        script_data = []
        
        # Aggregate neighborhood data.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Include local node state.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Apply computational logic.
            result = self.script.run(script_data)
            
            # Propagation: distribute results to all peers and self.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)

        # Release spatial mutex.
        self.device.location_lock[self.location].release()

class Device(object):
    """
    Representation of a node in the sensor network simulation.
    Functional Utility: Manages local data state and coordinates the allocation 
    and peer-sharing of global synchronization resources.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()
        # Primary lifecycle management thread.
        self.thread = DeviceThread(self)
        self.barrier = None
        self.list_thread = []
        self.thread.start()
        # Spatial lock pool (max 200 locations).
        self.location_lock = [None] * 200

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Coordinator node initializes and propagates the shared barrier. 
        Aggregates references to all peer nodes to support lock sharing.
        """
        nr_devices = len(devices)
        
        if self.barrier is None:
            # Singleton setup: ensure all nodes share the same rendezvous point.
            barrier = ReusableBarrier(nr_devices)
            self.barrier = barrier

            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        # Shared discovery registry.
        for device in devices:
            if device is not None:
                self.devices.append(device)


    def assign_script(self, script, location):
        """
        Task and Lock management.
        Logic: Lazily initializes spatial mutexes. Searches through peer devices 
        to ensure a shared lock instance is used across the network.
        """
        lock_location = False

        if script is None:
            # Signal end of step workload assignment.
            self.timepoint_done.set()

        else:
            self.scripts.append((script, location))
            # Block Logic: Lazy Spatial Mutex resolution.
            if self.location_lock[location] is None:
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        # Shared reference: use existing lock from a peer.
                        self.location_lock[location] = device.location_lock[location]
                        lock_location = True
                        break

                if lock_location is False:
                    # Fresh initialization.
                    self.location_lock[location] = Lock()

            self.script_received.set()
            

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
    Node-level manager.
    Functional Utility: Orchestrates simulation timepoints and manages the 
    parallel dispatch of tasks using a fork-join worker model.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for node orchestration.
        Algorithm: Iterative sequence: 
        Wait -> Fork Workers -> Join Workers -> Global Barrier.
        """
        while True:
            # Topology Discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block until workload assignment for the current step is complete.
            self.device.timepoint_done.wait()

            # Phase 1: Fork.
            # Spawns a dedicated thread for every script in the current step.
            for (script, location) in self.device.scripts:
                thread = RunScripts(self.device, location, script, neighbours) 
                self.device.list_thread.append(thread)

            for thread_elem in self.device.list_thread:
                thread_elem.start()

            # Phase 2: Join.
            # wait for all local parallel processing to finalize.
            for thread_elem in self.device.list_thread:
                thread_elem.join()

            # Phase Reset and Consensus.
            self.device.list_thread = []
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
