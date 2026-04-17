"""
@e79b0a4c-e77d-4b9c-96b8-66d2e58f4070/device.py
@brief Distributed sensor network simulation with sequential task processing and hierarchical locking.
This module implements a coordinated processing framework where computational scripts 
are executed sequentially by a node manager (DeviceThread). Consistency is guaranteed 
through a hierarchical locking model: a global spatial lock ensures network-wide 
mutual exclusion for specific sensor locations, while node-level mutexes protect 
individual data updates. Temporal synchronization is enforced via a two-phase barrier.

Domain: Sequential Task Orchestration, Hierarchical Locking, Distributed Consensus.
"""

from threading import Event, Thread, Lock               
from reusable_barrier_semaphore import ReusableBarrier  

class Device(object):
    """
    Core network node representation.
    Functional Utility: Manages local data state and coordinates the distribution 
    of global synchronization resources (barrier and spatial lock pool).
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        # Mutex for protecting local data state.
        self.data_lock = Lock()             
        self.barrier = None                 
        # Global spatial lock mapping.
        self.location_locks = {}            
        # Main lifecycle management thread.
        self.thread = DeviceThread(self)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization resource allocation.
        Logic: Node 0 initializes the shared barrier and distributes the 
        global spatial lock repository to all peer devices.
        """
        if self.device_id == 0:
            # Singleton setup: ensure all nodes share the same rendezvous point.
            self.barrier = ReusableBarrier(len(devices))        
            for device in devices:
                if device.device_id != self.device_id:          
                    device.barrier = self.barrier               
                    device.location_locks = self.location_locks 
        
        # Activate node management.
        self.thread.start()                                     

    def assign_script(self, script, location):
        """Registers a computational task and signals completion of the assignment phase."""
        if script is not None:
            self.scripts.append((script, location))
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
    Node-level lifecycle manager.
    Functional Utility: Orchestrates simulation timepoints and sequentially 
    executes assigned computational scripts using nested locking.
    """
    
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main simulation loop.
        Algorithm: Iterative sequence of topology refresh, task execution, and consensus.
        """
        while True:
            # Topology Discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block until supervisor indicates step readiness.
            self.device.timepoint_done.wait()       
            self.device.timepoint_done.clear()      

            # Block Logic: Task Execution Phase.
            # Logic: Iterates through assigned scripts, acquiring network-wide spatial locks.
            for (script, location) in self.device.scripts:
                # Lazy Mutex Initialization: ensures a spatial lock exists for the location.
                if location not in self.device.location_locks:      
                    self.device.location_locks[location] = Lock()   
                
                # Critical Section 1: Global spatial mutual exclusion.
                self.device.location_locks[location].acquire()      
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
                    # Compute result.
                    result = script.run(script_data)            
                    
                    # Atomic Result Propagation.
                    # Logic: uses per-device mutexes to ensure memory consistency during updates.
                    for device in neighbours:
                        device.data_lock.acquire()              
                        device.set_data(location, result)       
                        device.data_lock.release()              
                    
                    self.device.data_lock.acquire()             
                    self.device.set_data(location, result)      
                    self.device.data_lock.release()             

                # Release the spatial lock.
                self.device.location_locks[location].release()  

            # Global Rendezvous point for network-wide consensus.
            self.device.barrier.wait()          

from threading import *

class ReusableBarrier():
    """
    Two-phase reusable barrier implementation.
    Functional Utility: Uses a double-gate mechanism with semaphores to ensure 
    temporal alignment and prevents thread overtaking between simulation steps.
    """
    def __init__(self, num_threads):
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
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    
                                                 

class MyThread(Thread):
    """
    Diagnostic thread for barrier validation.
    """
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
    
    def run(self):
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",
