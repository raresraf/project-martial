"""
@e79b0a4c-e77d-4b9c-96b8-66d2e58f4070/device.py
@brief Distributed sensor processing simulation using a persistent coordinator thread and hierarchical locking.
* Algorithm: Event-driven execution loop with multi-level mutual exclusion (Global Location Locks + Local Device Locks) and two-phase semaphore barriers.
* Functional Utility: Orchestrates simulation timepoints across a network of devices by managing neighbor data aggregation and synchronized state updates using a centralized resource coordinator per node.
"""

from threading import Event, Thread, Lock               
from reusable_barrier_semaphore import ReusableBarrier  

class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings, coordination state, and shared synchronization infrastructure.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device state and prepares its internal coordinator thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event() # Intent: Signals completion of task delivery for the current phase.
        self.data_lock = Lock()       # Intent: Serializes local sensor data updates from peers.
        self.barrier = None           # Intent: Global barrier for cluster-wide alignment.
        self.location_locks = {}      # Intent: Registry of shared locks for specific sensor locations.
        self.thread = DeviceThread(self)

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global initialization and distribution of shared simulation resources.
        Invariant: Root node (ID 0) initializes the collective barrier and propagates its local lock registry.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))        
            for device in devices:
                if device.device_id != self.device_id:          
                    device.barrier = self.barrier               
                    device.location_locks = self.location_locks 
        self.thread.start()                                     

    def assign_script(self, script, location):
        """
        @brief Receives a processing task for the current simulation cycle.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Logic: Signals completion of script batch arrival for this node.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Standard data retrieval interface for local sensor readings.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Standard data update interface for local sensor readings.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Terminates the device coordination thread.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief main management thread coordinating timepoint progression and script execution.
    """
    
    def __init__(self, device):
        """
        @brief Initializes the coordinator thread for a specific device node.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core lifecycle loop of the device node coordination.
        Algorithm: Iterative execution with busy-wait for scripts and barrier alignment.
        """
        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Ensures all tasks for the current cycle have arrived.
            self.device.timepoint_done.wait()       
            self.device.timepoint_done.clear()      

            # Task Execution Phase.
            for (script, location) in self.device.scripts:
                # Logic: Lazy initialization of shared locks for new sensor locations.
                if location not in self.device.location_locks:      
                    self.device.location_locks[location] = Lock()   
                
                # Pre-condition: Acquire shared global location lock for atomic distributed update.
                self.device.location_locks[location].acquire()      
                script_data = []

                # Distributed Aggregation: Collect readings from neighbors and self.
                for device in neighbours:
                    data = device.get_data(location)            
                    if data is not None:                        
                        script_data.append(data)                
                
                data = self.device.get_data(location)           
                if data is not None:                            
                    script_data.append(data)                    

                if script_data != []:
                    # Execution and Propagation Phase.
                    result = script.run(script_data)            
                    for device in neighbours:
                        # Logic: Cluster-wide update under both location and per-device data locks.
                        device.data_lock.acquire()              
                        device.set_data(location, result)       
                        device.data_lock.release()              
                    
                    self.device.data_lock.acquire()             
                    self.device.set_data(location, result)      
                    self.device.data_lock.release()             

                # Post-condition: Release global location lock.
                self.device.location_locks[location].release()  

            # Synchronization Phase: Align all devices across the cluster.
            self.device.barrier.wait()          

from threading import *

class ReusableBarrier():
    """
    @brief Implementation of a two-phase synchronization barrier using semaphores.
    * Algorithm: Dual-stage arrival pattern to ensure strict thread alignment.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Intent: Shared mutable counter.
        self.count_threads2 = [self.num_threads] # Intent: Shared mutable counter.
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        """
        @brief Synchronizes the calling thread through both phases of the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
    
    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single synchronization stage.
        Invariant: The last thread to arrive releases the entire group.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    
                                                 

class MyThread(Thread):
    """
    @brief Simple worker thread for barrier validation.
    """
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
    
    def run(self):
        """
        @brief Demonstration loop for synchronized step progression.
        """
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",
