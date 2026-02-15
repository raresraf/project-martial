"""
This module defines a distributed device simulation framework where each device
is managed by a single controller thread that processes its assigned scripts
sequentially for each time step.
"""

from threading import Event, Semaphore, Lock, Thread

class Device(object):
    """
    Represents a device node in the simulation.
    
    Each device is controlled by a single `DeviceThread`. It holds the device's
    local data and references to synchronization objects shared among all devices.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device and starts its controller thread.
        
        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The local data store for this device.
            supervisor (Supervisor): The central simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event() # Signals the controller to start a step.
        self.data_lock = Lock()             # A per-device lock.
        self.barrier = None                 # The global time step barrier.
        self.location_locks = {}            # Shared dictionary of per-location locks.
        self.thread = DeviceThread(self)

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects.
        
        The first device to call this (device_id 0) creates the shared barrier
        and location lock dictionary and distributes them to all other devices.
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
        Assigns a script to the device. A `None` script signals the end of
        assignments for the current step, triggering the `timepoint_done` event.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data from the device's local sensor data.
        Note: This method is not thread-safe; locking is handled by the caller.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Sets data in the device's local sensor data.
        Note: This method is not thread-safe; locking is handled by the caller.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's controller thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main controller thread for a device, which sequentially executes all
    assigned scripts for a given time step.
    """
    
    def __init__(self, device):
        """Initializes the controller thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        while True:
            # Get neighbors for the current step. If None, simulation is over.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to signal that all scripts for this step are assigned.
            self.device.timepoint_done.wait()       
            self.device.timepoint_done.clear()      

            # Process all assigned scripts sequentially in this single thread.
            for (script, location) in self.device.scripts:
                # Lazy-initialize a lock for the location if not already present.
                if location not in self.device.location_locks:      
                    self.device.location_locks[location] = Lock()   
                
                # Acquire the global lock for this location to serialize work on it.
                self.device.location_locks[location].acquire()      
                
                script_data = []
                # Gather data from neighbors and self.
                for device in neighbours:
                    data = device.get_data(location)            
                    if data is not None:                        
                        script_data.append(data)                
                data = self.device.get_data(location)           
                if data is not None:                            
                    script_data.append(data)                    

                # If data was found, run script and distribute results.
                if script_data:
                    result = script.run(script_data)            
                    # This section uses a per-device data_lock inconsistently.
                    for device in neighbours:
                        device.data_lock.acquire()              
                        device.set_data(location, result)       
                        device.data_lock.release()              
                    self.device.data_lock.acquire()             
                    self.device.set_data(location, result)      
                    self.device.data_lock.release()             

                # Release the global lock for the location.
                self.device.location_locks[location].release()  

            # Synchronize with all other devices before starting the next step.
            self.device.barrier.wait()          

# The following ReusableBarrier class seems to be from an external file.
from threading import *

class ReusableBarrier():
    """A custom reusable barrier implemented using semaphores and a lock."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        """
        Blocks until all participating threads have called this method.
        Uses a two-phase system to ensure reusability.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
    
    def phase(self, count_threads, threads_sem):
        """Executes one of the two barrier phases."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for _ in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    
                                                 
# This MyThread class appears to be an unused example.
class MyThread(Thread):
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
    
    def run(self):
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",