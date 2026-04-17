"""
@c141d797-6b52-4010-b9e0-9ff8b203bba9/device.py
@brief Distributed sensor network simulation with shared-counter task distribution.
This implementation employs a multi-threaded architecture (8 threads per node) where 
workers compete for tasks using an atomic index reservation system. It utilizes 
class-level static resources for network-wide synchronization and designates an 
'initiator' thread to manage topology updates and simulation phase transitions.

Domain: Parallel Task Scheduling, Distributed Consensus, Class-level Shared State.
"""

from threading import Event, Thread, Condition, Lock

class Barrier(object):
    """
    Monitor-based reusable barrier implementation.
    Functional Utility: Provides a synchronization point for a fixed number of threads 
    using a condition variable to signal when the threshold is met.
    """
    def __init__(self, num_threads=0):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
 
    def wait(self):
        """Blocks the calling thread until all participants have arrived."""
        self.cond.acquire()                      
        self.count_threads -= 1;
        if self.count_threads == 0:
            # Last thread triggers release and resets for the next cycle.
            self.cond.notify_all()               
            self.count_threads = self.num_threads    
        else:
            self.cond.wait();                    
        self.cond.release(); 

        
class Device(object):
    """
    Node coordinator for the sensor network.
    Functional Utility: Manages local data and provides the interface for network-wide 
    synchronization through static class-level barriers and locks.
    """
    
    # Static Network-Wide Synchronization Primitives.
    DeviceBarrier = Barrier()
    DeviceLocks = []

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and its worker thread pool.
        @param device_id: Unique integer identifier.
        @param sensor_data: Initial state of local sensors.
        @param supervisor: Topology and control entity.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        self.script_received = Event()
        self.scripts = []
        self.locations = []
        
        # Local state for task tracking.
        self.currentScript = 0
        self.scriptNumber = 0
        self.timepoint_done = Event()

        self.neighbours = []
        self.neighbours_event = Event()
        self.lockScript = Lock()
        # Local barrier for internal thread pool synchronization.
        self.barrier = Barrier(8)
        
        # Spawns 8 threads: 1 initiator and 7 workers.
        self.thread = DeviceThread(self, True)
        self.thread.start()
        self.threads = []
        
        for _ in range(7):
            newThread = DeviceThread(self, False)
            self.threads.append(newThread)
            newThread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes global static resources.
        Logic: Configures a network-wide barrier and populates the lock pool 
        based on the total number of sensor locations.
        """
        size = len(devices)
        Device.DeviceBarrier = Barrier(size)
        if Device.DeviceLocks==[]:
            self.updateLocks()

    def getNeighbours(self):
        """Introspection helper to determine the scope of the sensor field."""
        return self.supervisor.supervisor.testcase.num_locations

    def updateLocks(self):
        """Pre-allocates a pool of mutual exclusion locks for all sensor locations."""
        for _ in range(self.getNeighbours()):
            Device.DeviceLocks.append(Lock())

    def assign_script(self, script, location):
        """Queues a task for the current processing cycle."""
        if script is not None:
            self.scripts.append(script)
            self.locations.append(location)
            self.scriptNumber += 1
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for the given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates local sensor state."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully terminates the entire local thread pool."""
        self.thread.join()
        for myThread in self.threads:
            myThread.join()


class DeviceThread(Thread):
    """
    Worker implementation for a Device node.
    Functional Utility: Participates in a shared-counter task distribution 
    to process scripts while maintaining global and local consistency.
    """

    def __init__(self, device, isInitiator):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.isInitiator = isInitiator
      
    def neighboursOperation(self):
        """Orchestrates neighborhood topology discovery at the start of a cycle."""
        self.device.neighbours = self.device.supervisor.get_neighbours()
        self.device.neighbours_event.set()
        self.device.currentScript = 0

    def reserve(self):
        """
        Atomic task reservation logic.
        @return: The index of the next task to be processed.
        """
        self.device.lockScript.acquire()
        index = self.device.currentScript
        self.device.currentScript += 1
        self.device.lockScript.release()    
        return index    

    def acquireLocation(self, location):
        """Claims exclusive access to a specific sensor location across the network."""
        Device.DeviceLocks[location].acquire()

    def releaseLocation(self, location):
        """Releases the network-wide lock for a sensor location."""
        Device.DeviceLocks[location].release()

    def ThreadWait(self):
        """Local synchronization for the internal thread pool."""
        self.device.barrier.wait()

    def CheckForInitiator(self):
        """Check if this thread is designated as the lifecycle controller."""
        return self.isInitiator

    def finishUp(self):
        """
        Simulation phase completion logic.
        Logic: Coordinates multiple barrier points to ensure all workers are finished 
        before resetting events and advancing to the next temporal step.
        """
        self.ThreadWait()
        if self.CheckForInitiator():
            self.device.neighbours_event.clear()
            self.device.timepoint_done.clear()
        self.ThreadWait()
        if self.CheckForInitiator():
            # Synchronizes with all other devices in the network.
            Device.DeviceBarrier.wait()

    def run(self):
        """
        Main worker execution loop.
        Algorithm: Iterative timepoint processing with lock-based data protection.
        """
        while True:
            # Control Logic: Only the initiator thread handles topology discovery.
            if self.isInitiator == True:
                self.neighboursOperation()
            
            # Wait for topology state to be ready.
            self.device.neighbours_event.wait()
            if self.device.neighbours is None:
                break
            
            # Wait for simulation start signal.
            self.device.timepoint_done.wait()
            
            # Block Logic: Task Execution Phase.
            while True:
                # Concurrent reservation of the next available script index.
                index = self.reserve()

                if index >= self.device.scriptNumber:
                    # All tasks for this timepoint have been claimed.
                    break
                
                location = self.device.locations[index]
                script = self.device.scripts[index]
                
                # Global Critical Section: ensures atomic read-modify-write across the network.
                self.acquireLocation(location)
                script_data = []
                
                # Aggregate neighborhood and local data.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Process and propagate results.
                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                self.releaseLocation(location)

            # Barrier-based completion of the current simulation timepoint.
            self.finishUp()
