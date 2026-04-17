"""
@c141d797-6b52-4010-b9e0-9ff8b203bba9/device.py
@brief Distributed sensor processing simulation using a collective thread pool and global synchronization locks.
* Algorithm: Atomic task reservation with per-location mutual exclusion and multi-level barrier synchronization.
* Functional Utility: Manages a fixed-size pool of worker threads that collaboratively process assigned scripts while maintaining global state consistency.
"""

from threading import Event, Thread, Condition, Lock

class Barrier(object):
    """
    @brief Synchronizes a specific number of threads using condition variables.
    * Functional Utility: Acts as a standard synchronization point for phased execution.
    """
    def __init__(self, num_threads=0):
        """
        @brief Initializes the barrier with a target thread count.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 
    def wait(self):
        """
        @brief Blocks calling thread until all expected participants have arrived.
        """
        self.cond.acquire()                      
        self.count_threads -= 1
        if self.count_threads == 0:
            # Logic: Last thread notifies all waiters and resets the count for re-use.
            self.cond.notify_all()               
            self.count_threads = self.num_threads    
        else:
            self.cond.wait()                    
        self.cond.release()

class Device(object):
    """
    @brief Main sensor device abstraction managing local data and worker threads.
    * Domain: Shared class-level synchronization primitives for cluster coordination.
    """
    
    DeviceBarrier = Barrier() # Intent: Global barrier for cross-device alignment.
    DeviceLocks = []          # Intent: Global locks for protecting specific sensor locations.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes device state and bootstraps the internal worker pool (8 threads).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        self.script_received = Event()
        self.scripts = []
        self.locations = []
        self.currentScript = 0
        self.scriptNumber = 0
        self.timepoint_done = Event()

        self.neighbours = []
        self.neighbours_event = Event()
        self.lockScript = Lock() # Intent: Protects the atomic reservation of tasks.
        self.barrier = Barrier(8) # Intent: Internal device barrier for worker coordination.
        
        # Block Logic: Spawns exactly 8 worker threads per device.
        self.thread = DeviceThread(self, True) # Functional Utility: Designates one thread as the "Initiator" for management tasks.
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
        @brief Global initialization of shared cluster resources.
        """
        size = len(devices)
        Device.DeviceBarrier = Barrier(size)
        if Device.DeviceLocks == []:
            self.updateLocks()

    def getNeighbours(self):
        """
        @brief Utility to retrieve simulation-wide location count.
        """
        return self.supervisor.supervisor.testcase.num_locations

    def updateLocks(self):
        """
        @brief Populates the global lock list for all possible sensor locations.
        """
        for _ in range(self.getNeighbours()):
            Device.DeviceLocks.append(Lock())

    def assign_script(self, script, location):
        """
        @brief Appends a script to the local execution queue.
        """
        if script is not None:
            self.scripts.append(script)
            self.locations.append(location)
            self.scriptNumber += 1
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Basic getter for sensor data at a specific location.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Basic setter for sensor data at a specific location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Waits for all worker threads to terminate.
        """
        self.thread.join()
        for myThread in self.threads:
            myThread.join()

class DeviceThread(Thread):
    """
    @brief worker thread implementing the core processing loop.
    """

    def __init__(self, device, isInitiator):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.isInitiator = isInitiator
      
    def neighboursOperation(self):
        """
        @brief Refreshes neighbor set and resets phase state.
        """
        self.device.neighbours = self.device.supervisor.get_neighbours()
        self.device.neighbours_event.set()
        self.device.currentScript = 0

    def reserve(self):
        """
        @brief Atomically reserves the next available script task for this thread.
        Algorithm: Fetch-and-add on the shared task index.
        """
        self.device.lockScript.acquire()
        index = self.device.currentScript
        self.device.currentScript += 1
        self.device.lockScript.release()    
        return index    

    def acquireLocation(self, location):
        """
        @brief Blocks until exclusive access to the target sensor location is granted.
        """
        Device.DeviceLocks[location].acquire()

    def releaseLocation(self, location):
        """
        @brief Releases exclusive access to the target sensor location.
        """
        Device.DeviceLocks[location].release()

    def ThreadWait(self):
        """
        @brief Internal device barrier wait.
        """
        self.device.barrier.wait()

    def CheckForInitiator(self):
        return self.isInitiator

    def finishUp(self):
        """
        @brief coordinates the end-of-phase cleanup and global synchronization.
        """
        self.ThreadWait()
        if self.CheckForInitiator():
            self.device.neighbours_event.clear()
            self.device.timepoint_done.clear()
        self.ThreadWait()
        # Invariant: Global barrier ensures all devices have completed their local tasks.
        if self.CheckForInitiator():
            Device.DeviceBarrier.wait()

    def run(self):
        """
        @brief Main execution lifecycle for a worker thread.
        """
        while True:
            # Logic: Only the designated initiator thread fetches the current neighbor set.
            if self.isInitiator == True:
                self.neighboursOperation()
            
            # Block Logic: Ensures all workers have current neighbor info before starting.
            self.device.neighbours_event.wait()
            if self.device.neighbours is None:
                break
            
            # Block Logic: Waits for script assignment to complete for the current timepoint.
            self.device.timepoint_done.wait()
            
            while True:
                # Task Processing Phase.
                index = self.reserve()
                if index >= self.device.scriptNumber:
                    # Logic: Current task pool exhausted.
                    break
                
                location = self.device.locations[index]
                script = self.device.scripts[index]
                
                # Logic: Acquire location-specific lock for atomic distributed update.
                self.acquireLocation(location)
                script_data = []
                
                # Distributed Aggregation: Accumulate data from self and peers.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Post-processing Phase: Execute and broadcast results.
                if script_data != []:
                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                self.releaseLocation(location)

            # Cleanup and Phase Transition.
            self.finishUp()
