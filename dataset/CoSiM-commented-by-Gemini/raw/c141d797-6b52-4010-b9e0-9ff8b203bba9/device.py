
"""
This module provides a complex, multi-threaded simulation of a distributed
device network. It uses a manual worker-pool implementation and a two-tiered
barrier system for synchronization.
"""

from threading import Event, Thread, Condition, Lock

class Barrier(object):
    """
    A reusable barrier for synchronizing a group of threads.

    When a thread calls wait(), it blocks until a predefined number of threads
    have all called wait(). Then, all threads are released simultaneously.
    """
    def __init__(self, num_threads=0):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 
 
    def wait(self):
        """
        Blocks the calling thread until all threads reach the barrier.
        """
        self.cond.acquire()                      
        self.count_threads -= 1
        if self.count_threads == 0:
            # Invariant: The last thread to arrive resets the barrier counter
            # and notifies all waiting threads to proceed.
            self.cond.notify_all()               
            self.count_threads = self.num_threads    
        else:
            self.cond.wait()                  
        self.cond.release()


class Device(object):
    """
    Represents a single node in the distributed system.

    This class manages the state for a device, including its sensor data and
    assigned scripts. It employs a complex concurrency model where each device
    spawns its own fixed-size pool of worker threads.

    Synchronization across the entire system is handled by static (class-level)
    primitives, `DeviceBarrier` and `DeviceLocks`, shared by all instances.
    """
    
    # Architecture: A system-wide barrier shared by all device instances.
    # All devices in the simulation must synchronize on this barrier.
    DeviceBarrier = Barrier()

    # Architecture: A system-wide list of locks for data locations. This implies
    # that locations are globally indexed and access is mutually exclusive
    # across the entire system, not just within a single device.
    DeviceLocks = []


    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device and its dedicated worker thread pool.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        self.script_received = Event()
        self.scripts = []
        self.locations = []
        self.DeviceLocks = []
        self.currentScript = 0 # A shared counter used by worker threads to claim script work.
        self.scriptNumber = 0
        
        self.timepoint_done = Event() # Signals that a batch of scripts has been assigned.
        self.neighbours = []
        self.neighbours_event = Event() # Signals that the neighbor list has been fetched.
        self.lockScript = Lock() # Protects access to the `currentScript` counter.
        
        # A per-device barrier for its own internal worker pool.
        self.barrier = Barrier(8)
        
        # Concurrency Model: Spawns a manual worker pool of 8 threads.
        # One thread is the "initiator" with special duties.
        self.thread = DeviceThread(self, True)
        self.thread.start()
        self.threads = []
        for _ in range(7):
            newThread = DeviceThread(self, False)
            self.threads.append(newThread)
            newThread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the static, system-wide synchronization primitives.
        """
        size = len(devices)
        Device.DeviceBarrier = Barrier(size)
        # Invariant: Ensures the global location locks are initialized only once.
        if Device.DeviceLocks==[]:
            self.updateLocks()

    def getNeighbours(self):
        """Helper function to get the number of locations from the supervisor."""
        return self.supervisor.supervisor.testcase.num_locations

    def updateLocks(self):
        """Populates the global list of location locks."""
        for _ in range(self.getNeighbours()):
            Device.DeviceLocks.append(Lock())

    def assign_script(self, script, location):
        """Assigns a script to the device or signals the end of a batch."""
        if script is not None:
            self.scripts.append(script)
            self.locations.append(location)
            self.scriptNumber += 1
        else:
            # Functional Utility: Signals to the worker threads that the full
            # script batch for the current timepoint is ready for processing.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specified location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a specified location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all of the device's worker threads to complete."""
        self.thread.join()
        for myThread in self.threads:
            myThread.join()


class DeviceThread(Thread):
    """
    A worker thread for a Device.

    Each thread can be an "initiator" or a "follower". The initiator has
    additional responsibilities for coordinating the start of a processing cycle.
    Work is distributed via a shared, locked counter.
    """

    def __init__(self, device, isInitiator):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.isInitiator = isInitiator
      
    def neighboursOperation(self):
        """
        [Initiator-only] Fetches neighbor list from the supervisor and signals
        other threads on the same device to proceed.
        """
        self.device.neighbours = self.device.supervisor.get_neighbours()
        self.device.neighbours_event.set()
        self.device.currentScript = 0

    def reserve(self):
        """
        Atomically claims a script index to process from the shared counter.

        Returns:
            int: The index of the script to be processed by this thread.
        """
        with self.device.lockScript:
            index = self.device.currentScript
            self.device.currentScript += 1
        return index    

    def acquireLocation(self, location):
        """Acquires the global lock for a specific location."""
        Device.DeviceLocks[location].acquire()

    def releaseLocation(self, location):
        """Releases the global lock for a specific location."""
        Device.DeviceLocks[location].release()

    def ThreadWait(self):
        """Waits on the per-device internal barrier."""
        self.device.barrier.wait()

    def CheckForInitiator(self):
        """Checks if this thread is an initiator."""
        return self.isInitiator

    def finishUp(self):
        """
        A complex, multi-stage synchronization routine to end a processing cycle.
        
        This constitutes a manual two-phase barrier implementation, first among
        the device's own threads, and then globally across all devices.
        """
        # --- First phase of internal barrier ---
        self.ThreadWait()
        # The initiator thread resets events for the next cycle.
        if self.CheckForInitiator():
            self.device.neighbours_event.clear()
            self.device.timepoint_done.clear()
        # --- Second phase of internal barrier ---
        self.ThreadWait()
        # The initiator thread synchronizes on the global, system-wide barrier.
        if self.CheckForInitiator():
            Device.DeviceBarrier.wait()

    def run(self):
        """Main execution loop for the worker thread."""
        while True:
            # Block Logic: The initiator fetches neighbors; all threads wait for it.
            if self.isInitiator:
                self.neighboursOperation()
            self.device.neighbours_event.wait()
            
            # Pre-condition: A None value for neighbors is the shutdown signal.
            if self.device.neighbours is None:
                break
            
            # Synchronization Point: All threads wait until the full script batch
            # has been assigned.
            self.device.timepoint_done.wait()
            
            # Block Logic: Each thread repeatedly reserves and processes scripts
            # until the shared pool of scripts for the current cycle is exhausted.
            while True:
                index = self.reserve()
                
                if index >= self.device.scriptNumber:
                    break # No more scripts to process in this cycle.

                location = self.device.locations[index]
                script = self.device.scripts[index]
                
                # Core Logic: Process one script.
                self.acquireLocation(location) # Lock the location globally.
                script_data = []
                
                # Gather data from neighbors and self.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Execute script and broadcast results.
                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                self.releaseLocation(location) # Unlock the location.

            # End-of-cycle synchronization.
            self.finishUp()
