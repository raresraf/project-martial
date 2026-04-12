
"""
@brief A complex, multi-threaded simulation of a distributed device network.
       It uses a manual worker-pool implementation and a two-tiered barrier
       system for synchronization.
@details This module simulates a network of devices where each device operates
         concurrently with its own dedicated pool of worker threads. Synchronization
         is managed through a combination of local (per-device) and global
         (system-wide) barriers and locks.
"""

from threading import Event, Thread, Condition, Lock

class Barrier(object):
    """
    A reusable barrier for synchronizing a group of threads.

    When a thread calls wait(), it blocks until a predefined number of threads
    have all called wait(). Then, all threads are released simultaneously. This
    is a classic cyclic barrier implementation.
    """
    def __init__(self, num_threads=0):
        """
        Initializes the barrier for a given number of threads.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 
 
    def wait(self):
        """
        Blocks the calling thread until all threads reach the barrier.
        The last thread to arrive resets the barrier for reuse.
        """
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                # Invariant: The last thread to arrive resets the barrier counter
                # and notifies all waiting threads to proceed. This makes the
                # barrier cyclic/reusable.
                self.cond.notify_all()               
                self.count_threads = self.num_threads    
            else:
                self.cond.wait()


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
    # All devices in the simulation must synchronize on this barrier at the end
    # of each processing cycle.
    DeviceBarrier = Barrier()

    # Architecture: A system-wide list of locks for data locations. This implies
    # that locations are globally indexed and access is mutually exclusive
    # across the entire system, not just within a single device, preventing
    # race conditions on shared sensor data.
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
        # Concurrency Model: A shared counter for the worker pool to claim
        # script work. Access is controlled by `lockScript`.
        self.currentScript = 0
        self.scriptNumber = 0
        
        # Functional Utility: Signals that the full script batch for the
        # current timepoint is assigned and ready for processing.
        self.timepoint_done = Event()
        self.neighbours = []
        # Functional Utility: Signals that the neighbor list has been fetched,
        # allowing worker threads to proceed.
        self.neighbours_event = Event()
        self.lockScript = Lock()
        
        # A per-device barrier for its own internal worker pool of 8 threads.
        self.barrier = Barrier(8)
        
        # Concurrency Model: Spawns a manual worker pool of 8 threads.
        # One thread is the "initiator" with special duties for coordination.
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
        This method must be called once before the simulation begins.
        """
        size = len(devices)
        Device.DeviceBarrier = Barrier(size)
        # Invariant: Ensures the global location locks are initialized only once
        # for the entire system.
        if Device.DeviceLocks==[]:
            self.updateLocks()

    def getNeighbours(self):
        """Helper function to get the number of locations from the supervisor."""
        return self.supervisor.supervisor.testcase.num_locations

    def updateLocks(self):
        """Populates the global list of location-specific locks."""
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
    Work is distributed via a shared, locked counter (`currentScript`),
    emulating a simple work queue.
    """

    def __init__(self, device, isInitiator):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.isInitiator = isInitiator
      
    def neighboursOperation(self):
        """
        [Initiator-only] Fetches neighbor list from the supervisor and signals
        other threads on the same device to proceed with the processing cycle.
        """
        self.device.neighbours = self.device.supervisor.get_neighbours()
        self.device.neighbours_event.set()
        self.device.currentScript = 0

    def reserve(self):
        """
        Atomically claims a script index from the device's shared work queue.
        This provides a simple, lock-based mechanism for work distribution
        among the threads in the device's pool.

        Returns:
            int: The index of the script to be processed by this thread.
        """
        with self.device.lockScript:
            index = self.device.currentScript
            self.device.currentScript += 1
        return index    

    def acquireLocation(self, location):
        """Acquires the global lock for a specific location to ensure mutual exclusion."""
        Device.DeviceLocks[location].acquire()

    def releaseLocation(self, location):
        """Releases the global lock for a specific location."""
        Device.DeviceLocks[location].release()

    def ThreadWait(self):
        """Waits on the per-device internal barrier to synchronize local workers."""
        self.device.barrier.wait()

    def CheckForInitiator(self):
        """Checks if this thread has special coordinating duties."""
        return self.isInitiator

    def finishUp(self):
        """
        A complex, multi-stage synchronization routine to end a processing cycle.
        
        This constitutes a manual two-phase barrier implementation. First, it
        synchronizes threads locally within the device. Then, the initiator
        thread synchronizes globally across all devices in the simulation.
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
            # Block Logic: The initiator thread fetches the neighbor list for the
            # device, while all other threads in the pool wait for this signal.
            if self.isInitiator:
                self.neighboursOperation()
            self.device.neighbours_event.wait()
            
            # Pre-condition: A None value for neighbors is the shutdown signal
            # propagated from the supervisor.
            if self.device.neighbours is None:
                break
            
            # Synchronization Point: All threads wait until the supervisor has
            # finished assigning the complete batch of scripts for this time step.
            self.device.timepoint_done.wait()
            
            # Block Logic: Each thread repeatedly reserves and processes scripts
            # from the shared pool until the work for the current cycle is exhausted.
            while True:
                index = self.reserve()
                
                # Post-condition: If the reserved index exceeds the number of
                # available scripts, the work for this cycle is complete.
                if index >= self.device.scriptNumber:
                    break

                location = self.device.locations[index]
                script = self.device.scripts[index]
                
                # Core Logic: Process one script.
                self.acquireLocation(location) # Ensure exclusive access to the location.
                script_data = []
                
                # Data Aggregation: Gather data from all neighbors and self for the script.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Execute script and broadcast results to all neighbors and self.
                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                self.releaseLocation(location)

            # End-of-cycle synchronization.
            self.finishUp()
