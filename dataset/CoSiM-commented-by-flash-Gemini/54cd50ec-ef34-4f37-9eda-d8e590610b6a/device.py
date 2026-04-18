
"""
@54cd50ec-ef34-4f37-9eda-d8e590610b6a/device.py
@brief Distributed device simulation framework with multi-phase barrier synchronization.
This module implements a simulation environment for autonomous devices that 
execute scripts and share sensor data. It features a custom multi-phase 
reusable barrier for global time-step synchronization and a granular locking 
mechanism to ensure data consistency across neighboring devices during parallel 
script execution.

Domain: Distributed Systems Simulation, Concurrent Programming, Synchronization Primitives.
"""


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """
    Functional Utility: Implements a re-usable N-thread barrier using a two-phase semaphore protocol.
    Logic: Uses two sequential rendezvous phases (phase1 and phase2) to ensure 
    that all threads have completed the current cycle before any can start 
    the next, preventing race conditions in periodic synchronization.
    """
    
    
    def __init__(self, num_threads):
        """
        Constructor: Initializes the barrier for a fixed number of participants.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               # Protects access to the count_threads variables.
        self.threads_sem1 = Semaphore(0)         # Semaphore for the first phase.
        self.threads_sem2 = Semaphore(0)         # Semaphore for the second phase.
    
    def wait(self):
        """
        Execution Logic: Synchronizes the calling thread across both barrier phases.
        """
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """
        Block Logic: Phase 1 rendezvous.
        Logic: The last thread to arrive triggers the release of all waiting 
        threads via the phase-1 semaphore.
        """
        with self.counter_lock: # Ensure atomic access to the counter.
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads): 
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads       # Reset counter for next use.
        self.threads_sem1.acquire() # Wait for all threads to reach this point.
    
    def phase2(self):
        """
        Block Logic: Phase 2 rendezvous (completion phase).
        Logic: Secondary synchronization gate to reset the barrier state for the 
        next temporal step.
        """
        with self.counter_lock: 
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads): 
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads       # Reset counter for next use.
        self.threads_sem2.acquire() # Wait for all threads to reach this point.


class Device(object):
    """
    Functional Utility: Represent an autonomous device in the simulation.
    Logic: Manages local sensor data, neighbors, and a pool of worker threads. 
    It coordinates script execution cycles and participates in cluster-wide 
    synchronization via a shared barrier.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Constructor: Initializes device identity, data storage, and the main coordination thread.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are assigned.
        self.scripts = [] # List to store assigned scripts (script, location) tuples.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        self.thread = DeviceThread(self) # The main thread for this device.

        
        self.neighbours = [] # List to store neighboring devices.
        self.alldevices = [] # List of all devices in the simulation.
        self.barrier = None # Global barrier for device synchronization.
        self.threads = [] # List to hold MyThread instances (worker threads).
        self.threads_number = 8 # Number of worker threads for this device.
        self.locks = [None] * 100 # List to store location-specific locks. Each lock protects access to a location's data.

        self.thread.start() # Start the main device thread.

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Functional Utility: Cluster-wide initialization of the global barrier.
        Logic: Propagates a single barrier instance across all devices to ensure 
        they all synchronize on the same temporal boundary.
        """
        
        # Block Logic: Global synchronization setup.
        if self.barrier is None:
            # Inline: Initialize the barrier with the total number of devices.
            barrier = ReusableBarrierSem(len(devices))
            self.barrier = barrier # Set the barrier for the current device.
            for d in devices:
                if d.barrier is None:
                    d.barrier = barrier
        
        for device in devices:
            if device is not None:
                self.alldevices.append(device)


    def assign_script(self, script, location):
        """
        Functional Utility: Maps scripts to specific spatial locations for execution.
        Logic: Retrieves existing locks for the location from other devices or 
        creates new ones to ensure exclusive access to data at that location 
        across the cluster.
        """
        
        
        no_lock_for_location = 0; # Flag to indicate if a lock for the location has been found/created.
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
            
            # Block Logic: Distributed lock discovery.
            for device in self.alldevices:

                if device.locks[location] is not None:
                    self.locks[location] = device.locks[location]
                    no_lock_for_location = 1; # Set flag to indicate lock found.
                    break;
            
            # Block Logic: Lock creation.
            if no_lock_for_location == 0:
                self.locks[location] = Lock()
            self.script_received.set() # Signal that a new script is available.
        else:
            self.timepoint_done.set() # Signal that processing for the current timepoint is complete.

    def get_data(self, location):
        """
        Functional Utility: Retrieves local sensor reading for a location.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Functional Utility: Updates local sensor reading for a location.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Functional Utility: Graceful termination of the device coordination thread.
        """
        
        self.thread.join()


class MyThread(Thread):
    """
    Functional Utility: Worker thread that executes a script over a neighborhood of devices.
    Logic: Acquires the location-specific lock, aggregates data from neighbors, 
    executes the script, and propagates the result back to the neighborhood.
    """

    def __init__(self, device, location, script, neighbours):
        """
        Constructor: Initializes the worker with its target location and neighbors.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Execution Logic: Atomic neighborhood data reconciliation.
        Invariant: All neighbors are updated with the script result before the 
        location lock is released.
        """
        self.device.locks[self.location].acquire()
        script_data = []
        
        # Block Logic: Neighborhood data aggregation.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Perform the core script logic.
            result = self.script.run(script_data)

            # Block Logic: Result propagation to neighbors and self.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)
        self.device.locks[self.location].release()

class DeviceThread(Thread):
    """
    Functional Utility: Main coordination thread for a Device instance.
    Logic: Implements an infinite simulation loop that retrieves neighborhood 
    topology from the supervisor and orchestrates worker threads for each 
    time-step before synchronizing at the global barrier.
    """
    

    def __init__(self, device):
        """
        Constructor: Binds the coordinator thread to its parent device.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Execution Logic: Main simulation temporal loop.
        """
        while True:
            
            # Block Logic: Temporal step initialization.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            self.device.neighbours = neighbours

            count = 0
            
            # Block Logic: Worker thread dispatch.
            for (script, location) in self.device.scripts:
                
                if count >= self.device.threads_number:
                    break
                count = count + 1
                thread = MyThread(self.device, location, script, neighbours)
                self.device.threads.append(thread)

            # Block Logic: Parallel execution rendezvous.
            for thread in self.device.threads:
                thread.start()
            for thread in self.device.threads:
                thread.join()
            self.device.threads = []

            
            # Block Logic: Temporal boundary synchronization.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
