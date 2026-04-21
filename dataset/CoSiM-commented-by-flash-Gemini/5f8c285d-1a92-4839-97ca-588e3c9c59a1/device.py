"""
@5f8c285d-1a92-4839-97ca-588e3c9c59a1/device.py
@brief Distributed device simulation framework for sensor data processing.
Functional Utility: Orchestrates a multi-threaded environment where virtual devices 
process sensor streams using assigned logic scripts. Implements local data locking 
and global barrier synchronization for temporal consistency.
Domain: Distributed Systems Simulation.
"""

from threading import Thread, Lock, Event, Semaphore

class Device(object):
    """
    @brief Represents a single computational node in a distributed simulation.
    Functional Utility: Manages local state (sensor_data) and coordinates script 
    execution via dedicated threading.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.
        @param device_id Unique numerical identifier for the node.
        @param sensor_data Repository of localized sensor values.
        @param supervisor Reference to the central coordination authority.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to hold assigned scripts (tuples of (script, location)).
        self.lock = Lock() # A lock to protect this device's `sensor_data` from concurrent access.
        self.all_scripts_received = Event() # Event to signal when all scripts for a timepoint have been assigned.
        self.barrier = None # Placeholder for the global ReusableBarrier, set in setup_devices.
        self.thread = None # Placeholder for the DeviceThread, initialized in setup_devices.
        self.devices = None # Placeholder for a list of all devices in the simulation, used by the master device.

    def __str__(self):
        """
        Standardized string identifier for debugging and logging.
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Functional Utility: Global orchestration logic executed by the master node (ID 0).
        Logic: Instantiates the shared synchronization barrier and initiates processing 
        threads for all participating nodes.
        @param devices Full collection of device instances in the topology.
        """
        # Block Logic: Only the device with device_id 0 performs this setup.
        if self.device_id is 0:
            self.devices = devices # Store references to all devices.
            
            self.barrier = ReusableBarrier(len(devices)) # Inline: Creates a global ReusableBarrier.
            self.thread = DeviceThread(self, self.lock, self.barrier) # Create the DeviceThread for this master device.
            for dev in devices:
                if dev.device_id is not 0: # For all other devices.
                    dev.barrier = self.barrier # Assign the global barrier.
                    # Inline: Create the DeviceThread for the current device and assign its private lock and the global barrier.
                    dev.thread = DeviceThread(dev, dev.lock, self.barrier)
                dev.thread.start() # Start each DeviceThread.

    def assign_script(self, script, location):
        """
        Functional Utility: Asynchronous dispatch of logic kernels to the device.
        @param script The algorithm object to be executed.
        @param location Target sensor data index for the script.
        """
        if script is not None:
            
            self.scripts.append((script, location)) # Add the script and its location to the list.
        else:
            
            self.all_scripts_received.set() # If script is None, signal that script assignments are done for the timepoint.

    def get_data(self, location):
        """
        @brief Local state retrieval with boundary checking.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None


    def set_data(self, location, data):
        """
        @brief Local state mutation for sensor data updates.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Functional Utility: Graceful termination sequence for the distributed cluster.
        Logic: Master node performs a blocking join on all active device threads.
        """
        
        # Block Logic: Only the device with device_id 0 is responsible for initiating the shutdown of all threads.
        if self.device_id is 0:
            for dev in self.devices: # Iterate through all devices in the simulation.
                dev.thread.join() # Wait for each device's DeviceThread to finish its execution.




class DeviceThread(Thread):
    """
    @brief Main execution lifecycle for a device node.
    Functional Utility: Implements the fetch-execute-sync cycle for simulated time-steps.
    """

    def __init__(self, device, lock, barrier):
        """
        Initializes a `DeviceThread` instance.
        @param device Owner device context.
        @param lock Mutex for local sensor data protection.
        @param barrier Shared barrier for temporal alignment.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.lock = lock
        self.barrier = barrier

    def run(self):
        """
        @brief Core execution loop for the device simulation.
        Invariant: Threads wait for all scripts to be dispatched before starting execution phase.
        Synchronization: Uses a two-stage barrier to prevent race conditions between simulation steps.
        """

        while True:
            # Block Logic: Fetch neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Inline: If `neighbours` is None, it signals termination for the device.
            if neighbours is None:
                break # Exit the main loop, terminating the DeviceThread.
            
            # Block Logic: Wait for the `all_scripts_received` event to be set.
            self.device.all_scripts_received.wait()
            
            # Inline: Clear the event for the next timepoint.
            self.device.all_scripts_received.clear()

            # Block Logic: Iterate through all assigned scripts for the current timepoint.
            for (script, location) in self.device.scripts:
                script_data = [] # List to collect input data for the current script.
                
                # Block Logic: Data collection phase from topological neighbors.
                # Optimization: Locks are acquired per-neighbor to ensure atomic data reads.
                for device in neighbours:
                    # Inline: Acquire the neighbor's private lock before accessing its sensor data.
                    device.lock.acquire()
                    data = device.get_data(location) # Get data from the neighbor.
                    if data is not None:
                        script_data.append(data) # Add to script input if available.
                
                # Block Logic: Include local node data in the script input.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data) # Add to script input if available.

                # Block Logic: Execute algorithm and propagate results back to the neighborhood.
                if script_data != []:
                    # Inline: Execute the script's `run` method with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Update sensor data for all involved devices.
                    for device in neighbours:
                        device.set_data(location, result) # Update neighbor's data.

                    self.device.set_data(location, result) # Update this device's own data.
                
                # Block Logic: Cleanup phase - release all neighbor mutexes.
                for device in neighbours:
                    device.lock.release() # Release neighbor's private lock.

            # Block Logic: Global temporal barrier.
            # Ensures all devices complete processing before the next simulation step begins.
            self.barrier.wait()

class ReusableBarrier():
    """
    @brief Semaphore-based reusable synchronization primitive.
    Functional Utility: Implements a two-phase barrier to allow safe reuse across 
    consecutive simulation cycles without state corruption.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier.
        @param num_threads Expected number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Counter for the first phase of the barrier.
        self.count_threads2 = [self.num_threads] # Counter for the second phase of the barrier.
        self.count_lock = Lock() # Lock to protect the shared counters during decrements and resets.
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase of threads to wait on.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase of threads to wait on.

    def wait(self):
        """
        @brief Blocking synchronization call.
        Logic: Executes two identical phases to ensure all threads have passed 
        the first gate before any can reset it for the next step.
        """
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Orchestrates a single gate-release cycle.
        @param count_threads Current thread countdown list.
        @param threads_sem Blocking semaphore for this gate.
        """
        
        with self.count_lock: # Protect shared counter access.
            count_threads[0] -= 1 # Decrement the count of threads remaining.
            
            if count_threads[0] == 0: # If this is the last thread to arrive:
                for i in range(self.num_threads):
                    threads_sem.release() # Release all waiting threads by incrementing the semaphore.
                count_threads[0] = self.num_threads # Reset counter for next use.
        
        threads_sem.acquire() # Wait (decrement) the semaphore, blocking until released by the last thread.
