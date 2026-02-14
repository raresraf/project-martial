
"""
@05314305-b286-4c5b-a80e-5c46defa6a97/arch/arm/crypto/Makefile device.py
@brief Implements a device-centric distributed processing system with custom barrier synchronization and fine-grained locking.
This module defines the `ReusableBarrier` class for coordinating multiple threads,
the `Device` class for managing sensor data and script execution,
`DeviceThread` for orchestrating the device's operational cycle,
and `ScriptThread` for executing individual scripts in a synchronized manner across a network of devices.
"""



from threading import Event, Thread, Semaphore, Lock

class ReusableBarrier():
    """
    @brief Implements a reusable barrier synchronization mechanism for a fixed number of threads.
    This barrier ensures that all participating threads wait at a synchronization point until
    every thread has reached it, after which all threads are released simultaneously.
    It can be reused multiple times.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes a ReusableBarrier instance.
        @param num_threads The total number of threads that must reach the barrier for it to release.
        """
        self.num_threads = num_threads
        # Counters for the two phases of the barrier. Using a list to allow modification within nested scopes.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock() # Lock to protect access to the thread counters.
        
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase of waiting.
        
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase of waiting.

    def wait(self):
        """
        @brief Blocks the calling thread at the barrier until all `num_threads` threads have arrived.
        This method uses a two-phase synchronization mechanism to allow for reusability.
        """
        # Phase 1: Threads arrive and are counted.
        self.phase(self.count_threads1, self.threads_sem1)
        # Phase 2: Threads wait for release and are counted again for the next cycle.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements one phase of the two-phase reusable barrier synchronization.
        Threads acquire a lock to decrement a shared counter. The last thread to decrement
        the counter releases all waiting threads via a semaphore.
        @param count_threads A list containing the current count of threads for this phase.
        @param threads_sem The semaphore associated with this phase.
        """
        with self.count_lock: # Protect access to the shared thread counter.
            count_threads[0] -= 1 # Decrement the count of threads yet to arrive.
            
            # Block Logic: If this is the last thread to arrive, release all waiting threads.
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release() # Release one permit for each thread.
                # Reset the counter for the next cycle of the barrier.
                count_threads[0] = self.num_threads
        
        threads_sem.acquire() # Blocks until a permit is released by the last arriving thread.
        

class Device(object):
    """
    @brief Represents a single device within the distributed processing system.
    Each device manages its own sensor data, interacts with a supervisor,
    and executes scripts, coordinating with other devices through a shared barrier
    and location-specific locks.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id A unique identifier for this device.
        @param sensor_data A dictionary or similar structure storing sensor data, keyed by location.
        @param supervisor The supervisor object responsible for managing the overall device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal that a script has been received.
        self.scripts = [] # List to hold scripts assigned to this device for execution.
        self.timepoint_done = Event() # Event to signal completion of script assignments for a timepoint.
        self.lock = Lock() # Lock to protect access to the device's sensor_data.
        self.locs = [] # Not used in current code, potentially for future expansion to track locations.
        self.hashset = {} # Dictionary to store Locks for each unique data location, enabling fine-grained access control.
        self.bariera = ReusableBarrier(1) # Reusable barrier for synchronizing all devices; initialized with 1 for placeholder.
        
        self.thread = DeviceThread(self) # The main thread responsible for the device's operational cycle.
        self.thread.start() # Starts the main device thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A formatted string "Device [device_id]".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Initializes shared synchronization primitives across all devices in the network.
        This method is designed to be called by a single coordinating device (e.g., `device_id == 0`)
        to create and distribute a shared `hashset` of location-specific locks and a shared `ReusableBarrier`.
        @param devices A list of all `Device` objects in the network.
        """
        # Block Logic: Ensure setup is performed only by a designated device (e.g., device with ID 0).
        if self.device_id == 0:
            self.hashset = {} # Initialize the hashset for location locks.
            # Block Logic: Create a Lock for each unique data location found across all devices.
            for device in devices:
                for location in device.sensor_data:
                    self.hashset[location] = Lock()
            
            # Create a single shared reusable barrier for all devices.
            self.bariera = ReusableBarrier(len(devices))
            # Block Logic: Distribute the shared hashset and barrier to all devices.
            for device in devices:
                device.bariera = self.bariera
                device.hashset = self.hashset

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location on this device.
        If `script` is `None`, it signals that all scripts for the current timepoint
        have been assigned, and the `timepoint_done` event is set.
        @param script The script (callable) to assign, or `None` to signal timepoint completion.
        @param location The data location associated with the script.
        """
        # Block Logic: If a script is provided, append it to the list of scripts and signal reception.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a script has been received.
        else:
            # If `script` is None, it acts as a sentinel to signal the end of script assignment for a timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location on this device in a thread-safe manner.
        @param location The identifier of the data location.
        @return The sensor data corresponding to the `location`, or `None` if the location is not found.
        """
        # Block Logic: Acquire lock to ensure thread-safe access to `sensor_data`.
        self.lock.acquire()
        # Inline: Safely access `sensor_data` dictionary, returning `None` if the key doesn't exist.
        aux = self.sensor_data[location] if location in self.sensor_data else None


        self.lock.release() # Release lock.
        return aux

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location on this device in a thread-safe manner.
        The data is only updated if the `location` already exists in the `sensor_data`.
        @param location The identifier of the data location.
        @param data The new data to be set for the specified location.
        """
        # Block Logic: Acquire lock to ensure thread-safe access to `sensor_data`.
        self.lock.acquire()
        # Block Logic: Update data only if the location key already exists in the sensor data.
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release() # Release lock.

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device.
        This involves waiting for the main device thread to complete its execution.
        """
        self.thread.join() # Wait for the main device thread to finish.


class DeviceThread(Thread):


    """


    @brief The main operational thread for a Device.


    This thread orchestrates the device's behavior over time, including fetching


    neighbor information, dispatching scripts to `ScriptThread`s for execution,


    and synchronizing with other devices using a shared `ReusableBarrier`.


    """


    





    def __init__(self, device):


        """


        @brief Initializes a new DeviceThread instance.


        @param device The `Device` instance that this thread controls.


        """


        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device








    def run(self):


        """


        @brief The main execution loop for the device's operational cycle.


        This loop continuously fetches neighbor information, waits for scripts


        to be assigned, dispatches them to helper threads, waits for their


        completion, and synchronizes with other devices at the end of each timepoint.


        """


        while True:


            # Block Logic: Fetch information about neighboring devices from the supervisor.


            neighbours = self.device.supervisor.get_neighbours()


            # Pre-condition: If `neighbours` is None, it signals that the simulation should terminate.


            if neighbours is None:


                break # Exit the main loop to terminate the thread.








            # Block Logic: Wait until all scripts for the current timepoint have been assigned.


            self.device.timepoint_done.wait()


            


            list_threads = [] # List to hold references to helper threads executing scripts.


            # Block Logic: Create and append `ScriptThread`s for each assigned script.


            for (script, location) in self.device.scripts:


                list_threads.append(ScriptThread(self.device, script,


                location, neighbours))


            


            # Block Logic: Start all `ScriptThread`s.


            for i in xrange(len(list_threads)):


                list_threads[i].start()


            


            # Block Logic: Wait for all `ScriptThread`s to complete execution.


            for i in xrange(len(list_threads)):


                list_threads[i].join()


            


            self.device.timepoint_done.clear() # Reset the `timepoint_done` event for the next cycle.


            self.device.bariera.wait() # Synchronize with other devices using the shared barrier.




class ScriptThread(Thread):
    """
    @brief A helper thread responsible for executing a single script within a timepoint.
    This thread handles the synchronized collection of data from the local device and
    its neighbors, executes the assigned script, and propagates the results back
    to the relevant devices.
    """
    

    def __init__(self, device, script, location, neighbours):
        """
        @brief Initializes a new ScriptThread instance.
        @param device The parent `Device` instance.
        @param script The script (callable) to be executed by this thread.
        @param location The data location associated with this script.
        @param neighbours A list of neighboring devices and the local device itself.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for the script thread.
        It acquires a lock for the specific data location, gathers data from
        relevant devices, executes the script, updates data, and releases the lock.
        """
        # Block Logic: Acquire the lock associated with the data location.
        # This ensures exclusive access to the data at this location during script execution.
        self.device.hashset[self.location].acquire()
        script_data = []
        
        # Block Logic: Collect data from all neighboring devices.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Collect data from the current device itself.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: If any script data was collected, execute the script.
        if script_data != []:
            result = self.script.run(script_data) # Execute the script with the aggregated data.

            # Block Logic: Update data on all neighboring devices with the script's result.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Block Logic: Update data on the current device with the script's result.
            self.device.set_data(self.location, result)

        self.device.hashset[self.location].release() # Release the lock for the data location.
