


"""
@file device.py
@brief This module defines the architecture for a simulated device environment, supporting concurrent script execution and synchronization.

@details It includes `ReusableBarrier` for thread synchronization, `Device` to represent
         individual simulation participants, `ScriptThread` for parallel script execution
         on specific data locations, and `DeviceThread` to manage the device's main
         operational loop and orchestrate script dispatches. Device 0 handles the
         initialization of shared synchronization primitives.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.

    @details This barrier uses a two-phase mechanism to ensure all participating
             threads wait until every thread has reached a common synchronization point,
             and then releases them simultaneously. It is designed to be reusable
             for subsequent synchronization points without needing re-initialization.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrier instance.

        @param num_threads The total number of threads that will participate in the barrier.
        """
        
        self.num_threads = num_threads
        # Two counters are used to manage the two phases of the barrier, allowing reusability.
        # Stored in a list to allow modification within nested scopes (e.g., `with self.count_lock`).
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock() # A lock to protect access to the thread counters.
        
        # Two semaphores, one for each phase, to block and release threads.
        self.threads_sem1 = Semaphore(0)
        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all threads have reached this barrier.

        @details This method orchestrates the two-phase synchronization, ensuring
                 all `num_threads` complete `phase1` and `phase2` before any
                 thread proceeds past the `wait` call.
        """
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single phase of the barrier synchronization.

        @details Decrements a shared counter. When the counter reaches zero, it
                 releases all waiting threads using a semaphore. It then resets
                 the counter for the next phase.
        @param count_threads A list containing the counter for the current phase (mutable).
        @param threads_sem The semaphore associated with the current phase.
        """
        
        with self.count_lock: # Ensure atomic access to the counter.
            count_threads[0] -= 1
            
            if count_threads[0] == 0: # If this is the last thread to reach the barrier.
                for _ in range(self.num_threads): # Release all waiting threads.
                    threads_sem.release()
                
                count_threads[0] = self.num_threads # Reset counter for reusability.
        threads_sem.acquire() # Block until released by the last thread.


class Device(object):
    """
    @brief Represents a single simulated device in the environment.

    @details Each device manages its own sensor data, interacts with a central supervisor,
             and executes scripts. It coordinates its activities through a dedicated
             `DeviceThread`, uses a shared `ReusableBarrier` for synchronization with
             other devices, and employs `location_locks` to ensure data consistency
             during concurrent script execution. Device 0 typically initializes shared resources.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id An integer representing the unique identifier for the device.
        @param sensor_data A dictionary containing initial sensor data for the device.
        @param supervisor An object responsible for overseeing and coordinating devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # List to store (script, location) tuples assigned to this device for current timepoint.
        self.scripts = []
        self.timepoint_done = Event() # Event to signal when all scripts for a timepoint are assigned.
        self.barrier = None # Shared ReusableBarrier for global device synchronization.
        self.thread = DeviceThread(self) # The dedicated thread for this device's main loop.
        self.thread.start() # Start the main device thread.
        self.location_locks = None # Shared list of locks for data locations.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization resources (barrier and location locks).

        @details This method is typically called once at the beginning of the simulation.
                 Only Device 0 (the leader) initializes the global `ReusableBarrier` and
                 a list of `Lock` objects for each unique data location. These shared
                 resources are then distributed to all other devices.
        @param devices A list of all Device objects in the simulation.
        """

        # Block Logic: Only device with ID 0 is responsible for initializing shared resources.
        if 0 == self.device_id:
            # Initialize the global reusable barrier for all devices.
            self.barrier = ReusableBarrier(len(devices))
            
            # Block Logic: Collect all unique data locations across all devices to create a lock for each.
            locations = []
            for device in devices:
                for location in device.sensor_data:
                    if location not in locations:
                        locations.append(location)
            
            # Initialize a list of locks, one for each unique data location.
            self.location_locks = []
            for _ in range(len(locations)):
                self.location_locks.append(Lock())
            
            # Distribute the initialized barrier and location locks to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific data location.

        @details If a script is provided, it's added to the device's script queue.
                 If `script` is None, it signals that all scripts for the current
                 timepoint have been assigned to this device.
        @param script The script object to assign, or None.
        @param location The data location (e.g., sensor ID) where the script will operate.
        """

        if script is not None:
            self.scripts.append((script, location)) # Add the script and its target location.
        else:
            # Signal that all script assignments for this timepoint are done.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location from the device's internal state.

        @param location The data location (e.g., sensor ID).
        @return The sensor data for the location, or None if the location is not found.
        """
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location in the device's internal state.

        @param location The data location (e.g., sensor ID).
        @param data The new sensor data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's dedicated thread.

        @details Joins the `DeviceThread`, ensuring all ongoing operations are completed
                 before the device fully shuts down.
        """
        
        self.thread.join() # Wait for the main device thread to finish its execution.



class ScriptThread(Thread):
    """
    @brief An auxiliary thread responsible for executing a single assigned script.

    @details This thread encapsulates the logic for running a specific script at
             a given data `location`. It coordinates with other threads by acquiring
             a lock for the `location` to ensure data consistency, gathers data from
             neighboring devices, executes the script, and propagates the results.
    """
    
    def __init__(self, device, script, location, neighbours):
        """
        @brief Initializes a new ScriptThread instance.

        @param device The parent `Device` object to which this script belongs.
        @param script The script object to be executed by this thread.
        @param location The data location (e.g., sensor ID) where the script operates.
        @param neighbours A list of neighboring `Device` objects for data interaction.
        """
        
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for the ScriptThread.

        @details This method acquires a lock for its designated data `location`,
                 collects sensor data from the current device and its neighbors,
                 executes the assigned script with this data, and then updates
                 the sensor data on the current device and its neighbors with
                 the script's result. Finally, it releases the location lock.
        """
        # Block Logic: Acquire a lock for the specific data location to ensure exclusive access
        # during data gathering, script execution, and data propagation.
        with self.device.location_locks[self.location]:
            script_data = [] # List to accumulate data for the script.
            
            # Block Logic: Gather data from neighboring devices for the current location.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Get data from the current device for the current location.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
            # Precondition: Execute the script only if there is data available.
            if script_data != []:
                # Execute the assigned script with the collected data.
                result = self.script.run(script_data)
                
                # Block Logic: Update sensor data on neighboring devices.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                // Update sensor data on the current device.
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    @brief Manages the main operational loop for a Device object, coordinating script execution.

    @details This thread is responsible for the overall lifecycle of a device's operations
             within a timepoint. It fetches neighbor information from the supervisor,
             waits for scripts to be assigned, and then spawns `ScriptThread` instances
             to execute each script in parallel. Finally, it uses the shared `ReusableBarrier`
             for global synchronization across all devices.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device The `Device` object that this thread is responsible for.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device # Reference to the parent Device object.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        @details This loop continuously processes timepoints in the simulation.
                 It fetches neighbor information, waits for scripts to be assigned,
                 creates and joins `ScriptThread`s for parallel script execution,
                 resets the `timepoint_done` event, and finally synchronizes with
                 all other devices using the global barrier. The loop terminates
                 when the supervisor signals the end of the simulation.
        """
        while True:
            # Block Logic: Query the supervisor for updated neighbor information for the current timepoint.
            vecini = self.device.supervisor.get_neighbours()
            # Precondition: If supervisor returns None, it signals the end of the simulation.
            if vecini is None:
                break # Exit the main loop.
            
            # Block Logic: Wait for all scripts for the current timepoint to be assigned by the Device.
            self.device.timepoint_done.wait()
            threads = [] # List to hold references to ScriptThread instances.
            
            # Precondition: Only proceed with script execution if there are neighbors (i.e., not an isolated device).
            # This condition seems to imply that scripts only run if there's interaction with neighbors.
            if len(vecini) != 0:
                # Block Logic: Create and start a ScriptThread for each assigned script.
                for (script, locatie) in self.device.scripts:
                    thread = ScriptThread(self.device, script, locatie, vecini)
                    threads.append(thread)
                    thread.start()
                
                # Block Logic: Wait for all spawned ScriptThreads to complete their execution.
                for thread in threads:
                    thread.join()
            
            # Block Logic: Reset the timepoint completion event and synchronize globally.
            self.device.timepoint_done.clear() # Reset the event for the next timepoint.
            
            # Global synchronization barrier to ensure all devices complete their timepoint processing.
            self.device.barrier.wait()

