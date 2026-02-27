


"""
@file device.py
@brief This module defines a simulated device environment, featuring a global reusable barrier and fine-grained concurrency for script execution.

@details It includes a `ReusableBarrier` for thread synchronization. The `Device` class
         represents an individual simulated entity, and `DeviceThread` manages its main
         operational loop. `MyThread` instances are spawned by `DeviceThread` to
         execute scripts concurrently, utilizing semaphores for resource control
         and location-specific locks for data consistency.
"""

from threading import Semaphore, Event, Lock, Thread

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
        # Separate counters for each phase, allowing the barrier to be reusable.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock() # A lock to protect access to the thread counters.
        # Semaphores for each phase to block and release threads.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all threads have reached this barrier.

        @details This method orchestrates the two-phase synchronization, ensuring
                 all `num_threads` complete `phase1` and `phase2` before any
                 thread proceeds past the `wait` call.
        """
        
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Executes the first phase of the barrier synchronization.

        @details Threads decrement a shared counter. The last thread to reach zero
                 releases all other threads waiting on `threads_sem1` and resets
                 the counter for the next cycle. All threads then acquire `threads_sem1`.
        """
        
        with self.counter_lock: # Protects the counter from race conditions.
            self.count_threads1 -= 1
            if self.count_threads1 == 0: # If this is the last thread in phase 1.
                for _ in xrange(self.num_threads): # Release all waiting threads.
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset counter for reusability.

        self.threads_sem1.acquire() # Block until released by the last thread.

    def phase2(self):
        """
        @brief Executes the second phase of the barrier synchronization.

        @details Similar to `phase1`, threads decrement `count_threads2`. The last
                 thread to reach zero releases others waiting on `threads_sem2`
                 and resets the counter. All threads then acquire `threads_sem2`.
        """
        
        with self.counter_lock: # Protects the counter from race conditions.
            self.count_threads2 -= 1
            if self.count_threads2 == 0: # If this is the last thread in phase 2.
                for _ in xrange(self.num_threads): # Release all waiting threads.
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset counter for reusability.

        self.threads_sem2.acquire() # Block until released by the last thread.


class Device(object):
    """
    @brief Represents a single simulated device in the environment.

    @details Each device manages its own sensor data, interacts with a central supervisor,
             and executes scripts. It coordinates its activities through a dedicated
             `DeviceThread`, and relies on a shared `ReusableBarrier` and location-specific
             `Lock` objects for synchronization and data consistency during concurrent
             script execution. The device with the minimum `device_id` typically
             initializes these shared resources.
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
        self.none_script_received = Event() # Event to signal that no more scripts are assigned for a timepoint.


        self.scripts = [] # List to store (script, location) tuples assigned to this device.
        self.timepoint_done = Event() # Placeholder, seems unused in this version.
        self.thread = DeviceThread(self) # The dedicated thread for this device's main loop.
        self.thread.start() # Start the main device thread.
        self.timepoint_end = 0 # Placeholder, seems unused.
        self.barrier = None # Shared ReusableBarrier for global device synchronization.
        self.lock_hash = None # Shared dictionary of locks for data locations.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """
        @brief Sets the shared `ReusableBarrier` for this device.

        @param barrier An instance of `ReusableBarrier` shared across all devices.
        """
        
        self.barrier = barrier

    def set_locks(self, lock_hash):
        """
        @brief Sets the shared dictionary of location-specific `Lock` objects for this device.

        @param lock_hash A dictionary where keys are data locations and values are `threading.Lock` instances.
        """
        
        self.lock_hash = lock_hash

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization resources (barrier and location locks).

        @details This method is typically called once at the beginning of the simulation.
                 The device with the minimum `device_id` (leader) initializes the global
                 `ReusableBarrier` and a dictionary of `Lock` objects for each unique
                 data location. These shared resources are then distributed to all
                 other devices.
        @param devices A list of all Device objects in the simulation.
        """
        
        # Block Logic: Find the device with the minimum device_id to act as the leader.
        ids_list = []
        for dev in devices:
            ids_list.append(dev.device_id)

        # Precondition: Only the device with the minimum ID initializes shared resources.
        if self.device_id == min(ids_list):
            
            self.barrier = ReusableBarrier(len(devices)) # Initialize the global reusable barrier.
            self.lock_hash = {} # Initialize the dictionary for location locks.

            # Block Logic: Create a lock for each unique data location found across all devices.
            for dev in devices:
                for location in dev.sensor_data:
                    if location not in self.lock_hash:
                        self.lock_hash[location] = Lock() # Create a new lock for this location.

            # Block Logic: Distribute the initialized barrier and location locks to all other devices.
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.set_barrier(self.barrier)
                    dev.set_locks(self.lock_hash)


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific data location.

        @details If a script is provided, it's added to the device's script queue.
                 If `script` is None, it signals the `none_script_received` event,
                 indicating that all scripts for the current timepoint have been
                 assigned to this device.
        @param script The script object to assign, or None.
        @param location The data location (e.g., sensor ID) where the script will operate.
        """
        
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its target location.
        else:
            self.none_script_received.set() # Signal that all script assignments for this timepoint are done.

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



class DeviceThread(Thread):
    """
    @brief Manages the main operational loop for a Device object, coordinating script execution.

    @details This thread is responsible for the overall lifecycle of a device's operations
             within a timepoint. It fetches neighbor information from the supervisor,
             waits for scripts to be assigned, and then spawns `MyThread` instances
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
        self.semaphore = Semaphore(value=8) # Controls the number of concurrent `MyThread`s.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        @details This loop continuously processes timepoints in the simulation.
                 It fetches neighbor information, waits for scripts to be assigned,
                 creates and joins `MyThread`s for parallel script execution,
                 clears the `none_script_received` event, and finally synchronizes with
                 all other devices using the global barrier. The loop terminates
                 when the supervisor signals the end of the simulation.
        """
        
        while True:
            # Block Logic: Query the supervisor for updated neighbor information for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            # Precondition: If supervisor returns None, it signals the end of the simulation.
            if neighbours is None:
                break # Exit the main loop.

            # Block Logic: Wait for the Device to signal that all scripts for the current timepoint have been assigned.
            self.device.none_script_received.wait()
            self.device.none_script_received.clear() # Reset the event for the next timepoint.

            thread_list = [] # List to hold references to MyThread instances.

            # Block Logic: Create and start a MyThread for each assigned script.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, neighbours, script, location,
                    self.semaphore) # Pass the semaphore to control concurrency.
                thread.start()
                thread_list.append(thread)

            # Block Logic: Wait for all spawned MyThreads to complete their execution.
            for i in xrange(len(thread_list)):
                thread_list[i].join()

            # Block Logic: Global synchronization barrier to ensure all devices complete their timepoint processing.
            self.device.barrier.wait()


class MyThread(Thread):
    """
    @brief An auxiliary thread responsible for executing a single assigned script concurrently.

    @details This thread encapsulates the logic for running a specific script at
             a given data `location`. It uses a shared semaphore to limit overall
             concurrency and acquires a location-specific lock to ensure data
             consistency during data gathering, script execution, and data propagation.
    """
    
    def __init__(self, device, neighbours, script, location, semaphore):
        """
        @brief Initializes a new MyThread instance.

        @param device The parent `Device` object to which this script belongs.
        @param neighbours A list of neighboring `Device` objects for data interaction.
        @param script The script object to be executed by this thread.
        @param location The data location (e.g., sensor ID) where the script operates.
        @param semaphore A shared `threading.Semaphore` to limit the number of concurrently running `MyThread`s.
        """
        
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.semaphore = semaphore

    def run(self):
        """
        @brief The main execution logic for the MyThread.

        @details This method first acquires a permit from the shared semaphore
                 to control concurrency. It then acquires a lock for its designated
                 data `location`, collects sensor data from the current device
                 and its neighbors, executes the assigned script with this data,
                 and then updates the sensor data on the current device and its
                 neighbors with the script's result. Finally, it releases the
                 location lock and the semaphore permit.
        """
        
        self.semaphore.acquire() # Acquire a permit from the shared semaphore.

        # Block Logic: Acquire a lock for the specific data location to ensure exclusive access
        # during data gathering, script execution, and data propagation.
        self.device.lock_hash[self.location].acquire()

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

            # Update sensor data on the current device.
            self.device.set_data(self.location, result)

        self.device.lock_hash[self.location].release() # Release the lock for the data location.

        self.semaphore.release() # Release the permit back to the shared semaphore.

