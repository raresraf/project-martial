


"""
@file device.py
@brief This module defines a simulated device environment featuring a custom reusable barrier and a thread pooling mechanism for concurrent script execution.

@details It implements the `Device` class to represent individual simulation entities.
         The `DeviceThread` manages the device's operational loop, dispatching script
         execution to a dynamic pool of `MyThread`s. Shared `Lock` objects ensure
         data consistency across different data locations. A custom `ReusableBarrierSem`
         implementation is used for global synchronization among devices.
"""

from threading import Event, Thread, Lock, Semaphore

class Device(object):
    """
    @brief Represents a single simulated device in the environment.

    @details Each device manages its own sensor data, interacts with a central supervisor,
             and executes assigned scripts. It coordinates its activities through a dedicated
             `DeviceThread` and utilizes a shared `ReusableBarrierSem` for global
             synchronization, and a global dictionary of `Lock` objects for location-specific data consistency.
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
        self.script_received = Event()  # Event to signal that all scripts for a timepoint are assigned.
        self.scripts = []               # List to store (script, location) tuples assigned to this device.
        self.timepoint_done = Event()   # Event to signal overall timepoint processing completion.
        self.lock = {}                  # Shared dictionary of locks for data locations.


        self.barrier = None             # Shared ReusableBarrierSem for global device synchronization.
        self.devices = []               # List of all devices in the simulation (set during setup).
        self.thread = DeviceThread(self) # The dedicated thread for this device's main loop.
        self.thread.start()             # Start the main device thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d" % self.device_id.
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization resources (barrier and location locks).

        @details This method is called once at the beginning of the simulation.
                 It initializes a global `ReusableBarrierSem` for all devices and
                 a shared dictionary of `Lock` objects for each unique data location
                 present across all devices. These shared resources are then distributed
                 to all devices.
        @param devices A list of all Device objects in the simulation.
        """
        
        # Block Logic: Store references to all devices and initialize the global barrier.
        self.devices = devices
        self.barrier = ReusableBarrierSem(len(self.devices)) # Initialize the global reusable barrier.

        # Block Logic: Initialize locks for all data locations present in this device.
        for location in self.sensor_data:
            self.lock[location] = Lock() # Create a new lock for each location.
        # Block Logic: Iterate through all devices to find all unique locations and create locks for them.
        for device in devices:
            for location in device.sensor_data:
                self.lock[location] = Lock() # Ensure every location across all devices has a lock.

        # Block Logic: Distribute the initialized barrier and shared lock dictionary to all devices.
        for i in xrange(len(self.devices)):
            self.devices[i].barrier = self.barrier # Set the shared barrier.
            self.devices[i].lock = self.lock     # Set the shared dictionary of locks.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific data location.

        @details If a script is provided, it's added to the device's script queue.
                 If `script` is None, it signals the `script_received` and `timepoint_done`
                 events, indicating that all scripts for the current timepoint have been
                 assigned and the device is ready for processing.
        @param script The script object to assign, or None.
        @param location The data location (e.g., sensor ID) where the script will operate.
        """
        
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its target location.
        else:
            self.script_received.set()  # Signal that all script assignments are done.
            self.timepoint_done.set()   # Signal completion for the timepoint.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location from the device's internal state.

        @param location The data location (e.g., sensor ID).
        @return The sensor data for the location, or None if the location is not found.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

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



class MyThread(Thread):
    """
    @brief An auxiliary thread responsible for executing a single script within a device's context.

    @details This thread encapsulates the logic for running a specific script at
             a given data `location`. It acquires a location-specific lock to ensure
             data consistency, gathers data from neighboring devices, executes the script,
             and propagates the results. This thread is part of a dynamic pool managed by `DeviceThread`.
    """

    def __init__(self, my_id, device, neighbours, lock, script, location):
        """
        @brief Initializes a new MyThread instance.

        @param my_id A unique identifier for this specific thread instance.
        @param device The parent `Device` object to which this script belongs.
        @param neighbours A list of neighboring `Device` objects for data interaction.
        @param lock A shared dictionary of `Lock` objects for data locations.
        @param script The script object to be executed by this thread.
        @param location The data location (e.g., sensor ID) where the script operates.
        """
        Thread.__init__(self, name="Thread %d from device %d" % (my_id, device.device_id))
        self.device = device
        self.my_id = my_id
        self.neighbours = neighbours
        self.lock = lock
        self.script = script
        self.location = location

    
    def run(self):
        """
        @brief The main execution logic for MyThread.

        @details This method acquires a lock for its designated data `location`,
                 collects sensor data from the current device and its neighbors,
                 executes the assigned script with this data, and then updates
                 the sensor data on the current device and its neighbors with
                 the script's result. Finally, it releases the location lock.
        """
        with self.lock[self.location]: # Acquire the lock for the specific data location.
            script_data = [] # List to accumulate data for the script.
            
            # Block Logic: Gather data from neighboring devices for the current location.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            # Block Logic: Gather data from the current device for the current location.
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

    def shutdown(self):
        """
        @brief Shuts down the MyThread by waiting for its completion.

        @details This method ensures that the thread finishes its execution
                 before proceeding.
        """
        self.join()


class DeviceThread(Thread):
    """
    @brief Manages the main operational loop for a Device object, coordinating script execution.

    @details This thread is responsible for the overall lifecycle of a device's operations
             within a timepoint. It fetches neighbor information from the supervisor,
             and dispatches script execution to a dynamically managed pool of `MyThread`s.
             It also handles global synchronization using the shared `ReusableBarrierSem`.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device The `Device` object that this thread is responsible for.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)

        self.device = device # Reference to the parent Device object.
        self.numThreads = 0  # Counter for the number of MyThread instances created.
        self.listThreads = [] # List to manage the pool of MyThread instances.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        @details This loop continuously processes timepoints in the simulation.
                 It fetches neighbor information from the supervisor. If the
                 simulation is ongoing, it waits for scripts to be assigned,
                 and then manages a pool of `MyThread`s to execute scripts.
                 It reuses `MyThread`s by joining completed ones and re-assigning
                 them. Finally, it synchronizes globally with other devices
                 using the `ReusableBarrierSem`. The loop terminates when the
                 supervisor signals the end of the simulation.
        """
        
        while True:
            # Block Logic: Query the supervisor for updated neighbor information for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            # Precondition: If supervisor returns None, it signals the end of the simulation.
            if neighbours is None:
                break # Exit the main loop.

            # Block Logic: Wait for the Device to signal that all scripts for the current timepoint have been assigned.
            self.device.script_received.wait()

            # Block Logic: Dynamically manage a pool of MyThread instances for concurrent script execution.
            for (script, location) in self.device.scripts:
                # If there are fewer than 8 active MyThreads, create a new one.
                if len(self.listThreads) < 8: # Arbitrary limit for concurrent threads.
                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.append(thread)
                    thread.start()
                    self.numThreads += 1
                else:
                    # If the limit is reached, find a completed thread to reuse.
                    index = -1
                    for i in xrange(len(self.listThreads)):
                        if not self.listThreads[i].is_alive(): # Find a thread that has finished its previous task.
                            self.listThreads[i].join() # Ensure the old thread is fully completed.
                            index = i
                            break # Found a reusable slot.

                    if index != -1: # If a reusable slot was found.
                        # Remove the completed thread from the list.
                        self.listThreads.remove(self.listThreads[index])
                        # Create a new MyThread for the current script and insert it into the same slot.
                        thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                        self.listThreads.insert(index,thread)
                        self.listThreads[index].start()
                        self.numThreads += 1
                    # Note: If no inactive thread is found and the limit is reached, the current script is not processed
                    # or an unhandled exception might occur if the list is full and no inactive threads.
                    # This implies that the 'listThreads' max size of 8 might not be respected if all are alive.

            # Block Logic: Wait for all currently active MyThreads to complete their execution.
            for i in xrange(len(self.listThreads)):
                self.listThreads[i].join()

            self.device.timepoint_done.wait() # Wait for the Device to signal its timepoint completion.
            
            self.device.script_received.clear()  # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()   # Reset the event for the next timepoint.
            
            self.device.barrier.wait() # Global synchronization barrier.




class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier for synchronizing multiple threads using `threading.Semaphore`.

    @details This custom barrier uses a two-phase mechanism to ensure all participating
             threads wait until every thread has reached a common synchronization point,
             and then releases them simultaneously. It is designed to be reusable
             for subsequent synchronization points without needing re-initialization.
    """
    

    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrierSem instance.

        @param num_threads The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # Two counters are used to manage the two phases of the barrier, allowing reusability.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               # A lock to protect access to the thread counters.
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
                for i in range(self.num_threads): # Release all waiting threads.
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
                for i in range(self.num_threads): # Release all waiting threads.
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset counter for reusability.

        self.threads_sem2.acquire() # Block until released by the last thread.

