




"""
@file device.py
@brief This module defines the behavior and synchronization mechanisms for individual devices and their associated threads in a distributed simulation environment.

@details It includes classes for managing device state, handling sensor data,
         executing assigned scripts across multiple auxiliary threads, and synchronizing
         operations using a custom `ReusableBarrier` implementation, Events, and Locks.
         The `Device` class represents an individual simulated entity, `DeviceThread`
         manages its main operational loop, and `ThreadAux` handles parallel script execution.
"""


from threading import Event, Thread, Lock, Semaphore, Lock

class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.

    @details This barrier uses a two-phase approach to allow a fixed number of threads
             to wait until all participants have reached a common synchronization point,
             and then releases them all. It is designed to be reusable for subsequent
             synchronization points without re-initialization.
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
                for _ in range(self.num_threads): # Release all waiting threads.
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
                for _ in range(self.num_threads): # Release all waiting threads.
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset counter for reusability.

        self.threads_sem2.acquire() # Block until released by the last thread.



class Device(object):
    """
    @brief Represents a single device within the simulation environment.

    @details Manages the device's state, sensor data, communication with a supervisor,
             and execution of assigned scripts. Each device can have multiple auxiliary
             threads (`ThreadAux`) for parallel script execution, and a main device thread
             (`DeviceThread`) for overall coordination. Devices synchronize using shared
             barriers and locks.
    """
    
    # Class-level attributes for synchronization across all Device instances.
    bar1 = ReusableBarrier(1) # Barrier for synchronizing all devices during setup/main loop.
    event1 = Event()          # Event to signal global initialization completion.
    locck = []                # List of locks, where each index corresponds to a data location.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id An integer representing the unique identifier for the device.
        @param sensor_data A dictionary containing initial sensor data for the device.
        @param supervisor An object responsible for overseeing and coordinating devices.
        """
        
        self.timepoint_done = Event() # Event to signal completion of a timepoint's tasks.
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = [] # List to store references to all other devices in the simulation.

        # A list of Events, used to signal auxiliary threads about available work.
        self.event = []
        for _ in xrange(11): # Arbitrary number of events, possibly related to number of timepoints or script types.
            self.event.append(Event())

        
        self.nr_threads_device = 8 # Number of auxiliary threads per device.
        
        self.nr_thread_atribuire = 0 # Index to cycle through auxiliary threads for script assignment.
        
        # Barrier for synchronizing the main device thread with its auxiliary threads.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device+1)

        # The main thread responsible for the device's high-level operations.
        self.thread = DeviceThread(self)
        self.thread.start() # Start the main device thread immediately.

        # Auxiliary threads for parallel script execution.
        self.threads = []
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start() # Start all auxiliary threads.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures shared synchronization resources and initializes the global barrier.

        @details This method is called once at the beginning of the simulation.
                 For device_id 0, it initializes the global `locck` (location locks)
                 and the global `bar1` (device synchronization barrier), then signals
                 other devices that setup is complete.
        @param devices A list of all Device objects participating in the simulation.
        """
        
        self.devices = devices
        
        if self.device_id == 0: # Only the leader device performs global setup.
            # Block Logic: Initialize a pool of locks for different data locations.
            for _ in xrange(30): # Arbitrary number of locks for different locations.
                Device.locck.append(Lock())
            
            Device.bar1 = ReusableBarrier(len(devices)) # Re-initialize global barrier with correct size.
            
            Device.event1.set() # Signal that global setup is complete.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to an auxiliary thread for execution at a specific data location.

        @details This method distributes scripts round-robin among the auxiliary threads.
                 If no script is provided (script is None), it signals the device's
                 `timepoint_done` event, indicating no more tasks for the current timepoint.
        @param script The script object to assign, or None.
        @param location The data location (e.g., sensor ID) where the script will operate.
        """
        
        if script is not None:
            # Assign the script and its location to the next available auxiliary thread.
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            
            # Rotate the assignment index to distribute scripts evenly.
            self.nr_thread_atribuire = (self.nr_thread_atribuire+1)%\
            self.nr_threads_device
        else:
            self.timepoint_done.set() # Signal completion of script assignments for this timepoint.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location from the device's internal state.

        @param location The data location (e.g., sensor ID).
        @return The sensor data for the location, or None if the location is not found.
        """
        
        return self.sensor_data[location] if location in \
        self.sensor_data else None

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
        @brief Initiates the shutdown sequence for the device and its associated threads.

        @details Joins the main device thread and all auxiliary threads, ensuring
                 all ongoing operations are completed before the program exits.
        """
        
        self.thread.join() # Wait for the main device thread to finish.
        for threadd in self.threads:
            threadd.join() # Wait for all auxiliary threads to finish.



class DeviceThread(Thread):
    """
    @brief Manages the main operational loop for a Device object in the simulation.

    @details This thread is responsible for coordinating the device's activities at each
             timepoint. It communicates with the supervisor, handles global synchronization
             using `Device.bar1`, and signals its auxiliary threads (`ThreadAux`) when
             new tasks are ready.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device The `Device` object that this thread is responsible for.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None # Stores a list of neighboring devices for communication.
        self.contor = 0      # Counter for signaling specific events to auxiliary threads.

    def run(self):
        """
        @brief The main execution loop for the device's coordinating thread.

        @details This loop continuously runs through timepoints. It first waits for
                 global simulation setup to complete, then synchronizes with other
                 devices using `Device.bar1`. It fetches neighbor information from the
                 supervisor and, if the simulation is ongoing, signals its auxiliary
                 threads to begin processing. It also synchronizes with its own
                 auxiliary threads using `self.device.bar_threads_device`.
        """
        # Block Logic: Wait for the global device setup (performed by device 0) to complete.
        Device.event1.wait()

        while True:
            # Block Logic: Query the supervisor for updated neighbor information for the current timepoint.
            self.neighbours = self.device.supervisor.get_neighbours()

            # Precondition: If supervisor returns None, it signals the end of the simulation.
            if self.neighbours is None:
                # Signal auxiliary threads that the simulation is ending.
                self.device.event[self.contor].set() 
                break # Exit the main loop.

            # Block Logic: Wait for all scripts for the current timepoint to be assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Reset the event for the next timepoint.

            # Inline: Signal auxiliary threads that scripts are ready for processing.
            self.device.event[self.contor].set()
            self.contor += 1 # Increment the counter for the next event.

            # Block Logic: Synchronize the main device thread with its auxiliary threads.
            self.device.bar_threads_device.wait()

            # Block Logic: Synchronize all devices globally after internal processing.
            Device.bar1.wait()

class ThreadAux(Thread):
    """
    @brief Represents an auxiliary thread responsible for executing assigned scripts.

    @details Each `Device` can have multiple `ThreadAux` instances to parallelize
             the execution of scripts. These threads wait for signals from the
             main `DeviceThread` to start processing, acquire locks for data
             locations, execute scripts, and update data on the device and its neighbors.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a new ThreadAux instance.

        @param device The parent `Device` object that this auxiliary thread belongs to.
        """
        Thread.__init__(self)
        self.device = device
        self.script_loc = {} # Dictionary to store scripts assigned to this thread, mapped to their locations.
        self.contor = 0      # Counter to track which event to wait on, synchronizing with `DeviceThread`.

    def run(self):
        """
        @brief The main execution loop for the auxiliary thread.

        @details This loop continuously waits for a signal from the main `DeviceThread`
                 to process new scripts. Upon receiving a signal, it retrieves neighbor
                 information, iterates through its assigned scripts, acquires necessary
                 locks for data consistency, executes the script, and updates sensor data.
                 It then synchronizes with other auxiliary threads and the main thread
                 using `bar_threads_device`. The loop breaks if the simulation ends.
        """
        while True:
            # Block Logic: Wait for the main DeviceThread to signal that scripts are ready for processing.
            self.device.event[self.contor].wait()
            self.contor += 1 # Increment the counter for the next event.

            # Retrieve neighbor information from the main DeviceThread.
            neigh = self.device.thread.neighbours
            # Precondition: If neighbors is None, it means the simulation is ending.
            if neigh is None:
                break # Exit the auxiliary thread's loop.

            # Block Logic: Iterate through all scripts assigned to this auxiliary thread.
            for script in self.script_loc:
                location = self.script_loc[script]
                
                # Inline: Acquire the global lock for the specific data location to ensure exclusive access.
                Device.locck[location].acquire()
                script_data = [] # List to accumulate data for the script.

                # Block Logic: Gather data from neighboring devices for the current location.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Get data from the current device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Precondition: Execute the script only if there is data available.
                if script_data != []:
                    # Execute the assigned script with the collected data.
                    result = script.run(script_data)
                    
                    # Block Logic: Update sensor data on neighboring devices and the current device.
                    for device in neigh:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Inline: Release the global lock for the specific data location.
                Device.locck[location].release()

            # Block Logic: Synchronize this auxiliary thread with other auxiliary threads and the main DeviceThread.
            self.device.bar_threads_device.wait()
