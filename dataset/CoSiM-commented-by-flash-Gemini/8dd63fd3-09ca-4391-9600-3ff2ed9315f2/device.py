"""
@8dd63fd3-09ca-4391-9600-3ff2ed9315f2/device.py
@brief This script implements device behavior for a distributed system simulation,
emphasizing complex thread management, synchronization, and parallel script execution.
It features a custom reusable barrier, location-specific data locks, and a batched
approach to running scripts across worker threads.
Domain: Concurrency, Distributed Systems, Simulation, Thread Synchronization, Parallel Processing.
"""

from threading import Thread, Semaphore, Event, Lock

class ReusableBarrierSem(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads using semaphores.
    It ensures that all participating threads pause at a specific point until every thread
    reaches that point, then they all proceed.
    Algorithm: Double-phase semaphore-based barrier.
    Time Complexity: O(N) for each `wait` call where N is the number of threads, due to semaphore releases.
    Space Complexity: O(1) for internal state.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # Inline: Counter for the first phase of the barrier.
        self.count_threads1 = self.num_threads
        # Inline: Counter for the second phase of the barrier.
        self.count_threads2 = self.num_threads


        # Inline: Lock to protect access to the thread counters.
        self.counter_lock = Lock()
        # Inline: Semaphore for synchronizing threads in the first phase.
        self.threads_sem1 = Semaphore(0)
        # Inline: Semaphore for synchronizing threads in the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all participating threads
        have reached this point. It coordinates two phases of synchronization.
        """
        # Block Logic: Execute the first phase of the barrier synchronization.
        self.phase1()
        # Block Logic: Execute the second phase of the barrier synchronization.
        self.phase2()

    def phase1(self):
        """
        @brief Implements the first phase of the barrier synchronization.
        Threads decrement a counter and the last one releases all others via a semaphore.
        """
        # Block Logic: Acquire lock to safely decrement the thread counter for phase 1.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Pre-condition: If this is the last thread to reach the barrier in phase 1.
            if self.count_threads1 == 0:
                # Block Logic: Release the semaphore 'num_threads' times to unblock all waiting threads.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Inline: Reset the counter for the next use of this phase of the barrier.
                self.count_threads1 = self.num_threads
        # Post-condition: Acquire the semaphore, waiting if not yet released by the last thread.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Implements the second phase of the barrier synchronization.
        Threads decrement a counter and the last one releases all others via a semaphore.
        """
        # Block Logic: Acquire lock to safely decrement the thread counter for phase 2.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Pre-condition: If this is the last thread to reach the barrier in phase 2.
            if self.count_threads2 == 0:
                # Block Logic: Release the semaphore 'num_threads' times to unblock all waiting threads.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Inline: Reset the counter for the next use of this phase of the barrier.
                self.count_threads2 = self.num_threads
        # Post-condition: Acquire the semaphore, waiting if not yet released by the last thread.
        self.threads_sem2.acquire()

class Device(object):
    """
    @brief Represents a single device in the simulated distributed system.
    This version manages its own sensor data, executes scripts using worker threads,
    and coordinates synchronization through global barriers and location-specific locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data for various locations.
        @param supervisor: A reference to the supervisor object for coordination.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Inline: Event to signal that a new script has been assigned to the device.
        self.script_received = Event()
        # Inline: List to store assigned scripts and their locations.
        self.scripts = []
        # Inline: Event to signal that all scripts for the current timepoint have been processed.
        self.timepoint_done = Event()
        # Inline: The main thread for the device's operational logic.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Inline: Shared barrier for device-level synchronization.
        self.barrier = None
        # Inline: List of shared locks for each data location.
        self.locks = []
        
        # Inline: Maximum location ID, used to determine the size of the 'locks' array.
        self.nrlocks = max(sensor_data) if sensor_data else 0 # Handle empty sensor_data gracefully
    def __str__(self):
        """
        @brief Returns a string representation of the device.
        @return: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the device with a list of other devices in the system.
        Device 0 initializes a shared barrier and a list of shared locks for all locations,
        then distributes them to all other devices.
        @param devices: A list of all Device objects in the simulation.
        """
        
        # Block Logic: Only device with device_id 0 initializes the shared barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            
            # Block Logic: Distribute the initialized barrier to all devices.
            for _, device in enumerate(devices):
                device.barrier = self.barrier
        
        # Block Logic: Only device with device_id 0 determines the maximum number of locks needed
        # and initializes them, then distributes them to all devices.
        if self.device_id == 0:
            listmaxim = []
            # Inline: Find the maximum number of locations across all devices to size the shared lock array.
            for _, device in enumerate(devices):
                listmaxim.append(device.nrlocks)
            
            number = max(listmaxim)
            
            # Block Logic: Initialize a Lock for each possible location ID.
            for _ in range(number + 1):
                self.locks.append(Lock())
            
            # Block Logic: Distribute the shared list of locks to all devices.
            for _, device in enumerate(devices):
                device.locks = self.locks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location for this device.
        If a script is provided, it's added to the device's script list. If no script
        is provided (script is None), it signals that the current timepoint is done.
        @param script: The script object to be executed.
        @param location: The data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The location identifier for which to retrieve data.
        @return: The sensor data for the location, or None if the location is not present.
        """
        # Block Logic: Check if the location exists in sensor_data, return its value or None.
        if location in self.sensor_data:
            data = self.sensor_data[location]
        else: data = None
        return data

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.
        @param location: The location identifier for which to set data.
        @param data: The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining all its worker threads.
        """
        for thread in self.thread: # Iterates through the list of DeviceThread objects.
            thread.join()


class MiniDeviceThread(Thread):
    """
    @brief A worker thread responsible for executing a single script on sensor data
    for a specific location, considering neighboring device data. This thread acquires
    and releases a lock specific to its location during data access and modification.
    """
    
    def __init__(self, device, script, location, neighbours):
        """
        @brief Initializes a MiniDeviceThread instance.
        @param device: The Device object owning this thread.
        @param script: The script object to execute.
        @param location: The data location this script operates on.
        @param neighbours: A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        @brief Executes the assigned script. It acquires a lock for its location,
        collects data from neighbors and itself, processes it, updates data, and then releases the lock.
        """
        # Block Logic: Acquire the location-specific lock to ensure exclusive access to data.
        self.device.locks[self.location].acquire()
        script_data = []
        
        # Block Logic: Collect data from neighboring devices for the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        # Block Logic: Collect data from the current device for the specified location.
        data = self.device.get_data(self.location)
        
        # Pre-condition: Only execute the script if there is data to process.
        if data is not None:
            script_data.append(data)
        if script_data != []:
            # Inline: Execute the script with the collected data.
            result = self.script.run(script_data)
            
            # Block Logic: Update data on neighboring devices with the script's result.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Block Logic: Update data on the current device with the script's result.
            self.device.set_data(self.location, result)
        
        # Post-condition: Release the location-specific lock.
        self.device.locks[self.location].release()


class DeviceThread(Thread):
    """
    @brief The main thread responsible for a Device's operational loop.
    It manages the lifecycle of script execution, fetches neighbors, and orchestrates
    the parallel execution of scripts via MiniDeviceThreads, batching them if necessary.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device: The Device object that this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Inline: Stores the number of iterations/batches for script execution.
        self.nr_iter = None

    def run(self):
        """
        @brief The main loop for the device thread. It continuously fetches neighbors,
        waits for timepoint signals, and dispatches script execution to MiniDeviceThreads,
        handling them in batches for efficiency.
        """
        # Block Logic: Continuous operational loop for the device thread.
        while True:
            # Block Logic: Fetch the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If supervisor returns None, it indicates simulation termination.
            if neighbours is None:
                break
            
            # Block Logic: Wait until the current timepoint's scripts are ready to be processed.
            self.device.timepoint_done.wait()
            
            # Inline: Calculate the number of batches based on total scripts and a batch size of 8.
            self.nr_iter = len(self.device.scripts) / 8
            
            # Block Logic: If there are fewer than 8 scripts, execute them all directly without batching.
            if self.nr_iter == 0:
                scriptthreads = []
                # Block Logic: Create a MiniDeviceThread for each script.
                for (script, location) in self.device.scripts:
                    scriptthreads.append(MiniDeviceThread(self.device, script, location, neighbours))
                
                # Block Logic: Start all script execution threads.
                for _, thread in enumerate(scriptthreads):
                    thread.start()
                
                # Block Logic: Wait for all script execution threads to complete.
                for _, thread in enumerate(scriptthreads):
                    thread.join()
            
            
            # Block Logic: If there are 8 or more scripts, process them in batches of 8.
            else:
                count = 0 # Inline: Keeps track of the starting index for the current batch.
                size = 8  # Inline: Keeps track of the ending index for the current batch.
                # Block Logic: Iterate through the batches of scripts.
                for _ in range(self.nr_iter):
                    scriptthreads = []
                    # Block Logic: Create MiniDeviceThreads for the current batch of 8 scripts.
                    for idx in range(count, size):
                        script = self.device.scripts[idx][0]
                        location = self.device.scripts[idx][1]
                        scriptthreads.append(MiniDeviceThread(self.device, script, location, neighbours))
                    
                    # Block Logic: Start all threads in the current batch.
                    for _, thread in enumerate(scriptthreads):
                        thread.start()
	                
                    # Block Logic: Wait for all threads in the current batch to complete.
                    for _, thread in enumerate(scriptthreads):
                        thread.join()
                    count = count + 8 # Inline: Move starting index for the next batch.
                    # Block Logic: Adjust the ending index for the next batch, handling the last partial batch.
                    if size + 8 > len(self.device.scripts):
                        size = len(self.device.scripts) - size
                    else:
                        size = size + 8
            
            # Block Logic: Synchronize with other devices via the shared barrier.
            self.device.barrier.wait()
            # Inline: Clear the timepoint_done event for the next cycle.
            self.device.timepoint_done.clear()
