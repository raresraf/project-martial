"""
@file device.py
@brief Implements core components for a distributed system, likely a simulation or sensor network.
This module defines Device objects that can communicate and process data concurrently
using multiple threads, employing synchronization primitives like locks, events, and barriers.
It models a system where individual devices process sensor data and interact with neighbors
under the orchestration of a supervisor.

Key classes:
- Device: Represents an individual processing unit with sensor data, scripts, and synchronization.
- DeviceThread: Manages the lifecycle and execution logic for a Device.
- ReusableBarrierSem: Provides a reusable barrier for thread synchronization.
- Master: A helper thread class for executing scripts on data.
"""

from threading import *


class Device(object):
    """
    @brief Represents a single device in the distributed system.
    Each device has a unique ID, manages its sensor data, interacts with a supervisor,
    and processes assigned scripts in a dedicated thread. It includes various
    synchronization primitives for coordinated operation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id (int): A unique identifier for the device.
        @param sensor_data (dict): A dictionary holding sensor readings for different locations.
        @param supervisor (Supervisor): A reference to the central supervisor managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal when new scripts are assigned.
        self.scripts = []  # List to hold (script, location) tuples.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's processing.
        self.thread = DeviceThread(self)  # The dedicated thread for this device's operations.
        self.thread.start()  # Starts the device's operational thread.
        self.barrier = 0  # Placeholder for a reusable barrier object, initialized by supervisor.
        self.lock = Lock()  # A device-specific lock for protecting its internal state.
        self.locks = []  # A list of shared locks used for data locations.

    def __str__(self):
        """
        @brief Provides a string representation of the Device.

        @return str: A formatted string indicating the device ID.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (locks and barriers) across devices.
        This method is typically called by the supervisor or a coordinating entity.
        Only device_id 0 performs the global setup.

        @param devices (list): A list of all Device instances in the system.
        """
        
        if(self.device_id == 0):
            # Block Logic: Device 0 initializes a set of shared locks for sensor data locations.
            # It creates 100 auxiliary locks and distributes them among all devices.
            for i in xrange(100):  # Creates 100 locks, assuming 100 data locations.
                aux_lock = Lock()
                self.add_lock(aux_lock)
                for j in devices:  # Distributes the same lock to all devices for the same location.
                    j.add_lock(aux_lock)
            nr = len(devices)  # Total number of devices.
            # Initializes a reusable barrier for all devices to synchronize.
            barrier = ReusableBarrierSem(nr)
            # Assigns the same barrier instance to all devices.
            for i in devices:
                i.barrier = barrier
        

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        @param script (callable): The script (function or object with a run method) to execute.
        @param location (int): The identifier of the data location the script operates on.
                               If script is None, signals that timepoint processing is done.
        """
        if script is not None:
            self.scripts.append((script, location))  # Stores the script and its target location.
            self.script_received.set()  # Signals that new scripts have been received.
        else:
            self.timepoint_done.set()  # Signals that all scripts for the current timepoint are assigned.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location (int): The identifier of the data location.
        @return any: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a given location if the new data is greater.
        This implies a specific update policy where data only increases.

        @param location (int): The identifier of the data location to update.
        @param data (any): The new data value.
        """
        if location in self.sensor_data:
            # Conditional Logic: Updates data only if the new value is greater than the current.
            if self.sensor_data[location] < data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread, waiting for its completion.
        """
        self.thread.join()

    def get_dev_lock(self):
        """
        @brief Returns the device's internal lock.

        @return Lock: The threading.Lock object associated with this device.
        """
    	return self.lock

    def add_lock(self, lock):
        """
        @brief Adds a shared lock to the device's list of locks.
        These locks are typically associated with specific data locations.

        @param lock (Lock): The threading.Lock object to add.
        """
    	self.locks.append(lock)

    def get_locks(self):
        """
        @brief Returns the list of shared locks held by the device.

        @return list: A list of threading.Lock objects.
        """
    	return self.locks

class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a Device.
    This thread continuously monitors for new scripts, processes them,
    updates sensor data based on script results and neighbor interactions,
    and synchronizes with other device threads using a barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device (Device): The Device object this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop of the DeviceThread.
        It continuously fetches neighbor information, waits for scripts,
        processes them by collecting data from neighbors, executing scripts
        in master threads, and then propagating results. It concludes each
        cycle by synchronizing via a barrier.
        """
        while True:
            # Block Logic: Continuously check for neighbor updates and process scripts.
            # Invariant: Each iteration represents a timepoint or processing cycle.
            
            # Retrieves neighbor devices from the supervisor. If no neighbors are returned, the system is shutting down.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break  # Terminates the thread if no neighbors are found (supervisor signals shutdown).

            # Initializes a list of Master threads for concurrent script execution.
            threads = []
            # Index to distribute scripts among Master threads.
            index = 0

            # Block Logic: Initializes a fixed pool of 8 Master threads.
            for i in xrange(8):
            	aux_thread = Master(i)
            	threads.append(aux_thread)

            # Synchronization: Waits until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()
            
            # Block Logic: Processes each assigned script.
            for (script, location) in self.device.scripts:
            	script_data = []  # Data collected for the current script.
            	# Acquires the lock specific to the current data location to prevent race conditions.
            	self.device.locks[location].acquire()
            	
            	# Block Logic: Collects relevant data from neighboring devices for the script.
            	for device in neighbours:
            		# Avoids processing self as a neighbor.
            		if device.device_id != self.device.device_id:
            			# Acquires the neighbor's device lock to safely access its data.
            			device.lock.acquire()
            			data = device.get_data(location)  # Gets data from the neighbor at the specified location.
            			device.lock.release()  # Releases the neighbor's device lock.
            			if data is not None:
            				script_data.append(data) # Appends collected data if valid.

            	# Block Logic: Collects data from its own device.
            	self.device.lock.acquire()  # Acquires its own device lock.
            	data = self.device.get_data(location)
            	self.device.lock.release()  # Releases its own device lock.
            	if data is not None:
            		script_data.append(data) # Appends collected data if valid.

            	# Conditional Logic: If data was collected, assigns the script and data to a Master thread.
            	if script_data != []:
            		# Assigns the script and collected data to a Master thread.
            		threads[index].set_worker(script, script_data)
            		threads[index].add_location(location) # Informs the Master thread about the location.
            		aux_lock = self.device.locks[location] # Passes the shared lock to the Master thread.
            		threads[index].add_lock(aux_lock)
            		index = index + 1 # Moves to the next Master thread in the pool.
            		# Wraps around the Master thread pool if the end is reached.
            		if index == 8:
            			index = 0
            	self.device.locks[location].release() # Releases the location-specific lock.

            # Block Logic: Starts all Master threads to execute their assigned scripts concurrently.
            for i in xrange(8):
            	threads[i].start()
            # Block Logic: Waits for all Master threads to complete their execution.
            for i in xrange(8):
            	threads[i].join()
            
            # Block Logic: Collects results from Master threads and updates sensor data on all relevant devices.
            for i in xrange(8):
            	result_list = threads[i].get_result()
            	location_list = threads[i].get_location()
            	dim = len(result_list)
            	for j in xrange(dim) :
            		
            		r = result_list[j] # Result from the script.
            		l = location_list[j] # Location associated with the result.
            		self.device.locks[l].acquire() # Acquires the shared lock for the location.
            		# Block Logic: Propagates the result to all neighboring devices.
            		for device in neighbours:
            			device.lock.acquire() # Acquires neighbor's device lock.
            			device.set_data(l, r) # Updates neighbor's data.
            			device.lock.release() # Releases neighbor's device lock.
            			
            		# Block Logic: Updates its own device's sensor data with the result.
            		self.device.lock.acquire() # Acquires its own device lock.
            		self.device.set_data(l, r) # Updates its own data.
            		self.device.lock.release() # Releases its own device lock.
            		self.device.locks[l].release() # Releases the shared lock for the location.

            # Resets the event flags for the next timepoint.
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            # Synchronization: Waits at the reusable barrier until all device threads have completed their timepoint processing.
            self.device.barrier.wait()

class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier using semaphores for thread synchronization.
    This barrier allows a fixed number of threads to wait for each other before
    proceeding, and can be reused multiple times. It employs a two-phase
    approach to ensure proper synchronization.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.

        @param num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for the first phase.
        self.count_threads2 = self.num_threads # Counter for the second phase.
        self.counter_lock = Lock()             # Lock to protect the counters.
        self.threads_sem1 = Semaphore(0)       # Semaphore for threads waiting in phase 1.
        self.threads_sem2 = Semaphore(0)       # Semaphore for threads waiting in phase 2.
    
    def wait(self):
        """
        @brief Blocks the calling thread until all 'num_threads' have reached the barrier.
        This method executes both phases of the barrier.
        """
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """
        @brief The first phase of the barrier.
        Threads decrement a counter. The last thread to reach zero releases all
        waiting threads in this phase.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread in phase 1 releases all others.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset counter for reuse.
        
        self.threads_sem1.acquire() # Threads wait here until released.
    
    def phase2(self):
        """
        @brief The second phase of the barrier.
        Similar to phase 1, but for the second set of counters and semaphores,
        ensuring all threads from the previous cycle have fully exited the barrier logic
        before the next cycle can begin.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread in phase 2 releases all others.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset counter for reuse.
        
        self.threads_sem2.acquire() # Threads wait here until released.

class Master(Thread):
    """
    @brief A helper thread class designed to execute a specific script with its data.
    This class acts as a worker, taking a script and its associated data, running the script,
    and storing the results. It's used within DeviceThread to parallelize script execution.
    """
    def __init__(self, id):
        """
        @brief Initializes a new Master thread.

        @param id (int): An identifier for this Master thread.
        """
        Thread.__init__(self)
        self.Thread_script = []       # List to hold scripts to be executed.
        self.Thread_script_data = []  # List to hold data for the scripts.
        self.Thread_location = []     # List to hold locations associated with the scripts.
        self.Thread_lock = []         # List to hold locks associated with the scripts/locations.
        self.Thread_result = []       # List to store results after script execution.
        self.Thread_id = id           # Unique ID for the Master thread.
        self.Thread_iterations = 0    # Number of scripts assigned to this Master.

    def add_result(self, result):
        """
        @brief Adds a script execution result to the internal list.

        @param result (any): The result returned by an executed script.
        """
    	self.Thread_result.append(result)

    def add_script(self, script):
        """
        @brief Adds a script to the list of scripts to be executed.

        @param script (callable): The script to add.
        """
    	self.Thread_script.append(script)

    def add_script_data(self, script_data):
        """
        @brief Adds script-specific data to the internal list.

        @param script_data (list): The data to be passed to the script.
        """
    	self.Thread_script_data.append(script_data)

    def add_location(self, location):
        """
        @brief Adds a data location associated with a script.

        @param location (int): The identifier of the data location.
        """
    	self.Thread_location.append(location)

    def add_lock(self, lock):
        """
        @brief Adds a lock associated with a script's data location.

        @param lock (Lock): The threading.Lock object to add.
        """
    	self.Thread_lock.append(lock)
    
    def set_worker(self, script, script_data):
        """
        @brief Configures the Master thread with a script and its data for execution.

        @param script (callable): The script to execute.
        @param script_data (list): The data to be processed by the script.
        """
    	self.add_script(script)
    	self.add_script_data(script_data)

    def set_iterations(self):
        """
        @brief Sets the number of scripts this Master thread needs to execute.
        """
    	if self.Thread_script != []:
    		self.Thread_iterations = len(self.Thread_script)
    	else:
    		self.Thread_iterations = 0

    def get_result(self):
        """
        @brief Returns the list of results obtained from executing the assigned scripts.

        @return list: A list of results.
        """
    	return self.Thread_result

    def get_location(self):
        """
        @brief Returns the list of locations associated with the executed scripts.

        @return list: A list of location identifiers.
        """
    	return self.Thread_location

    def get_lock(self):
        """
        @brief Returns the list of locks associated with the executed scripts.

        @return list: A list of threading.Lock objects.
        """
    	return self.Thread_lock
    
    def run(self):
        """
        @brief The main execution method for the Master thread.
        It runs all assigned scripts with their respective data and stores the results.
        """
    	self.set_iterations()
        # Block Logic: Iterates through each script assigned to this Master thread and executes it.
        # Invariant: 'i' represents the index of the current script being processed by this Master.
    	for i in xrange(self.Thread_iterations):
    		aux_script = self.Thread_script[i]       # Retrieves the script.
    		aux_script_data = self.Thread_script_data[i] # Retrieves the data for the script.
    		aux_rez = aux_script.run(aux_script_data)    # Executes the script.
    		self.add_result(aux_rez)                 # Stores the result.
