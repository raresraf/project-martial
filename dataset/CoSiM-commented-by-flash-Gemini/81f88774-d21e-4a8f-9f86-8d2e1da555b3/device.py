"""
@file device.py
@brief Implements a device simulation for a distributed sensing and processing system,
       featuring a custom reusable barrier for synchronization and a master-worker
       thread model for script execution.

This module is designed for Python 2.x environments, indicated by the use of `xrange` and `iteritems`.

Key Components:
- `ReusableBarrierSem`: A semaphore-based reusable barrier for synchronizing multiple threads.
- `Device`: Represents an individual device responsible for managing sensor data,
            assigning and coordinating script execution, and interacting with a supervisor
            and other devices.
- `DeviceThread`: The main worker thread for a `Device`, orchestrating script assignments,
                  neighbor interactions, and the lifecycle of `Master` threads.
- `Master`: A worker thread specialized in executing assigned scripts concurrently,
            managing its own set of scripts, data, and locks, and reporting results.
"""


from threading import *


class Device(object):
    """
    Represents a simulated device in a distributed sensing and processing network.
    Manages sensor data, coordinates script execution through a dedicated `DeviceThread`,
    and interacts with a central supervisor and other devices in the network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): Initial sensor readings for various locations.
            supervisor (object): A reference to the central supervisor managing the network.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been received by the device.
        self.script_received = Event()
        # List to hold assigned scripts, paired with their target locations.
        self.scripts = []
        # Event to signal when processing for a specific timepoint is complete.
        self.timepoint_done = Event()
        # The main worker thread for this device.
        self.thread = DeviceThread(self)
        # Initiates the device's main worker thread.
        self.thread.start()
        # Placeholder for a barrier object, to be set during setup.
        self.barrier = 0
        # A local lock for the device to protect its internal state.
        self.lock = Lock()
        # List of locks for each sensor data location, managed by the device.
        self.locks = []

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A formatted string "Device %d" where %d is the device's ID.
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures inter-device synchronization mechanisms, primarily handled by device 0.
        It initializes shared locks for sensor data locations and sets up a global
        reusable barrier for all devices.

        Args:
            devices (list): A list of all Device instances in the simulation.

        Pre-condition: This method is invoked as part of a global setup phase, usually by a supervisor.
        Invariant: After device 0 completes this method, all participating devices share common
                   location locks and a global synchronization barrier.
        """
        
        # Conditional Logic: Ensures that only device 0 orchestrates the global setup.
        if(self.device_id == 0):
            # Block Logic: Initializes a predefined number (100) of shared locks for sensor data locations.
            # Functional Utility: These locks are then distributed among all devices to ensure
            #                     thread-safe access to shared sensor data.
        	for i in xrange(100):
        		aux_lock = Lock()
        		self.add_lock(aux_lock)
        		# Block Logic: Shares each newly created lock with all other devices.
        		for j in devices:
        			j.add_lock(aux_lock)
            # Functional Utility: Calculates the total number of threads across all devices for the global barrier.
        	nr = len(devices)
            # Functional Utility: Creates a reusable barrier for global synchronization across all devices.
        	barrier = ReusableBarrierSem(nr)
            # Block Logic: Assigns the newly created global barrier to all devices.
        	for i in devices:
        		i.barrier = barrier
        

    def assign_script(self, script, location):
        """
        Assigns a script to be executed on sensor data at a specific location.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the sensor data location.

        Pre-condition: `script` is either a valid script object or `None`.
        Post-condition: If `script` is not `None`, it's added to the scripts list and `script_received` is set.
                        If `script` is `None`, `timepoint_done` is set, indicating no more scripts for the current timepoint.
        """
        # Conditional Logic: Checks if a valid script is provided or if the current timepoint is being signaled as done.
        if script is not None:
            # Functional Utility: Appends the script and its target location to the device's processing queue.
            self.scripts.append((script, location))
            # Functional Utility: Signals to worker threads that new scripts are available for processing.
            self.script_received.set()
        else:
            # Functional Utility: Signals that all scripts for the current timepoint have been assigned (or none were assigned).
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location.

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            any: The sensor data value if the location exists, otherwise `None`.
        """
        # Conditional Logic: Returns data if the location is valid, otherwise returns None.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data for a specified location, but only if the new data is greater than the existing data.

        Args:
            location (int): The identifier for the sensor data location.
            data (any): The new data value to potentially update.

        Pre-condition: The location must exist in the device's sensor data.
        Post-condition: `self.sensor_data[location]` is updated with `data` if `data` is greater than the current value.
        """
        # Conditional Logic: Updates data only if the location exists and the new data is larger.
        if location in self.sensor_data:
        	if self.sensor_data[location] < data:
        		self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates a graceful shutdown of the device's main worker thread.
        """
        # Functional Utility: Waits for the `DeviceThread` to complete its execution before allowing the device to shut down.
        self.thread.join()

    def get_dev_lock(self):
    	"""
        Retrieves the device's internal lock.

        Returns:
            Lock: The Lock object associated with this device.
        """
    	return self.lock

    def add_lock(self, lock):
    	"""
        Adds a shared lock to the device's list of location-specific locks.

        Args:
            lock (Lock): A Lock object to be added.
        """
    	self.locks.append(lock)

    def get_locks(self):
    	"""
        Retrieves the list of all location-specific locks associated with this device.

        Returns:
            list: A list of Lock objects.
        """
    	return self.locks


class DeviceThread(Thread):
    """
    The primary worker thread for a `Device` instance.
    It manages the lifecycle of script processing, coordinating with the supervisor
    and other devices, and delegating actual script execution to `Master` threads.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent Device instance this thread operates for.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop of the DeviceThread. It continuously fetches neighbor
        information, processes assigned scripts using `Master` threads, and synchronizes
        with a global barrier.
        
        Invariant: The loop continues until the supervisor signals termination by
                   returning `None` for neighbors.
        """
        while True:
            
            # Functional Utility: Retrieves the current list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Conditional Logic: If no neighbors are returned, it signifies a shutdown signal, and the thread terminates.
            if neighbours is None:
                break

            threads = []
            index = 0

            # Block Logic: Initializes a fixed pool of 8 `Master` worker threads for concurrent script execution.
            for i in xrange(8):
            	aux_thread = Master(i)
            	threads.append(aux_thread)

            # Functional Utility: Waits until the current timepoint's data assignment is complete before proceeding to script execution.
            self.device.timepoint_done.wait()
            # Block Logic: Iterates through each assigned script to prepare it for execution by `Master` threads.
            # Invariant: Each script is processed, and its associated data and lock are assigned to a `Master` thread.
            for (script, location) in self.device.scripts:
            	script_data = []
            	# Functional Utility: Acquires the specific lock for the current sensor data location to ensure exclusive access during data collection.
            	self.device.locks[location].acquire()
            	
            	# Block Logic: Collects sensor data from neighboring devices for the current location.
            	for device in neighbours:
            		# Conditional Logic: Ensures not to collect data from itself if it's in the neighbors list.
            		if device.device_id != self.device.device_id:
                        # Functional Utility: Acquires a lock on the neighbor device to safely retrieve its data.
            			device.lock.acquire()
            			data = device.get_data(location)
                        # Functional Utility: Releases the lock on the neighbor device.
            			device.lock.release()
            			# Conditional Logic: Adds collected data to `script_data` if it's valid.
            			if data is not None:
            				script_data.append(data)

            	# Block Logic: Collects sensor data from the current device for the given location.
            	self.device.lock.acquire()
            	data = self.device.get_data(location)
            	self.device.lock.release()
            	# Conditional Logic: Adds collected data to `script_data` if it's valid.
            	if data is not None:
            		script_data.append(data)

            	# Conditional Logic: If relevant script data was collected, prepares a `Master` thread for execution.
            	if script_data != []:
            		
            		# Functional Utility: Assigns the script and its data to a `Master` thread.
            		threads[index].set_worker(script, script_data)
            		# Functional Utility: Records the location associated with the script for result propagation.
            		threads[index].add_location(location)
            		# Functional Utility: Adds the location-specific lock to the `Master` thread for later release.
            		aux_lock = self.device.locks[location]
            		threads[index].add_lock(aux_lock)
            		# Functional Utility: Rotates through the `Master` threads to distribute work.
            		index = index + 1
            		if index == 8:
            			index = 0
            	# Functional Utility: Releases the location-specific lock after data collection and assignment to a `Master` thread.
            	self.device.locks[location].release()

            # Block Logic: Starts all `Master` threads to concurrently execute their assigned scripts.
            for i in xrange(8):
            	threads[i].start()
            # Block Logic: Waits for all `Master` threads to complete their execution.
            for i in xrange(8):
            	threads[i].join()
            # Block Logic: Collects results from each `Master` thread and propagates them.
            for i in xrange(8):
            	result_list = threads[i].get_result()
            	location_list = threads[i].get_location()
            	dim = len(result_list)
            	# Block Logic: Iterates through the results from a `Master` thread.
            	for j in xrange(dim) :
            		
            		r = result_list[j]
            		l = location_list[j]
            		# Functional Utility: Acquires the location-specific lock before updating sensor data.
            		self.device.locks[l].acquire()
            		# Block Logic: Propagates the script result to all neighboring devices.
            		for device in neighbours:
                        # Functional Utility: Acquires a lock on the neighbor device before setting its data.
            			device.lock.acquire()
            			device.set_data(l, r)
                        # Functional Utility: Releases the lock on the neighbor device.
            			device.lock.release()
            			
            		# Block Logic: Updates the current device's sensor data with the script result.
            		self.device.lock.acquire()
            		self.device.set_data(l, r)
            		self.device.lock.release()
            		# Functional Utility: Releases the location-specific lock after updating all relevant sensor data.
            		self.device.locks[l].release()

            # Functional Utility: Clears the script received event to prepare for the next round of assignments.
            self.device.script_received.clear()
            # Functional Utility: Clears the timepoint done event to prepare for the next timepoint.
            self.device.timepoint_done.clear()
            # Functional Utility: Synchronizes all devices at a global barrier, marking the end of the current processing cycle.
            self.device.barrier.wait()


class ReusableBarrierSem():
    """
    Implements a reusable barrier synchronization mechanism using semaphores,
    similar to a cyclic barrier, allowing a fixed number of threads to wait for each other.
    """

    
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrierSem.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                before any of them can proceed.
        """
        self.num_threads = num_threads
        # Counter for threads in the first phase of the barrier.
        self.count_threads1 = self.num_threads
        # Counter for threads in the second phase of the barrier.
        self.count_threads2 = self.num_threads
        # Lock to protect access to the thread counters, ensuring atomic updates.
        self.counter_lock = Lock()               
        # Semaphore for threads waiting in phase 1. Initialized to 0, so all threads
        # will block until explicitly released.
        self.threads_sem1 = Semaphore(0)         
        # Semaphore for threads waiting in phase 2. Initialized to 0.
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all `num_threads`
        have also called `wait()`. This method orchestrates both phases of the barrier.
        """
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """
        The first phase of the barrier. Threads decrement a shared counter, and the
        last thread to reach the counter releases all threads waiting for phase 1.

        Pre-condition: All threads entering this phase are ready to synchronize.
        Invariant: `count_threads1` accurately reflects the number of threads yet to reach the barrier in this phase.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            # Conditional Logic: If this is the last thread to arrive at phase 1.
            # Functional Utility: The final thread arriving at the barrier signals all other waiting threads to proceed.
            if self.count_threads1 == 0:
                # Block Logic: Releases all `num_threads` from the first semaphore.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Functional Utility: Resets the counter for the next cycle of phase 1.
                self.count_threads1 = self.num_threads
        
        # Functional Utility: Blocks the current thread until it is released by the last thread in phase 1.
        self.threads_sem1.acquire()
    
    def phase2(self):
        """
        The second phase of the barrier. Threads decrement a shared counter, and the
        last thread to reach the counter releases all threads waiting for phase 2.
        This phase is critical for ensuring the barrier's reusability.

        Pre-condition: All threads have successfully completed phase 1 of the current cycle.
        Invariant: `count_threads2` accurately reflects the number of threads yet to reach the barrier in this phase.
        Functional Utility: This phase prevents threads from "slipping" into the next cycle's phase 1
                            before all threads have completed the current cycle.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            # Conditional Logic: If this is the last thread to arrive at phase 2.
            # Functional Utility: The final thread arriving at the second phase signals all other waiting threads to proceed, enabling reuse.
            if self.count_threads2 == 0:
                # Block Logic: Releases all `num_threads` from the second semaphore.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Functional Utility: Resets the counter for the next cycle of phase 2.
                self.count_threads2 = self.num_threads
        
        # Functional Utility: Blocks the current thread until it is released by the last thread in phase 2.
        self.threads_sem2.acquire()


class Master(Thread):
    """
    A worker thread class responsible for executing a collection of scripts.
    It manages the scripts, their associated data, locations, and locks,
    and reports the results of script executions.
    """

    def __init__(self, id):
        """
        Initializes a Master thread.

        Args:
            id (int): A unique identifier for this Master thread.
        """
        Thread.__init__(self)
        # List of scripts assigned to this Master thread.
        self.Thread_script = []
        # List of script data corresponding to each assigned script.
        self.Thread_script_data = []
        # List of sensor data locations corresponding to each assigned script.
        self.Thread_location = []
        # List of locks associated with each script's location.
        self.Thread_lock = []
        # List to store the results of executed scripts.
        self.Thread_result = []
        # Unique identifier for this Master thread.
        self.Thread_id = id
        # Number of iterations (scripts) this Master thread needs to process.
        self.Thread_iterations = 0

    def add_result(self, result):
    	"""
        Adds a script execution result to the internal list.

        Args:
            result (any): The result returned by a script.
        """
    	self.Thread_result.append(result)

    def add_script(self, script):
    	"""
        Adds a script to the internal list for execution.

        Args:
            script (object): The script object.
        """
    	self.Thread_script.append(script)

    def add_script_data(self, script_data):
    	"""
        Adds script data to the internal list, corresponding to an assigned script.

        Args:
            script_data (list): The input data for a script.
        """
    	self.Thread_script_data.append(script_data)

    def add_location(self, location):
    	"""
        Adds a sensor data location to the internal list, corresponding to an assigned script.

        Args:
            location (int): The identifier for the sensor data location.
        """
    	self.Thread_location.append(location)

    def add_lock(self, lock):
    	"""
        Adds a lock to the internal list, associated with a script's location.

        Args:
            lock (Lock): The Lock object.
        """
    	self.Thread_lock.append(lock)
    
    def set_worker(self, script, script_data):
    	"""
        Assigns a single script and its data to this Master thread for processing.

        Args:
            script (object): The script object.
            script_data (list): The input data for the script.
        """
    	self.add_script(script)
    	self.add_script_data(script_data)

    def set_iterations(self):
    	"""
        Calculates and sets the number of scripts (iterations) this Master thread needs to process.
        """
    	# Conditional Logic: Sets iterations to the number of assigned scripts if any, otherwise 0.
    	if self.Thread_script != []:
    		self.Thread_iterations = len(self.Thread_script)
    	else:
    		self.Thread_iterations = 0

    def get_result(self):
    	"""
        Retrieves the list of results from all executed scripts.

        Returns:
            list: A list containing the results of script executions.
        """
    	return self.Thread_result

    def get_location(self):
    	"""
        Retrieves the list of sensor data locations associated with the executed scripts.

        Returns:
            list: A list of integer location identifiers.
        """
    	return self.Thread_location

    def get_lock(self):
    	"""
        Retrieves the list of locks associated with the scripts' locations.

        Returns:
            list: A list of Lock objects.
        """
    	return self.Thread_lock
    
    def run(self):
    	"""
        The main execution loop for the Master thread. It processes all assigned
        scripts sequentially and stores their results.
        """
    	self.set_iterations()
    	# Block Logic: Iterates through each assigned script and executes it with its corresponding data.
    	for i in xrange(self.Thread_iterations):
    		aux_script = self.Thread_script[i]
    		aux_script_data = self.Thread_script_data[i]
    		# Functional Utility: Executes the script and stores its result.
    		aux_rez = aux_script.run(aux_script_data)
    		self.add_result(aux_rez)
