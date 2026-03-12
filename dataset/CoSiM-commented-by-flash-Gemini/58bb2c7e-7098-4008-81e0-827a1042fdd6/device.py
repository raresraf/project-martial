


"""
This module implements a simulated distributed device system with a layered threading model.

It defines classes for:
- `Device`: Represents a single device, managing sensor data and orchestrating operations.
- `DeviceThread`: The main thread for a `Device`, acting as an orchestrator that
  spawns `Master` threads for concurrent script execution.
- `ReusableBarrierSem`: A reusable barrier for synchronizing multiple threads in phases.
- `Master`: A worker thread spawned by `DeviceThread` to execute a subset of scripts.

The system features a complex shared lock management strategy and uses `threading.Lock`
and `threading.Semaphore` for synchronization.
"""

from threading import *


class Device(object):
    """
    Represents a single device within the simulated distributed environment.
    Each device manages its own sensor data, interacts with a supervisor,
    and orchestrates multi-threaded script execution through its `DeviceThread`.
    It maintains its own lock for `sensor_data` access and participates in
    a system of shared global locks (`self.locks`).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for the device.
            supervisor (object): A reference to a supervisor object for inter-device communication.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are assigned.
        self.scripts = [] # List to hold assigned scripts.
        self.timepoint_done = Event() # Event to signal when the current timepoint's work is complete.
        self.thread = DeviceThread(self) # The main thread for this device.
        self.thread.start() # Start the DeviceThread.
        self.barrier = 0 # Placeholder for the ReusableBarrierSem assigned in setup_devices.
        self.lock = Lock() # Lock to protect this device's sensor_data.
        self.locks = [] # List of shared global locks, managed by the master device (device_id 0).

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared locks and a global barrier across all devices.
        This method is designed to be called only by the device with `device_id == 0`.
        It creates 100 global locks and a `ReusableBarrierSem` for synchronization among devices.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        # Block Logic: Only the device with device_id 0 performs this setup.
        if(self.device_id == 0):
        	# Inline: Creates 100 shared Lock objects.
        	for i in xrange(100):
        		aux_lock = Lock()
        		self.add_lock(aux_lock) # Add to this device's shared locks list.
        		# Inline: Distribute each created lock to all other devices.
        		for j in devices:
        			j.add_lock(aux_lock)
        	nr = len(devices) # Get the total number of devices.
        	# Inline: Creates a global ReusableBarrierSem for synchronization across all devices.
        	barrier = ReusableBarrierSem(nr)
        	# Inline: Assigns the created barrier to all devices.
        	for i in devices:
        		i.barrier = barrier
        

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific data location.

        Args:
            script (object): The script object to be executed.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
            self.script_received.set() # Signal that a new script has been received.
        else:
            self.timepoint_done.set() # If no script, signal that the timepoint's script assignment is done.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location.
        This implementation specifically updates the data only if the new `data`
        value is greater than the existing value, implementing a "max" aggregation.

        Args:
            location (int): The location identifier for which to set data.
            data (any): The new data value to be set.
        """
        if location in self.sensor_data:
        	if self.sensor_data[location] < data:
        		self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device by waiting for its main `DeviceThread` to complete.
        """
        self.thread.join() # Wait for the DeviceThread to finish its execution.

    def get_dev_lock(self):
    	"""
        Returns the device's private lock used to protect its `sensor_data`.

        Returns:
            threading.Lock: The private lock for this device.
        """
    	return self.lock

    def add_lock(self, lock):
    	"""
        Adds a shared global lock to the device's list of locks.
        These locks are typically managed by the master device and distributed.

        Args:
            lock (threading.Lock): The shared lock to add.
        """
    	self.locks.append(lock)

    def get_locks(self):
    	"""
        Returns the list of shared global locks available to this device.

        Returns:
            list: A list of `threading.Lock` objects.
        """
    	return self.locks

class DeviceThread(Thread):
    """
    The main orchestrating thread for a `Device`.
    This thread is responsible for managing communication with the supervisor,
    delegating script execution to a pool of `Master` worker threads,
    collecting their results, and managing synchronization across timepoints.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the `DeviceThread`.
        It continuously fetches neighbor data, distributes scripts to worker `Master` threads,
        manages their execution, collects results, and ensures synchronization
        across timepoints using global locks and barriers.
        """
        while True:
            # Block Logic: Fetch neighbor information from the supervisor.
            # If the supervisor returns None, it signals termination for this device.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Exit the loop, signaling device shutdown.

            threads = [] # List to hold the `Master` worker threads.
            index = 0    # Used for round-robin assignment of scripts to `Master` threads.

            # Inline: Create 8 `Master` worker threads.
            for i in xrange(8):
            	aux_thread = Master(i)
            	threads.append(aux_thread)

            # Block Logic: Wait until the `timepoint_done` event is set, indicating scripts are assigned.
            self.device.timepoint_done.wait()
            # Block Logic: Distribute assigned scripts to `Master` threads for processing.
            for (script, location) in self.device.scripts:
            	script_data = [] # Data to be passed to the script.
            	# Inline: Acquire the global lock for the specific location before accessing data.
            	self.device.locks[location].acquire()
            	
            	# Block Logic: Collect data from neighboring devices for the current script's location.
            	for device in neighbours:
            		if device.device_id != self.device.device_id: # Exclude self, already handled separately.
            			device.lock.acquire() # Acquire neighbor's private lock for its sensor data.
            			data = device.get_data(location)
            			device.lock.release() # Release neighbor's private lock.
            			if data is not None:
            				script_data.append(data)

            	# Block Logic: Collect data from this device's own sensor data.
            	self.device.lock.acquire() # Acquire this device's private lock.
            	data = self.device.get_data(location)
            	self.device.lock.release() # Release this device's private lock.
            	if data is not None:
            		script_data.append(data)

            	# Block Logic: If data is available, assign the script and its data to a `Master` thread.
            	if script_data != []:
            		threads[index].set_worker(script, script_data) # Assign script and data.
            		threads[index].add_location(location) # Store the location.
            		aux_lock = self.device.locks[location] # Get the global lock for this location.
            		threads[index].add_lock(aux_lock) # Assign the lock to the Master thread.
            		index = index + 1 # Move to the next `Master` thread for round-robin assignment.
            		if index == 8: # Reset index if all 8 `Master` threads have been assigned.
            			index = 0
            	self.device.locks[location].release() # Release the global lock for the current location.

            # Block Logic: Start all `Master` threads and wait for their completion.
            for i in xrange(8):
            	threads[i].start()
            for i in xrange(8):
            	threads[i].join()
            
            # Block Logic: Collect results from `Master` threads and update device sensor data.
            for i in xrange(8):
            	result_list = threads[i].get_result()
            	location_list = threads[i].get_location()
            	dim = len(result_list)
            	for j in xrange(dim) : # Iterate through results processed by this `Master` thread.
            		
            		r = result_list[j] # The calculated result.
            		l = location_list[j] # The location associated with the result.
            		self.device.locks[l].acquire() # Acquire global lock for this location before updating data.
            		# Block Logic: Update sensor data for all neighbors and this device.
            		for device in neighbours:
            			device.lock.acquire() # Acquire neighbor's private lock.
            			device.set_data(l, r) # Update neighbor's data.
            			device.lock.release() # Release neighbor's private lock.
            			
            		self.device.lock.acquire() # Acquire this device's private lock.
            		self.device.set_data(l, r) # Update this device's data.
            		self.device.lock.release() # Release this device's private lock.
            		self.device.locks[l].release() # Release global lock for this location.

            # Block Logic: Clear events and synchronize devices using the barrier.
            self.device.script_received.clear() # Clear script received event.
            self.device.timepoint_done.clear() # Clear timepoint done event.
            self.device.barrier.wait() # Wait on the global device barrier for all devices to finish.

class ReusableBarrierSem():
    """
    A reusable barrier synchronization mechanism for multiple threads using semaphores.
    This barrier allows a fixed number of threads to wait at a synchronization point,
    and once all threads arrive, they are all released simultaneously. It can then
    be reused for subsequent synchronization points.
    """

    def __init__(self, num_threads):
        """
        Initializes the reusable barrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that will participate
                               in the barrier synchronization.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()                # Lock to protect the shared counters.
        self.threads_sem1 = Semaphore(0)          # Semaphore for the first phase of threads to wait on.
        self.threads_sem2 = Semaphore(0)          # Semaphore for the second phase of threads to wait on.
    
    def wait(self):
        """
        Causes the calling thread to wait until all threads have reached this barrier.
        This method orchestrates a two-phase synchronization to ensure reusability
        without deadlocks.
        """
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """
        The first phase of the barrier synchronization.
        Threads decrement a shared counter, and the last thread to reach zero
        releases all waiting threads for this phase.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        
        self.threads_sem1.acquire()
    
    def phase2(self):
        """
        The second phase of the barrier synchronization, necessary for reusability.
        Similar to phase1, threads decrement a counter, and the last thread
        releases all waiting threads for this phase.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        
        self.threads_sem2.acquire()

class Master(Thread):
    """
    A worker thread spawned by `DeviceThread` to execute a subset of scripts concurrently.
    Each `Master` thread is responsible for running its assigned scripts, storing
    results, and managing associated metadata like locations and locks.
    """
    def __init__(self, id):
        """
        Initializes a `Master` worker thread.

        Args:
            id (int): A unique identifier for this `Master` thread.
        """
        Thread.__init__(self)
        self.Thread_script = [] # List to store scripts assigned to this worker.
        self.Thread_script_data = [] # List to store input data for each script.
        self.Thread_location = [] # List to store locations associated with each script.
        self.Thread_lock = [] # List to store locks associated with each script's location.
        self.Thread_result = [] # List to store results after executing each script.
        self.Thread_id = id # Unique ID of this master thread.
        self.Thread_iterations = 0 # Number of scripts assigned to this worker.

    def add_result(self, result):
    	"""
        Adds a script execution result to the `Thread_result` list.

        Args:
            result (any): The result obtained from executing a script.
        """
    	self.Thread_result.append(result)

    def add_script(self, script):
    	"""
        Adds a script object to the `Thread_script` list for this worker to execute.

        Args:
            script (object): The script object.
        """
    	self.Thread_script.append(script)

    def add_script_data(self, script_data):
    	"""
        Adds the input data required for a script to the `Thread_script_data` list.

        Args:
            script_data (list): The list of data for the script.
        """
    	self.Thread_script_data.append(script_data)

    def add_location(self, location):
    	"""
        Adds a data location associated with a script to the `Thread_location` list.

        Args:
            location (int): The data location identifier.
        """
    	self.Thread_location.append(location)

    def add_lock(self, lock):
    	"""
        Adds a shared lock object associated with a script's location to the `Thread_lock` list.

        Args:
            lock (threading.Lock): The shared lock object.
        """
    	self.Thread_lock.append(lock)
    
    def set_worker(self, script, script_data):
    	"""
        Assigns a single script and its corresponding data to this worker.

        Args:
            script (object): The script object to be executed.
            script_data (list): The input data for the script.
        """
    	self.add_script(script)
    	self.add_script_data(script_data)

    def set_iterations(self):
    	"""
        Calculates and sets the number of scripts this worker needs to execute (`Thread_iterations`).
        This is based on the number of scripts currently assigned to `Thread_script`.
        """
    	if self.Thread_script != []:
    		self.Thread_iterations = len(self.Thread_script)
    	else:
    		self.Thread_iterations = 0

    def get_result(self):
    	"""
        Retrieves the list of results from all scripts executed by this worker.

        Returns:
            list: A list of execution results.
        """
    	return self.Thread_result

    def get_location(self):
    	"""
        Retrieves the list of locations associated with the scripts executed by this worker.

        Returns:
            list: A list of location identifiers.
        """
    	return self.Thread_location

    def get_lock(self):
    	"""
        Retrieves the list of shared locks associated with the scripts executed by this worker.

        Returns:
            list: A list of `threading.Lock` objects.
        """
    	return self.Thread_lock
    
    def run(self):
        """
        The main execution method for the `Master` worker thread.
        It sets the number of iterations based on assigned scripts, then iterates through
        each script, executes it with its corresponding data, and stores the result.
        """
    	self.set_iterations() # Determine the number of scripts to process.
    	# Block Logic: Iterate through all assigned scripts, execute them, and store results.
    	for i in xrange(self.Thread_iterations):
    		aux_script = self.Thread_script[i] # Get the current script object.
    		aux_script_data = self.Thread_script_data[i] # Get the data for the current script.
    		aux_rez = aux_script.run(aux_script_data) # Execute the script with its data.
    		self.add_result(aux_rez) # Store the result.
