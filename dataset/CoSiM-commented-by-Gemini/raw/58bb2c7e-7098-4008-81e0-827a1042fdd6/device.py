"""
Models a distributed, parallel computation over a network of devices.

This script simulates a network of devices that operate in synchronized time
steps. It uses a fixed pool of worker threads (`Master`) to process computational
scripts and a two-phase semaphore barrier to ensure all devices complete a
time step before proceeding to the next. The overall architecture is a variant
of the Bulk Synchronous Parallel (BSP) model.
"""

from threading import *


class Device(object):
    """
    Represents a single device (node) in the simulated network.

    Each device has its own execution thread, manages its local sensor data, and
    participates in a centralized setup of shared synchronization primitives
    (locks and a global barrier).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a device and starts its main execution thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been assigned.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that script assignment is done for a timepoint.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = 0
        # A per-device lock, used for fine-grained access to this device's data.
        self.lock = Lock()
        # A list of shared locks for specific data locations.
        self.locks = []

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for the entire device network.

        This method uses a centralized approach where the device with ID 0
        (the master) creates a set of 100 shared locks and a global reusable
        barrier, then distributes these resources to all other devices.
        """
        if(self.device_id == 0):
        	for i in xrange(100):
        		aux_lock = Lock()
        		self.add_lock(aux_lock)
        		for j in devices:
        			j.add_lock(aux_lock)
        	nr = len(devices)
        	barrier = ReusableBarrierSem(nr)
        	for i in devices:
        		i.barrier = barrier
        

    def assign_script(self, script, location):
        """
        Assigns a script to this device for execution in the current timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A `None` script signals that work assignment is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates sensor data at a location, but only if the new data is greater.
        
        This implies a "maximum value wins" conflict resolution strategy.
        """
        if location in self.sensor_data:
        	if self.sensor_data[location] < data:
        		self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()

    def get_dev_lock(self):
        """Returns the device's specific lock."""
    	return self.lock

    def add_lock(self, lock):
        """Adds a shared location lock to this device's list."""
    	self.locks.append(lock)

    def get_locks(self):
        """Returns the list of shared location locks."""
    	return self.locks

class DeviceThread(Thread):
    """
    The main worker thread for a Device.

    Orchestrates work for each time step by dispatching tasks to a local pool
    of worker (`Master`) threads and managing complex data synchronization.
    """

    def __init__(self, device):
        """Initializes the worker thread for a given device."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop for the device."""
        while True:
            # Get the set of neighbors for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            # `None` neighbors is the shutdown signal.
            if neighbours is None:
                break

            # Create a fixed-size pool of 8 worker threads for this timepoint.
            threads = []
            index = 0
            for i in xrange(8):
            	aux_thread = Master(i)
            	threads.append(aux_thread)

            # Wait until the supervisor signals that script assignment is done.
            self.device.timepoint_done.wait()
            
            # --- Data Gathering and Work Distribution Phase ---
            for (script, location) in self.device.scripts:
            	script_data = []
                # Acquire the lock for the specific data location.
            	self.device.locks[location].acquire()
            	
            	# Gather data from all neighbors for the current location.
            	for device in neighbours:
            		if device.device_id != self.device.device_id:
                                # Acquire the individual lock for each neighbor device.
            			device.lock.acquire()
            			data = device.get_data(location)
            			device.lock.release()
            			if data is not None:
            				script_data.append(data)
            	
                # Gather data from the local device.
            	self.device.lock.acquire()
            	data = self.device.get_data(location)
            	self.device.lock.release()
            	if data is not None:
            		script_data.append(data)

            	if script_data != []:
            		# Assign the script and gathered data to a worker thread (round-robin).
            		threads[index].set_worker(script, script_data)
            		threads[index].add_location(location)
            		aux_lock = self.device.locks[location]
            		threads[index].add_lock(aux_lock)
            		index = (index + 1) % 8
                # Release the location lock after dispatching the work.
            	self.device.locks[location].release()

            # --- Computation Phase ---
            for i in xrange(8):
            	threads[i].start()
            for i in xrange(8):
            	threads[i].join()
                
            # --- Results Dissemination Phase ---
            for i in xrange(8):
            	result_list = threads[i].get_result()
            	location_list = threads[i].get_location()
            	dim = len(result_list)
            	for j in xrange(dim) :
            		r = result_list[j]
            		l = location_list[j]
                        # Re-acquire the location lock to update data safely.
            		self.device.locks[l].acquire()
                        # Write the result back to all neighbors.
            		for device in neighbours:
            			device.lock.acquire()
            			device.set_data(l, r)
            			device.lock.release()
            		# Write the result to the local device.	
            		self.device.lock.acquire()
            		self.device.set_data(l, r)
            		self.device.lock.release()
            		self.device.locks[l].release()

            # --- Synchronization Phase ---
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            # Wait at the global barrier for all devices to complete the timepoint.
            self.device.barrier.wait()

class ReusableBarrierSem():
    """
    A reusable barrier implemented with two semaphores to prevent race conditions.

    This is a classic two-phase barrier construction. All threads must enter
    and leave the first phase before any thread can begin the second phase,
    which ensures that threads from a future iteration cannot get ahead of
    threads from the current iteration.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters for each phase.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        # Semaphores to control entry to and exit from the barrier phases.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)
    
    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """First synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread to arrive releases all threads waiting in this phase.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()
    
    def phase2(self):
        """Second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread to arrive releases all threads waiting in this phase.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Master(Thread):
    """
    A worker thread that executes a batch of computational scripts.

    Despite the name, this class functions as a worker in a thread pool. Each
    `DeviceThread` creates its own pool of `Master` workers for a timepoint.
    """
    def __init__(self, id):
        Thread.__init__(self)
        self.Thread_script = []
        self.Thread_script_data = []
        self.Thread_location = []
        self.Thread_lock = []
        self.Thread_result = []
        self.Thread_id = id
        self.Thread_iterations = 0

    def add_result(self, result):
        """Appends a computation result to the internal list."""
    	self.Thread_result.append(result)

    def add_script(self, script):
        """Appends a script to the batch of work."""
    	self.Thread_script.append(script)

    def add_script_data(self, script_data):
        """Appends script data to the batch of work."""
    	self.Thread_script_data.append(script_data)

    def add_location(self, location):
        """Adds the location associated with a script."""
    	self.Thread_location.append(location)

    def add_lock(self, lock):
        """Adds the lock associated with a script."""
    	self.Thread_lock.append(lock)
    
    def set_worker(self, script, script_data):
        """Convenience method to add a script and its data."""
    	self.add_script(script)
    	self.add_script_data(script_data)

    def set_iterations(self):
        """Sets the number of iterations based on the number of scripts assigned."""
    	if self.Thread_script != []:
    		self.Thread_iterations = len(self.Thread_script)
    	else:
    		self.Thread_iterations = 0

    def get_result(self):
        """Returns the list of computed results."""
    	return self.Thread_result

    def get_location(self):
        """Returns the list of locations corresponding to the results."""
    	return self.Thread_location

    def get_lock(self):
        """Returns the list of locks."""
    	return self.Thread_lock
    
    def run(self):
        """Executes all assigned scripts sequentially."""
    	self.set_iterations()
    	for i in xrange(self.Thread_iterations):
    		aux_script = self.Thread_script[i]
    		aux_script_data = self.Thread_script_data[i]
    		aux_rez = aux_script.run(aux_script_data)
    		self.add_result(aux_rez)
