"""
Models a distributed sensor network simulation using multithreading.

This script simulates a network of devices that process sensor data in discrete,
synchronized time steps. Each device runs in its own thread, communicates with
its neighbors, and performs computations on aggregated data using a pool of
worker threads.

Classes:
    Device: Represents a node in the sensor network, managing its data and state.
    DeviceThread: The main control loop for a Device, orchestrating the
                  processing for each time step.
    ReusableBarrierSem: A classic two-phase reusable barrier implementation to
                        synchronize all device threads at the end of a time step.
    Master: A worker thread responsible for executing a single computational
            script on a set of data.
"""

from threading import *


class Device(object):
    """Represents a single device or node in the simulated network.

    Each device holds its own sensor data, has a unique ID, and runs a dedicated
    thread (DeviceThread) to handle its operations. It uses various synchronization
    primitives to coordinate with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local
                                sensor readings, typically mapping locations to values.
            supervisor (object): An external object that manages the network
                                 topology, providing neighbor information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been assigned for the current time step.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that a timepoint simulation can begin.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # A reusable barrier for synchronizing all devices at the end of a time step.
        self.barrier = 0
        # A lock to protect this device's own sensor_data.
        self.lock = Lock()
        # A list of locks, indexed by location, to protect shared data locations.
        self.locks = []

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources across all devices.

        This method should be called on a single device (e.g., device 0) to set up
        shared locks for each data location and a shared barrier for
        synchronization.
        """
        # This setup is performed only by the master device (id 0).
        if(self.device_id == 0):
        	# Create 100 shared locks, one for each potential data location.
        	for i in xrange(100):
        		aux_lock = Lock()
        		self.add_lock(aux_lock)
        		for j in devices:
        			j.add_lock(aux_lock)
        	# Create and distribute a reusable barrier to all devices.
        	nr = len(devices)
        	barrier = ReusableBarrierSem(nr)
        	for i in devices:
        		i.barrier = barrier
        

    def assign_script(self, script, location):
        """Assigns a computational script to be executed for a specific location.

        Args:
            script (object): The script object to be executed. Must have a `run` method.
            location (any): The data location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of a timepoint's script assignments.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data for a location, only if the new data has a higher value.
        """
        if location in self.sensor_data:
            # This logic suggests the data represents a maximum value found so far.
        	if self.sensor_data[location] < data:
        		self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()

    def get_dev_lock(self):
        """Returns the device's personal lock."""
    	return self.lock

    def add_lock(self, lock):
        """Adds a shared location lock to the device's list."""
    	self.locks.append(lock)

    def get_locks(self):
        """Returns the list of shared location locks."""
    	return self.locks

class DeviceThread(Thread):
    """The main worker thread for a Device."""

    def __init__(self, device):
        """Initializes the thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop for the device.
        
        This loop continuously executes time steps. In each step, it aggregates
        data from neighbors, runs computations in parallel, updates data, and
        synchronizes with all other devices before starting the next step.
        """
        while True:
            # Get the list of neighbors for the current time step.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbors is the signal to terminate.
                break

            # Create a pool of 8 worker threads for this time step.
            threads = []
            index = 0
            for i in xrange(8):
            	aux_thread = Master(i)
            	threads.append(aux_thread)

            # Wait until the supervisor signals that all scripts for this time step are assigned.
            self.device.timepoint_done.wait()
            
            # --- AGGREGATION AND WORK ASSIGNMENT PHASE ---
            for (script, location) in self.device.scripts:
            	script_data = []
                # Lock this specific location to prevent race conditions with other devices.
            	self.device.locks[location].acquire()
            	
            	# Collect data for the current location from all neighbors.
            	for device in neighbours:
            		if device.device_id != self.device.device_id:
            			device.lock.acquire()
            			data = device.get_data(location)
            			device.lock.release()
            			if data is not None:
            				script_data.append(data)

            	# Collect data for the current location from this device itself.
            	self.device.lock.acquire()
            	data = self.device.get_data(location)
            	self.device.lock.release()
            	if data is not None:
            		script_data.append(data)

                # Assign the script and aggregated data to a worker thread.
            	if script_data != []:
            		threads[index].set_worker(script, script_data)
            		threads[index].add_location(location)
            		aux_lock = self.device.locks[location]
            		threads[index].add_lock(aux_lock)
            		index = (index + 1) % 8 # Round-robin assignment.
            	
                # Release the lock for this location.
            	self.device.locks[location].release()

            # --- COMPUTATION PHASE ---
            for i in xrange(8):
            	threads[i].start()
            for i in xrange(8):
            	threads[i].join()

            # --- DATA DISSEMINATION PHASE ---
            for i in xrange(8):
            	result_list = threads[i].get_result()
            	location_list = threads[i].get_location()
            	dim = len(result_list)
            	for j in xrange(dim) :
            		r = result_list[j]
            		l = location_list[j]
                    # Update all devices with the new result for the processed location.
            		self.device.locks[l].acquire()
            		for device in neighbours:
            			device.lock.acquire()
            			device.set_data(l, r)
            			device.lock.release()
            			
            		self.device.lock.acquire()
            		self.device.set_data(l, r)
            		self.device.lock.release()
            		self.device.locks[l].release()

            # --- CLEANUP AND SYNCHRONIZATION PHASE ---
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            # Wait for all other device threads to reach this point before the next time step.
            self.device.barrier.wait()

class ReusableBarrierSem():
    """A reusable barrier implemented with semaphores.

    This allows a group of threads to all synchronize at a point in code. It's
    reusable, meaning it can be used multiple times (e.g., in a loop). It uses
    a two-phase signaling protocol.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        """Causes a thread to block until all threads have called wait."""
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """First phase of the barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread to arrive releases all waiting threads.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        
        self.threads_sem1.acquire()
    
    def phase2(self):
        """Second phase to ensure no thread races ahead to the next `wait` call."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        
        self.threads_sem2.acquire()

class Master(Thread):
    """A worker thread that executes computational scripts."""
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
    	self.Thread_result.append(result)

    def add_script(self, script):
    	self.Thread_script.append(script)

    def add_script_data(self, script_data):
    	self.Thread_script_data.append(script_data)

    def add_location(self, location):
    	self.Thread_location.append(location)

    def add_lock(self, lock):
    	self.Thread_lock.append(lock)
    
    def set_worker(self, script, script_data):
        """Sets up a work item for the thread."""
    	self.add_script(script)
    	self.add_script_data(script_data)

    def set_iterations(self):
    	if self.Thread_script != []:
    		self.Thread_iterations = len(self.Thread_script)
    	else:
    		self.Thread_iterations = 0

    def get_result(self):
    	return self.Thread_result

    def get_location(self):
    	return self.Thread_location

    def get_lock(self):
    	return self.Thread_lock
    
    def run(self):
        """Executes all assigned scripts."""
    	self.set_iterations()
    	for i in xrange(self.Thread_iterations):
    		aux_script = self.Thread_script[i]
    		aux_script_data = self.Thread_script_data[i]
    		aux_rez = aux_script.run(aux_script_data)
    		self.add_result(aux_rez)
