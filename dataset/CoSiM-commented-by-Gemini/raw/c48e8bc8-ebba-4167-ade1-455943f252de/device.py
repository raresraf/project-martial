"""
This module implements a simulation of a distributed device network.

It uses a pool of 'Master' worker threads to process tasks and features a
complex locking strategy with a pre-allocated list of locks for data locations
and per-device locks. The synchronization between devices at the end of a
time-step is handled by a two-phase reusable semaphore barrier.

NOTE: This code appears to be written for Python 2 (e.g., it uses `xrange`).
The concurrency model has significant bottlenecks, as data gathering and updating
are performed sequentially in the main device thread, not in parallel by the
worker threads.
"""


from threading import *


class Device(object):
    """
    Represents a single device in the simulation.

    Manages device state, data, and shared synchronization objects like locks
    and barriers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = 0
        self.lock = Lock() # A per-device lock.
        self.locks = []    # A shared list of location-specific locks.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (locks and barrier).

        This is orchestrated by device 0, which creates a list of 100 locks and
        a barrier and shares the instances with all other devices.
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
        """Assigns a script to the device or signals the start of processing."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data, but only if the new data value is greater than the old one.
        """
        if location in self.sensor_data:
        	if self.sensor_data[location] < data:
        		self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device thread."""
        self.thread.join()

    def get_dev_lock(self):
    	return self.lock

    def add_lock(self, lock):
    	self.locks.append(lock)

    def get_locks(self):
    	return self.locks

class DeviceThread(Thread):
    """
    Main control thread for a device, orchestrating the multi-stage
    process of data gathering, computation, and data update.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main operational loop, which contains a significant concurrency bottleneck.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal

            threads = []
            index = 0

            # Pre-create a pool of worker threads.
            for i in xrange(8):
            	aux_thread = Master(i)
            	threads.append(aux_thread)

            # Wait for the signal to start processing for the time-step.
            self.device.timepoint_done.wait()

            # --- DATA GATHERING AND WORK DISTRIBUTION (SEQUENTIAL BOTTLENECK) ---
            # This entire loop runs sequentially in the DeviceThread, not in parallel.
            for (script, location) in self.device.scripts:
            	script_data = []
            	# Acquire a lock for the specific data location.
            	self.device.locks[location].acquire()
            	
            	# Gather data from neighbors, using a per-device lock for each.
            	for device in neighbours:
            		if device.device_id != self.device.device_id:
            			device.lock.acquire()
            			data = device.get_data(location)
            			device.lock.release()
            			if data is not None:
            				script_data.append(data)
            	
            	# Gather data from self.
            	self.device.lock.acquire()
            	data = self.device.get_data(location)
            	self.device.lock.release()
            	if data is not None:
            		script_data.append(data)

            	if script_data != []:
            		# Assign the script and its gathered data to a worker thread.
            		threads[index].set_worker(script, script_data)
            		threads[index].add_location(location)
            		aux_lock = self.device.locks[location]
            		threads[index].add_lock(aux_lock)
            		index = index + 1
            		if index == 8:
            			index = 0
            	
            	self.device.locks[location].release()

            # --- EXECUTION ---
            # Start the worker threads to perform the computation (script.run).
            for i in xrange(8):
            	threads[i].start()
            for i in xrange(8):
            	threads[i].join()
            
            # --- DATA UPDATE (SEQUENTIAL BOTTLENECK) ---
            # This loop also runs sequentially, re-acquiring all locks.
            for i in xrange(8):
            	result_list = threads[i].get_result()
            	location_list = threads[i].get_location()
            	dim = len(result_list)
            	for j in xrange(dim) :
            		r = result_list[j]
            		l = location_list[j]
            		self.device.locks[l].acquire()
            		# Update data on all neighbors and self.
            		for device in neighbours:
            			device.lock.acquire()
            			device.set_data(l, r)
            			device.lock.release()
            			
            		self.device.lock.acquire()
            		self.device.set_data(l, r)
            		self.device.lock.release()
            		self.device.locks[l].release()

            # --- SYNCHRONIZATION ---
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            # Wait for all devices to finish the time-step.
            self.device.barrier.wait()

class ReusableBarrierSem():
    """A reusable, two-phase thread barrier implemented with Semaphores."""
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        """Blocks threads in two phases to ensure safe reuse."""
        self.phase1()
        self.phase2()
    
    def phase1(self):
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()
    
    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Master(Thread):
    """
    A batch worker thread that executes a list of scripts.
    
    Note: This thread only performs the `script.run()` computation. Data
    gathering and updates are handled by the parent DeviceThread.
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
        """Runs all assigned scripts and stores their results."""
    	self.set_iterations()
    	for i in xrange(self.Thread_iterations):
    		aux_script = self.Thread_script[i]
    		aux_script_data = self.Thread_script_data[i]
    		aux_rez = aux_script.run(aux_script_data)
    		self.add_result(aux_rez)