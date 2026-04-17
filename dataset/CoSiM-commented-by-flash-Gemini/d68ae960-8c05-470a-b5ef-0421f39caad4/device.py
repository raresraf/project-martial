"""
@d68ae960-8c05-470a-b5ef-0421f39caad4/device.py
@brief Distributed sensor processing simulation using a two-stage aggregation-update model and persistent thread pool.
* Algorithm: Batch script execution via 8 `Master` worker threads with decoupled aggregation (pre-execution) and propagation (post-execution) phases.
* Functional Utility: Orchestrates simulation timepoints across multiple devices by managing distributed data acquisition and synchronized state updates using location and device-level locks.
"""

from threading import *

class Device(object):
    """
    @brief Encapsulates a sensor node with its local readings, coordination state, and lock pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device state and bootstraps the main coordination thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = 0
        self.lock = Lock() # Intent: Device-level lock for protecting local readings.
        self.locks = []    # Intent: Global pool of locks for location-specific synchronization.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Global synchronization and lock distribution.
        Invariant: Root node (ID 0) initializes a pool of 100 locks and broadcasts them along with the barrier.
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
        @brief Receives a task for the current simulation phase.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Logic: Signals completion of script arrival for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Standard data retrieval for sensor locations.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates local sensor data if the new value represents a state transition (greater than current).
        """
        if location in self.sensor_data:
        	if self.sensor_data[location] < data:
        		self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Terminates the device coordination thread.
        """
        self.thread.join()

    def get_dev_lock(self):
    	return self.lock

    def add_lock(self, lock):
    	"""
    	@brief Appends a synchronization lock to the device's shared pool.
    	"""
    	self.locks.append(lock)

    def get_locks(self):
    	return self.locks

class DeviceThread(Thread):
    """
    @brief Main coordinator thread managing simulation phases and worker orchestration.
    """

    def __init__(self, device):
        """
        @brief Initializes the coordinator for a specific device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core lifecycle loop of the device node.
        Algorithm: Phased execution consisting of Data Aggregation -> Parallel Worker Dispatch -> Result Propagation.
        """
        while True:
            # Logic: Refresh neighbor set from supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Setup Phase: Spawns 8 persistent Master threads for this timepoint.
            threads = []
            index = 0
            for i in xrange(8):
            	aux_thread = Master(i)
            	threads.append(aux_thread)

            # Block Logic: Ensures all tasks for the current cycle have arrived.
            self.device.timepoint_done.wait()
            
            # Phase 1: Aggregation.
            # Logic: Collects required data for each script under location-specific locks.
            for (script, location) in self.device.scripts:
            	script_data = []
            	self.device.locks[location].acquire()
            	
            	# Distributed Aggregation: Acquire readings from neighbors using device-level mutual exclusion.
            	for device in neighbours:
            		if device.device_id != self.device.device_id:
            			device.lock.acquire()
            			data = device.get_data(location)
            			device.lock.release()
            			if data is not None:
            				script_data.append(data)

            	self.device.lock.acquire()
            	data = self.device.get_data(location)
            	self.device.lock.release()
            	if data is not None:
            		script_data.append(data)

            	if script_data != []:
            		# Task Assignment: Distributes aggregated data and script to worker pool.
            		threads[index].set_worker(script, script_data)
            		threads[index].add_location(location)
            		aux_lock = self.device.locks[location]
            		threads[index].add_lock(aux_lock)
            		index = (index + 1) % 8
            	self.device.locks[location].release()

            # Phase 2: Parallel Execution.
            for i in xrange(8):
            	threads[i].start()
            for i in xrange(8):
            	threads[i].join()
            
            # Phase 3: Result Propagation.
            # Logic: Commits computed results back to the device cluster under appropriate locks.
            for i in xrange(8):
            	result_list = threads[i].get_result()
            	location_list = threads[i].get_location()
            	dim = len(result_list)
            	for j in xrange(dim) :
            		r = result_list[j]
            		l = location_list[j]
            		self.device.locks[l].acquire()
            		for device in neighbours:
            			device.lock.acquire()
            			device.set_data(l, r)
            			device.lock.release()
            			
            		self.device.lock.acquire()
            		self.device.set_data(l, r)
            		self.device.lock.release()
            		self.device.locks[l].release()

            # Synchronization Phase: Align all devices across the cluster.
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

class ReusableBarrierSem():
    """
    @brief Implementation of a two-phase synchronization barrier using semaphores.
    * Algorithm: Dual-stage arrival pattern to ensure strict thread alignment.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        """
        @brief Blocks calling thread through both stages of the barrier.
        """
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
    @brief Worker thread implementing the computational component of the simulation.
    """
    def __init__(self, id):
        """
        @brief Initializes the worker with private result and task buffers.
        """
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
    	"""
    	@brief Main execution logic for the worker.
    	Algorithm: Executes all assigned scripts sequentially using pre-aggregated data.
    	"""
    	self.set_iterations()
    	for i in xrange(self.Thread_iterations):
    		aux_script = self.Thread_script[i]
    		aux_script_data = self.Thread_script_data[i]
    		# Logic: Pure compute phase - no synchronization primitives acquired here.
    		aux_rez = aux_script.run(aux_script_data)
    		self.add_result(aux_rez)
