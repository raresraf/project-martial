"""
@d68ae960-8c05-470a-b5ef-0421f39caad4/device.py
@brief Distributed sensor network simulation with centralized I/O and parallel compute.
This module implements a coordinated processing model where a central node manager 
(DeviceThread) handles all neighborhood communication (aggregation and propagation) 
while offloading computational tasks to a pool of worker threads (Master). The system 
utilizes a monotonic update strategy for state convergence and ensures temporal 
consistency across the network through a two-phase semaphore barrier.

Domain: Centralized Orchestration, Parallel Computation, Monotonic Convergence.
"""

from threading import *


class Device(object):
    """
    Representation of a node in the sensor network.
    Functional Utility: Manages local data state and provides the interface for 
    monotonic data updates and synchronization resource distribution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # Primary orchestration thread.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = 0
        # Node-level mutex for protecting local sensor state.
        self.lock = Lock()
        self.locks = []

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global synchronization initialization.
        Logic: Coordinator node (ID 0) initializes a pool of 100 spatial locks 
        and a network-wide barrier, distributing them to all peer nodes.
        """
        if(self.device_id == 0):
        	# Atomic Resource Allocation: pre-populates the spatial lock pool.
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
        """Registers a task and signals the orchestration thread."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Safe retrieval of local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Monotonic State Update.
        Logic: Updates the local sensor value only if the new data is greater 
        than the existing value, ensuring forward-only convergence.
        """
        if location in self.sensor_data:
        	if self.sensor_data[location] < data:
        		self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully joins the orchestration thread."""
        self.thread.join()

    def get_dev_lock(self):
    	return self.lock

    def add_lock(self, lock):
    	"""Appends a new spatial lock to the local repository."""
    	self.locks.append(lock)

    def get_locks(self):
    	return self.locks

class DeviceThread(Thread):
    """
    Main orchestration thread for the node.
    Functional Utility: Implements the 'Centralized Orchestrator' pattern, 
    managing all cross-node communication phases.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop.
        Algorithm: Iterative sequence: 
        Wait -> Sequential Aggregate -> Round-Robin Dispatch -> Join -> Sequential Propagate.
        """
        while True:
            # Topology Discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Initialize the computational worker pool for the current step.
            threads = []
            index = 0
            for i in xrange(8):
            	aux_thread = Master(i)
            	threads.append(aux_thread)

            # Block until work assignment phase is complete.
            self.device.timepoint_done.wait()
            
            # Phase 1: Sequential Data Aggregation.
            # Orchestrator gathers all necessary inputs from neighbors under spatial locks.
            for (script, location) in self.device.scripts:
            	script_data = []
            	self.device.locks[location].acquire()
            	
            	# Aggregate neighbor state.
            	for device in neighbours:
            		if device.device_id != self.device.device_id:
            			device.lock.acquire()
            			data = device.get_data(location)
            			device.lock.release()
            			if data is not None:
            				script_data.append(data)

            	# Include local state.
            	self.device.lock.acquire()
            	data = self.device.get_data(location)
            	self.device.lock.release()
            	if data is not None:
            		script_data.append(data)

            	if script_data != []:
            		# Phase 2: Task Dispatch.
            		# logic: Distributes computational work to the parallel pool.
            		threads[index].set_worker(script, script_data)
            		threads[index].add_location(location)
            		aux_lock = self.device.locks[location]
            		threads[index].add_lock(aux_lock)
            		
            		# Round-robin assignment.
            		index = index + 1
            		if index == 8:
            			index = 0
            	self.device.locks[location].release()

            # Phase 3: Parallel Execution.
            for i in xrange(8):
            	threads[i].start()
            for i in xrange(8):
            	threads[i].join()
            
            # Phase 4: Sequential Data Propagation.
            # Orchestrator distributes pre-calculated results back to the network.
            for i in xrange(8):
            	result_list = threads[i].get_result()
            	location_list = threads[i].get_location()
            	dim = len(result_list)
            	for j in xrange(dim) :
            		r = result_list[j]
            		l = location_list[j]
            		
            		# Atomic propagation per spatial location.
            		self.device.locks[l].acquire()
            		for device in neighbours:
            			device.lock.acquire()
            			device.set_data(l, r)
            			device.lock.release()

            		self.device.lock.acquire()
            		self.device.set_data(l, r)
            		self.device.lock.release()
            		self.device.locks[l].release()

            # Cleanup and Synchronization.
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

class ReusableBarrierSem():
    """
    Two-phase reusable barrier implementation using semaphores.
    Functional Utility: Provides temporal rendezvous points for simulation cycles.
    """
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        """Orchestrates the double-gate synchronization protocol."""
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """Arrival phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()
    
    def phase2(self):
        """Exit phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Master(Thread):
    """
    Computational worker thread.
    Functional Utility: Executes a batch of computational scripts using pre-aggregated 
    inputs provided by the orchestrator.
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
    	"""Worker execution loop: processes the batch of assigned computational tasks."""
    	self.set_iterations()
    	for i in xrange(self.Thread_iterations):
    		aux_script = self.Thread_script[i]
    		aux_script_data = self.Thread_script_data[i]
    		aux_rez = aux_script.run(aux_script_data)
    		self.add_result(aux_rez)
