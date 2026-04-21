
"""
@file device.py
@brief Distributed device simulation environment using a master-worker orchestration pattern.

Functional Intent: Implements a multi-threaded framework where devices coordinate 
parallel execution of data-processing scripts. Features a custom reusable barrier 
for synchronizing discrete simulation steps across all participating entities 
and manages concurrent access to shared data locations via fine-grained locking.

Domain: Production Systems, Distributed Computing, Parallel Processing.
"""

from threading import *


class Device(object):
    """
    @brief Represents a computational node in the simulated distributed network.
    
    Functional Utility: Maintains local sensor state and manages the lifecycle 
     of assigned tasks. Collaborates with peers to perform collective computations 
     while ensuring global synchronization through a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device and starts its management thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # Synchronization: Events for coordinating script assignment and execution flow.
        self.script_received = Event()
        self.timepoint_done = Event()
        
        # Logic: Dedicated thread for high-level device orchestration.
        self.thread = DeviceThread(self)
        self.thread.start()
        
        self.barrier = 0 # Logic: Shared synchronization barrier (to be set during setup).
        self.lock = Lock() # Logic: Mutex for protecting device-local sensor data.
        self.locks = [] # Logic: Shared lock pool for location-specific data access.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Orchestrates the initialization of shared synchronization primitives.
        
        Algorithm: Single-node master initialization (Device 0).
        Logic: Generates a pool of shared locks and a global barrier, then distributes 
        references to all peers to ensure uniform synchronization behavior.
        """
        # Pre-condition: Exactly one device initializes the shared state to prevent fragmentation.
        if(self.device_id == 0):
            # Block Logic: Lock pool generation.
            # Invariant: Creates 100 mutexes to protect distinct data locations in the global namespace.
            for i in xrange(100):
                aux_lock = Lock()
                self.add_lock(aux_lock)
                for j in devices:
                    j.add_lock(aux_lock)
            
            # Synchronization: Global barrier to coordinate discrete time-steps.
            nr = len(devices)
            barrier = ReusableBarrierSem(nr)
            for i in devices:
                i.barrier = barrier
        

    def assign_script(self, script, location):
        """
        @brief Binds a processing script to a specific data location.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Logic: Signaling None terminates the assignment phase for the current epoch.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a location. (Requires device lock).
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates sensor data with monotonic growth constraint.
        """
        if location in self.sensor_data:
            # Logic: Only permits updates that increase the stored value (e.g. tracking maxima).
            if self.sensor_data[location] < data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully terminates the orchestration thread.
        """
        self.thread.join()

    def get_dev_lock(self):
    	return self.lock

    def add_lock(self, lock):
    	self.locks.append(lock)

    def get_locks(self):
    	return self.locks


class DeviceThread(Thread):
    """
    @brief Orchestration thread that manages task distribution to worker (Master) threads.
    
    Algorithm: Neighborhood data aggregation followed by parallel execution.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Core device operational loop.
        
        Logic: 
        1. Identifies neighbors via supervisor.
        2. Aggregates data from the neighborhood for each assigned script.
        3. Dispatches work to a pool of dedicated worker threads.
        4. Broadcasts results back to the neighborhood after consensus.
        """
        while True:
            # Block Logic: Neighbor discovery.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Optimization: Pre-allocated pool of 8 worker threads for local concurrency.
            threads = []
            index = 0
            for i in xrange(8):
            	aux_thread = Master(i)
            	threads.append(aux_thread)

            # Synchronization: Wait for all scripts to be assigned for this timepoint.
            self.device.timepoint_done.wait()
            
            # Block Logic: Task preparation pass.
            for (script, location) in self.device.scripts:
            	script_data = []
            	# Synchronization: Lock acquisition for the specific data location.
            	self.device.locks[location].acquire()
            	
            	# Logic: Gathers state from neighbors while respecting individual device locks.
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
            		# Logic: Distributes tasks to worker threads using simple round-robin scheduling.
            		threads[index].set_worker(script, script_data)
            		threads[index].add_location(location)
            		aux_lock = self.device.locks[location]
            		threads[index].add_lock(aux_lock)
            		index = (index + 1) % 8
            		
            	self.device.locks[location].release()

            # Parallel Execution: Trigger and wait for all local workers.
            for i in xrange(8):
            	threads[i].start()
            for i in xrange(8):
            	threads[i].join()
            
            # Block Logic: Result dissemination pass.
            for i in xrange(8):
            	result_list = threads[i].get_result()
            	location_list = threads[i].get_location()
            	dim = len(result_list)
            	for j in xrange(dim) :
            		r = result_list[j]
            		l = location_list[j]
            		
            		# Synchronization: Atomic update of neighborhood state with the computed result.
            		self.device.locks[l].acquire()
            		for device in neighbours:
            			device.lock.acquire()
            			device.set_data(l, r)
            			device.lock.release()
            			
            		self.device.lock.acquire()
            		self.device.set_data(l, r)
            		self.device.lock.release()
            		self.device.locks[l].release()

            # Finalization: Resets events and synchronizes at the global barrier.
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

class ReusableBarrierSem():
    """
    @brief Custom reusable synchronization barrier implemented using semaphores.
    
    Algorithm: Two-phase (turnstile) synchronization.
    Logic: Uses two semaphores to ensure all threads arrive before any are released, 
    and all threads leave before the barrier can be reused for the next epoch.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes barrier counters and semaphores.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """
        @brief Blocks calling thread until the collective rendezvous is met.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Entrance phase: Blocks threads until the full group arrives.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Logic: The final thread releases the entire group.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Exit phase: Ensures all threads have cleared phase1 before resetting.
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
    @brief Worker thread (Master) responsible for executing processing scripts.
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
    	"""
    	@brief Sequential execution of assigned scripts.
    	"""
    	self.set_iterations()
    	for i in xrange(self.Thread_iterations):
    		aux_script = self.Thread_script[i]
    		aux_script_data = self.Thread_script_data[i]
    		# Functional Intent: Executes transformation logic on gathered neighborhood data.
    		aux_rez = aux_script.run(aux_script_data)
    		self.add_result(aux_rez)
