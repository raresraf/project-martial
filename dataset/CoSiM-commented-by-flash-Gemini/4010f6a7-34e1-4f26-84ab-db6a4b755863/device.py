


"""
@file device.py
@brief This module defines a simulated device environment utilizing a master-worker thread pattern for concurrent script execution and a custom reusable barrier for synchronization.

@details It implements the `Device` class to represent individual simulation entities.
         The `DeviceThread` manages the device's operational loop, dispatching
         scripts to a fixed pool of `Master` threads. These `Master` threads
         execute the scripts and store results. Shared `Lock` objects ensure
         data consistency across different data locations. A custom `ReusableBarrierSem`
         implementation is used for global synchronization among devices.
"""

from threading import *


class Device(object):
    """
    @brief Represents a single simulated device in the environment.

    @details Each device manages its own sensor data, interacts with a central supervisor,
             and executes assigned scripts. It coordinates its activities through a dedicated
             `DeviceThread`, which dispatches scripts to a pool of `Master` threads.
             A shared `ReusableBarrierSem` ensures global synchronization, and a shared
             list of `Lock` objects (`self.locks`) handles data consistency for locations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id An integer representing the unique identifier for the device.
        @param sensor_data A dictionary containing initial sensor data for the device.
        @param supervisor An object responsible for overseeing and coordinating devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal that all scripts for a timepoint are assigned.
        self.scripts = []               # List to store (script, location) tuples assigned to this device.
        self.timepoint_done = Event()   # Event to signal overall timepoint processing completion.
        self.thread = DeviceThread(self) # The dedicated thread for this device's main loop.
        self.thread.start()             # Start the main device thread.
        self.barrier = 0                # Placeholder for the shared ReusableBarrierSem.
        self.lock = Lock()              # A device-specific lock, potentially for internal state protection.
        self.locks = []                 # Shared list of locks for data locations.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization resources (barrier and location locks).

        @details This method is called once at the beginning of the simulation.
                 Only Device 0 initializes the global `ReusableBarrierSem` and
                 a shared list of `Lock` objects for a fixed number of data locations
                 (up to 100). These shared resources are then distributed to all devices.
        @param devices A list of all Device objects in the simulation.
        """
        
        # Precondition: Only device with ID 0 is responsible for initializing shared resources.
        if(self.device_id == 0):
            # Block Logic: Create and distribute 100 location-specific locks.
            for i in xrange(100): # Hardcoded number of locks.
                aux_lock = Lock()
                self.add_lock(aux_lock) # Add to this device's shared locks list.
                for j in devices:
                    j.add_lock(aux_lock) # Add to all other devices' shared locks lists.
            
            # Block Logic: Initialize and distribute the global ReusableBarrierSem.
            nr = len(devices)
            barrier = ReusableBarrierSem(nr)
            for i in devices:
                i.barrier = barrier # Set the shared barrier for all devices.
        

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific data location.

        @details If a script is provided, it's added to the device's script queue
                 and signals `script_received`. If `script` is None, it signals
                 `timepoint_done`, indicating that all scripts for the current
                 timepoint have been assigned and the device is ready for processing.
        @param script The script object to assign, or None.
        @param location The data location (e.g., sensor ID) where the script will operate.
        """
        
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its target location.
            self.script_received.set() # Signal that a script has been received.
        else:
            self.timepoint_done.set() # Signal completion for the timepoint (no more scripts to assign).

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location from the device's internal state.

        @param location The data location (e.g., sensor ID).
        @return The sensor data for the location, or None if the location is not found.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location in the device's internal state.

        @details This method updates the `sensor_data` only if the new `data` is
                 greater than the existing value for that `location`.
        @param location The data location (e.g., sensor ID).
        @param data The new sensor data to set.
        """
        
        if location in self.sensor_data:
            if self.sensor_data[location] < data: # Only update if new data is greater.
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's dedicated thread.

        @details Joins the `DeviceThread`, ensuring all ongoing operations are completed
                 before the device fully shuts down.
        """
        
        self.thread.join() # Wait for the main device thread to finish its execution.

    def get_dev_lock(self):
        """
        @brief Returns the device's internal lock.

        @return The `threading.Lock` instance specific to this device.
        """
    	return self.lock

    def add_lock(self, lock):
        """
        @brief Adds a shared location lock to this device's list of locks.

        @param lock A `threading.Lock` object for a specific data location.
        """
    	self.locks.append(lock)

    def get_locks(self):
        """
        @brief Returns the list of shared location locks.

        @return A list of `threading.Lock` objects.
        """
    	return self.locks


class DeviceThread(Thread):
    """
    @brief Manages the main operational loop for a Device object, coordinating script execution.

    @details This thread is responsible for the overall lifecycle of a device's operations
             within a timepoint. It fetches neighbor information from the supervisor,
             dispatches scripts to a pool of `Master` threads for concurrent processing,
             and handles global synchronization using the shared `ReusableBarrierSem`.
    """
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            threads = []
            index = 0

            for i in xrange(8):
            	aux_thread = Master(i)
            	threads.append(aux_thread)

            self.device.timepoint_done.wait()
            for (script, location) in self.device.scripts:
            	script_data = []
            	self.device.locks[location].acquire()
            	
            	
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
            		
            		threads[index].set_worker(script, script_data)
            		threads[index].add_location(location)
            		aux_lock = self.device.locks[location]
            		threads[index].add_lock(aux_lock)
            		index = index + 1
            		if index == 8:
            			index = 0
            	self.device.locks[location].release()

            for i in xrange(8):
            	threads[i].start()
            for i in xrange(8):
            	threads[i].join()
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

            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier for synchronizing multiple threads using `threading.Semaphore`.

    @details This custom barrier uses a two-phase mechanism to ensure all participating
             threads wait until every thread has reached a common synchronization point,
             and then releases them simultaneously. It is designed to be reusable
             for subsequent synchronization points without needing re-initialization.
    """
    

    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrierSem instance.

        @param num_threads The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # Two counters are used to manage the two phases of the barrier, allowing reusability.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               # A lock to protect access to the thread counters.
        # Two semaphores, one for each phase, to block and release threads.
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """
        @brief Blocks the calling thread until all threads have reached this barrier.

        @details This method orchestrates the two-phase synchronization, ensuring
                 all `num_threads` complete `phase1` and `phase2` before any
                 thread proceeds past the `wait` call.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Executes the first phase of the barrier synchronization.

        @details Threads decrement a shared counter. The last thread to reach zero
                 releases all other threads waiting on `threads_sem1` and resets
                 the counter for the next cycle. All threads then acquire `threads_sem1`.
        """
        with self.counter_lock: # Protects the counter from race conditions.
            self.count_threads1 -= 1
            if self.count_threads1 == 0: # If this is the last thread in phase 1.
                for i in range(self.num_threads): # Release all waiting threads.
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset counter for reusability.

        self.threads_sem1.acquire() # Block until released by the last thread.

    def phase2(self):
        """
        @brief Executes the second phase of the barrier synchronization.

        @details Similar to `phase1`, threads decrement `count_threads2`. The last
                 thread to reach zero releases others waiting on `threads_sem2`
                 and resets the counter. All threads then acquire `threads_sem2`.
        """
        with self.counter_lock: # Protects the counter from race conditions.
            self.count_threads2 -= 1
            if self.count_threads2 == 0: # If this is the last thread in phase 2.
                for i in range(self.num_threads): # Release all waiting threads.
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset counter for reusability.

        self.threads_sem2.acquire() # Block until released by the last thread.


class Master(Thread):
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
    	self.set_iterations()
    	for i in xrange(self.Thread_iterations):
    		aux_script = self.Thread_script[i]
    		aux_script_data = self.Thread_script_data[i]
    		aux_rez = aux_script.run(aux_script_data)
    		self.add_result(aux_rez)
