"""
@file device.py
@brief This file defines a simulated device for a distributed system that processes scripts sequentially due to a global lock.
@details The script models a network of devices that execute computational scripts based on sensor data.
         Each device runs a single thread. A custom condition-based barrier is used for synchronization
         between time steps. A significant design feature is the use of a single, global lock around
         script execution, which effectively serializes the operations of all devices in the network,
         preventing any parallel execution between them.
"""


from threading import Event, Thread, Condition, Lock


class Device(object):
    """
    @brief Represents a single device in a simulated network.
    @details Each device has its own thread for processing and uses shared synchronization primitives
             (a barrier and a global lock) to coordinate with other devices.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary holding the local sensor data.
        @param supervisor An object for querying global state, like neighbors.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barr = None
        self.lock = None
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs collective setup for all devices.
        @details The first device in the list is responsible for creating a shared barrier.
                 This method also attempts to create and distribute a global lock, but due to an
                 indentation error, it likely does not function as intended, resulting in
                 improperly shared state.
        """
        
        
        
		
        if devices[0].barr is None and self.device_id == devices[0].device_id:
                barr = CondBarrier(len(devices))
                for i in devices:
                        i.barr = barr
        # Bug: This lock is created by every device that calls this method. Because it is not
        # indented within the `if` block above, the lock is not a true singleton. The last
        # device to execute this line will overwrite the lock for all other devices.
        lock = Lock()	
        for d in devices:
                d.lock = lock 
		
		

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for processing.
        @param script The script to be executed.
        @param location The location context for the script.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A `None` script signals that all scripts for the timepoint have been assigned.
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @return The sensor data or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data at a given location.
        @param location The location to update.
        @param data The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
	
    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its thread to complete.
        """
        self.thread.join()

class CondBarrier():
	"""
    @brief A reusable barrier implementation using a `threading.Condition`.
    @details This barrier allows a specified number of threads to wait until all of them
             have reached the barrier point. It is reusable after all threads have passed.
    """
	def __init__(self, num_threads):
		self.num_threads = num_threads
		self.count_threads = self.num_threads
		self.cond = Condition()

	def wait(self):
		"""
        @brief Causes a thread to wait at the barrier.
        @details When the last thread arrives, it notifies all waiting threads and resets the barrier
                 for the next use.
        """
		self.cond.acquire()
		self.count_threads -= 1
		if self.count_threads == 0:
			self.cond.notify_all()
			self.count_threads = self.num_threads
		else:
			self.cond.wait()
		self.cond.release()



class DeviceThread(Thread):
    """
    @brief The main worker thread for a device.
    @details This thread is responsible for executing all assigned scripts for a given time step.
             It uses a global lock, which serializes its execution with all other devices.
    """
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
		

    def run(self):
        """
        @brief The main execution loop for the device thread.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A `None` value for neighbors signals termination.
                break



            # Invariant: Wait until all scripts for the current time step have been assigned.
            self.device.timepoint_done.wait()

            
            
            # Block Logic: Process all assigned scripts for this time step.
            for (script, location) in self.device.scripts:
                # Inline: A single global lock is acquired here. This is a major performance bottleneck,
                # as it prevents any two devices from executing scripts concurrently. The entire
                # computation phase across all devices becomes sequential.
                self.device.lock.acquire()
                script_data = []
                
                # Gathers data from neighbors for the script's execution context.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    


                    # Broadcast the result to all neighboring devices and the local device.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                self.device.lock.release()
            
            # Reset the event for the next time step.
            self.device.timepoint_done.clear()
            # Invariant: Synchronize with all other devices before starting the next time step.
            self.device.barr.wait()