"""
This module implements a simulation of a networked device that processes sensor data.
It features a custom condition-based barrier for synchronization between devices and
executes data processing scripts within each device's own thread.
"""

from threading import Event, Thread, Condition, Lock


class Device(object):
    """
    Represents a single device in the simulated network. Each device manages its
    sensor data, executes assigned scripts, and synchronizes with other devices
    using a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance and starts its execution thread.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's sensor data.
            supervisor (Supervisor): The supervisor object managing the network.
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
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects (a barrier and a lock)
        to all devices in the network. This method should be called by one designated device.
        """
        
        
        
		
        if devices[0].barr is None and self.device_id == devices[0].device_id:
                barr = CondBarrier(len(devices))
                for i in devices:
                        i.barr = barr
        lock = Lock()	
        for d in devices:
                d.lock = lock 
		
		

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device. If the script is None,
        it signals that the current timepoint's script assignment is complete.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data
	
    def shutdown(self):
        """Shuts down the device by waiting for its thread to complete."""
        self.thread.join()

class CondBarrier():
	"""
    A custom barrier implementation using a `threading.Condition` object.
    It blocks threads until a specified number of threads have called the wait() method.
    """
	def __init__(self, num_threads):
		"""
        Initializes the barrier for a given number of threads.
        """
		self.num_threads = num_threads
		self.count_threads = self.num_threads
		self.cond = Condition()

	def wait(self):
		"""
        Causes a thread to wait at the barrier. When the required number of threads
        are waiting, all are released.
        """
		self.cond.acquire()
		self.count_threads -= 1
		if self.count_threads == 0:
			# All threads have arrived; notify all waiting threads.
			self.cond.notify_all()
			self.count_threads = self.num_threads
		else:
			# Wait for the remaining threads to arrive.
			self.cond.wait()
		self.cond.release()



class DeviceThread(Thread):
    """
    The main execution thread for a Device. It handles the device's lifecycle,
    including script execution and synchronization.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
		

    def run(self):
        """
        The main loop of the device thread. It continuously waits for scripts,
        executes them, and synchronizes with other devices at each timepoint.
        """
        while True:
            
            # Fetches the list of neighbors for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break


            # Waits until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            
            # Block Logic: Executes all assigned scripts for the current timepoint.
            for (script, location) in self.device.scripts:
                self.device.lock.acquire()
                script_data = []
                
                # Gathers data from all neighboring devices for the script execution.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Adds its own data to the dataset for the script.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    


                    # Disseminates the result of the script to all neighbors and itself.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                self.device.lock.release()
            
            self.device.timepoint_done.clear()
            # Synchronization point: All devices wait here until every device has finished its computation for the current timepoint.
            self.device.barr.wait()
