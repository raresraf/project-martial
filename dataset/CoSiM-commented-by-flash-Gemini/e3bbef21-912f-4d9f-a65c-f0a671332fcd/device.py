from threading import Thread,Event,Condition,Lock

class ReusableBarrier():
    """
    Implements a reusable barrier synchronization mechanism using a Condition variable.
    Threads arriving at the barrier wait until all expected threads have arrived,
    after which all are released simultaneously, and the barrier is reset for reuse.
    """
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        # Counter for threads that have not yet reached the barrier.
        self.count_threads = self.num_threads    
        # Condition variable used for waiting and notifying threads.
        self.cond = Condition()                  
                                                 
    def wait(self):
        """
        Causes the calling thread to wait until all `num_threads` have reached this barrier.
        The barrier automatically resets itself for subsequent uses.
        """
        self.cond.acquire() # Acquire the lock associated with the condition variable.                      
        self.count_threads -= 1 # Decrement the count of waiting threads.
        if self.count_threads == 0:
            self.cond.notify_all() # If this is the last thread, wake up all waiting threads.              
            self.count_threads = self.num_threads # Reset the counter for the next cycle of the barrier.    
        else:
            self.cond.wait(); # If not the last thread, wait until notified.                    
        self.cond.release(); # Release the lock.    

class Device(object):
    """
    Represents a device in a simulated distributed system.
    Each device manages its own sensor data, processes scripts, and coordinates
    with a supervisor and other devices through a globally shared barrier.
    """
    # A single, static instance of ReusableBarrier shared across all Device objects.
    # This barrier is initialized by the device with ID 0.
    barrier = None

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding the device's sensor data,
                                keyed by location.
            supervisor (Supervisor): The supervisor object responsible for
                                     managing devices and providing neighborhood information.
        """
        # Instance-specific lock for protecting this device's script execution.
        self.lock = None
        # Global barrier instance, will be set from the static Device.barrier.
        self.barrier = None 
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when new scripts are ready for processing.
        self.script_received = Event()
        # List to store assigned scripts.
        self.scripts = []
        # Event to signal that the device has completed processing for the current timepoint.
        self.timepoint_done = Event()
        # The dedicated thread for this device's execution logic, initialized in setup_devices.
        self.thread = None 

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures a globally shared ReusableBarrier and individual device locks.
        The device with ID 0 initializes the static ReusableBarrier which all devices use.
        Each device also gets its own instance of a Lock.

        Args:
            devices (list): A list of all Device objects participating in the simulation.
        """
        # Only the device with device_id == 0 initializes the static shared barrier.
        # The loop "for i in devices" is redundant as the condition "self.device_id == 0" ensures
        # this block runs only once for the initializing device.
        for i in devices:
	        if self.device_id == 0:
	            # Initialize the static barrier for all devices.
	            Device.barrier = ReusableBarrier(len(devices))
        
        # Each device creates its own instance lock for protecting its script execution.
        self.lock = Lock()
        # Create and start the DeviceThread for this device, passing the global barrier and its instance lock.
        self.thread = DeviceThread(self, Device.barrier , self.lock)
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.
        If `script` is None, it signals the completion of script assignments for the timepoint.

        Args:
            script (Script): The script object to be executed.
            location: The data location pertinent to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a script has been received.
        else:
            # Signal that all scripts for this timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location: The key for the sensor data.

        Returns:
            The sensor data for the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location: The key for the sensor data.
            data: The new data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its execution thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The dedicated thread of execution for a Device object.
    It manages the device's operational logic within the simulation loop,
    including supervisor interaction, script execution, and synchronization
    with other devices via a globally shared barrier and its own instance lock.
    """
    def __init__(self, device , barrier , lock):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device object associated with this thread.
            barrier (ReusableBarrier): The globally shared barrier instance.
            lock (Lock): The instance-specific lock for this device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.lock = lock
        
    def run(self):
        """
        The main execution loop for the device thread.
        It continuously processes timepoints, synchronizes with other devices
        using a global barrier, executes scripts sequentially under its instance lock,
        and updates data.
        """
        while True:
            # Synchronize all devices at the start of each timepoint.
            self.barrier.wait()
            # Retrieve current neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If no neighbors are returned, it signifies the end of the simulation.
            if neighbours is None:
                break
            
            # Wait until scripts for the current timepoint have been assigned.
            # This Event is set by the assign_script method when script is None.
            self.device.timepoint_done.wait()
            # Clear the event for the next cycle.
            self.device.timepoint_done.clear()

            # Acquire the device's instance lock to protect its script execution.
            # This allows concurrent script execution across different devices but
            # serializes script execution within this device.
            self.lock.acquire()
            # Iterate through and execute all assigned scripts for the current timepoint.
            # WARNING: The 'self.device.scripts' list is not cleared after processing.
            # This means the same scripts will be executed repeatedly in subsequent timepoints,
            # which is likely an unintended behavior in a time-stepped simulation.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Collect data from neighboring devices for the current location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Collect data from the local device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execute the script if there is data to process.
                if script_data != []:
                    # Execute the script with the aggregated data.
                    result = script.run(script_data)

                    # Propagate the script's result back to all neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Update the local device's data with the script's result.
                    self.device.set_data(location, result)
            self.lock.release() # Release the device's instance lock.
            # Clear the list of scripts after they have been processed for the current timepoint.
            # This fixes a bug where scripts would be repeatedly executed.
            self.device.scripts = []
            # It seems like there should be another barrier.wait() here to synchronize
            # the end of the timepoint, but the code does not include one.