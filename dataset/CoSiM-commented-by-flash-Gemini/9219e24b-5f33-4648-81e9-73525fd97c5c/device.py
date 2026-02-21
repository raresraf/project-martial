


from threading import Event, Thread, Condition, Lock


class Device(object):
    """
    Represents a single device in a simulated distributed system.
    Each device manages its own sensor data, interacts with a supervisor,
    executes assigned scripts, and coordinates its operations through
    a dedicated thread and shared synchronization primitives.
    """
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
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to the device, or timepoint is done.
        self.script_received = Event()
        # List to store (script, location) tuples for execution.
        self.scripts = []
        # Event to signal that the device has completed processing for the current timepoint.
        self.timepoint_done = Event()
        # The dedicated thread for this device's execution logic.
        self.thread = DeviceThread(self)
        # Global barrier for synchronizing all devices. Set by setup_devices.
        self.barr = None
        # Global lock for critical sections across devices. Set by setup_devices.
        self.lock = None
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures a shared barrier and a global lock among a group of devices.
        This method ensures that all devices have access to the same synchronization
        primitives for timepoint progression and thread-safe operations.
        The device with the smallest ID typically initializes these shared resources.

        Args:
            devices (list): A list of all Device objects participating in the simulation.
        """
        # The first device in the list (assuming it represents the leader or device with ID 0)
        # initializes the barrier if it hasn't been initialized already.
        if devices[0].barr is None and self.device_id == devices[0].device_id:
                # Create a new CondBarrier for all participating devices.
                barr = CondBarrier(len(devices))
                # Assign the created barrier to all devices.
                for i in devices:
                        i.barr = barr
        # Create a single global lock to be shared among all devices.
        lock = Lock()
        # Assign the global lock to all devices.
        for d in devices:
                d.lock = lock 

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.
        If `script` is None, it signals the completion of script assignments for the timepoint.

        Args:
            script (Script): The script object to be executed. If None, it signals
                             the end of scripts for the current timepoint.
            location: The data location pertinent to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # When script is None, it means the assignment phase is complete for this timepoint.
            # Both events are set to release any waiting threads/processes.
            self.timepoint_done.set()
            self.script_received.set()

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

class CondBarrier():
    """
    Implements a reusable barrier synchronization mechanism using a Condition variable.
    Threads arriving at the barrier wait until all expected threads have arrived,
    after which all are released simultaneously, and the barrier is reset for reuse.
    """
    def __init__(self, num_threads):
        """
        Initializes the CondBarrier.

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
            # If this is the last thread, wake up all waiting threads.
            self.cond.notify_all()
            # Reset the counter for the next cycle of the barrier.
            self.count_threads = self.num_threads
        else:
            # If not the last thread, wait until notified.
            self.cond.wait()
        self.cond.release() # Release the lock.


class DeviceThread(Thread):
    """
    The dedicated thread of execution for a Device object.
    It manages the device's operational logic within the simulation loop,
    including supervisor interaction, sequential script execution,
    and coordinating with other devices using a global barrier.
    """
    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device object associated with this thread.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
		

    def run(self):
        """
        The main execution loop for the device thread.
        It continuously processes timepoints, executes scripts sequentially
        under a global lock, and synchronizes with other devices.
        """
        while True:
            # Retrieve current neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If no neighbors are returned, it signifies the end of the simulation.
            if neighbours is None:
                break

            # Wait until scripts for the current timepoint have been assigned.
            # This Event is set by the assign_script method when script is None.
            self.device.timepoint_done.wait()

            # Iterate through and execute all assigned scripts for the current timepoint.
            # Note: All script execution is protected by a single global lock,
            # which can serialize operations across devices and locations.
            for (script, location) in self.device.scripts:
                # Acquire the global lock to ensure exclusive access during data processing.
                self.device.lock.acquire()
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
                self.device.lock.release() # Release the global lock.
            
            # Reset the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()
            # Synchronize with all other devices at the global barrier before starting the next timepoint.
            self.device.barr.wait()
