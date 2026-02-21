


from threading import Event, Thread, Semaphore, Lock

"""
Implements a reusable barrier synchronization mechanism for a fixed number of threads.
This barrier ensures that all participating threads wait at a specific point until
every thread has reached that point before any can proceed. It utilizes two phases
of semaphores to allow for multiple synchronization cycles without reinitialization.
"""
class ReusableBarrierCond():
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        # Tracks the number of threads waiting in the first phase.
        self.count_threads1 = [self.num_threads]
        # Tracks the number of threads waiting in the second phase (for reusability).
        self.count_threads2 = [self.num_threads]
        # Mutex to protect access to thread counters.
        self.count_lock = Lock()                 
        # Semaphore for the first phase of waiting. Initialized to 0 so threads block immediately.
        self.threads_sem1 = Semaphore(0)         
        # Semaphore for the second phase of waiting, enabling reusability. Initialized to 0.
        self.threads_sem2 = Semaphore(0)         
 
    def wait(self):
        """
        Causes the calling thread to wait until all `num_threads` have reached this barrier.
        The barrier is reusable after all threads have passed.
        """
        # First phase of synchronization
        self.phase(self.count_threads1, self.threads_sem1)
        # Second phase of synchronization for reusability
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        Handles a single phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the current count of threads
                                  that have not yet reached this phase of the barrier.
                                  (Uses a list to allow modification within the Lock context).
            threads_sem (Semaphore): The semaphore associated with this phase.
        """
        with self.count_lock:
            # Decrement the count of waiting threads.
            count_threads[0] -= 1
            # Check if this is the last thread to reach the barrier.
            if count_threads[0] == 0:            
                # If all threads have arrived, release the semaphore for each waiting thread.
                for i in range(self.num_threads):
                    threads_sem.release()        
                # Reset the counter for the next use of this phase of the barrier.
                count_threads[0] = self.num_threads  
        # Each thread acquires the semaphore, blocking until all threads have released it.
        threads_sem.acquire()                    
        # This comment block was part of the original code, removed for brevity in new_string,
        # but conceptually, threads would proceed beyond this point once the semaphore is acquired.


class Device(object):
    """
    Represents a single device in a simulated distributed system.
    Each device manages its own sensor data, interacts with a supervisor,
    executes assigned scripts, and operates within its own thread of execution.
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
        # Event to signal when a new script has been assigned to the device.
        self.script_received = Event()
        # List to store (script, location) tuples for execution.
        self.scripts = list()

        # Barrier for synchronizing with other devices at the end of a timepoint.
        self.bar = None
        # Event to signal that the device has completed processing for the current timepoint.
        self.timepoint_done = Event()
        # The dedicated thread for this device's execution logic.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id
    
    
    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.

        Args:
            script (Script): The script object to be executed. If None, it signals
                             the end of scripts for the current timepoint.
            location: The data location pertinent to the script.
        """
        if script is None:
            # If no script, mark the current timepoint as done.
            self.timepoint_done.set()
        else:
            # Add the script and its location to the list for processing.
            self.scripts.append((script, location))
            # Signal that a new script has been received.
            self.script_received.set()


    
    def setup_devices(self, devices):
        """
        Configures a shared barrier among a group of devices.
        This method is typically called by a supervisor or central entity
        to set up synchronization for all participating devices.

        Args:
            devices (list): A list of Device objects that will share the barrier.
        """
        # Sort devices to ensure consistent barrier initialization,
        # specifically identifying the device with the maximum ID.
        devices.sort(key = lambda x: x.device_id, reverse = True)
        # The device with the highest ID is chosen to potentially initialize the barrier.
        id_maximum = devices[0].device_id
        # In this implementation, only the device with the maximum ID creates the barrier,
        # and then it is assigned to all devices in the list.
        if self.device_id == id_maximum:
            # Create a reusable barrier for all participating devices.
            barrier = ReusableBarrierCond(len(devices))
            for i in range(len(devices)):
                # Assign the same barrier instance to all devices.
                devices[i].barrier = barrier
    

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
    It handles the device's operational logic, including data exchange,
    script execution, and synchronization.
    """
    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device object associated with this thread.
        """
        self.device = device
        # Initialize the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        # Placeholder or unused variable, no clear functional utility here.
        locki =0


    def run(self):
        """
        The main execution loop for the device thread.
        It continuously processes timepoints, executes scripts,
        and synchronizes with other devices.
        """
        # Retrieve initial neighbor information from the supervisor.
        neighbours = self.device.supervisor.get_neighbours()
        # Loop continues as long as there are neighbors (i.e., the simulation is active).
        while neighbours is not None:
            # Wait until the current timepoint's scripts are assigned or marked as done.
            self.device.timepoint_done.wait()
            # Iterate through all assigned scripts for the current timepoint.
            for (script, location) in self.device.scripts:
                # List to accumulate data for the current script's execution.
                script_data = list()
                # Gather data from neighboring devices.
                for i in range (len(neighbours)):
                
                    data = neighbours[i].get_data(location)
                    # Skip if no data is available from a neighbor for this location.
                    if data is None:
                        continue
                    script_data.append(data)
                # Also include the device's own data if available.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # If there is data to process, execute the script.
                if script_data != list():
                    # Execute the script with the collected data.
                    result = script.run(script_data)

                    # Propagate the script's result back to neighbors.
                    for i in range(len(neighbours)):
                        neighbours[i].set_data(location, result)
                    
                    # Update the device's own data with the script's result.
                    self.device.set_data(location, result)
            # Clear the timepoint_done event, resetting it for the next timepoint.
            self.device.timepoint_done.clear()
            # Synchronize with other devices using the shared barrier.
            self.device.barrier.wait()
            # Request updated neighbor information for the next timepoint.
            neighbours = self.device.supervisor.get_neighbours()

