
"""
This module implements a device simulation framework with a focus on distributed
script execution and synchronization using a reusable barrier. It defines:
- ReusableBarrier: A re-usable barrier mechanism for threads.
- Device: Represents a simulated device managing sensor data and script assignments.
- DeviceThread: The main thread for a Device, coordinating script execution and data sharing.

The system uses events and locks for inter-thread communication and data consistency.
"""


from threading import Event, Thread, Semaphore, Lock

class ReusableBarrier(object):
    """
    Implements a reusable barrier synchronization mechanism using semaphores and a lock.
    This barrier allows a fixed number of threads to wait until all have reached a certain point
    before any are allowed to proceed, and can then be reset for subsequent synchronizations.
    """


    def __init__(self, num_threads):
        """
        Initializes the reusable barrier.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can proceed.
        """


        self.num_threads = num_threads
        # Using lists to allow modification within nested scopes (e.g., 'with self.count_lock:')
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]


        self.count_lock = Lock() # Protects access to the count_threads variables

        # Semaphores for each phase of the barrier
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have reached this barrier.
        Uses a two-phase approach to allow reusability.
        """
        

        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier synchronization. Threads decrement a counter,
        and the last thread to reach zero releases all waiting threads for that phase.

        Args:
            count_threads (list): A single-element list containing the current count of waiting threads for this phase.
            threads_sem (Semaphore): The semaphore used to block/release threads for this phase.
        """
        

        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                
                for i in range(self.num_threads):
                    
                    
                    threads_sem.release()
                
                count_threads[0] = self.num_threads
        
        threads_sem.acquire()
        

class Device(object):
    """
    Represents a simulated device in a distributed system. Each device manages
    its sensor data, assigns and executes scripts, and interacts with a supervisor.
    Synchronization across devices is managed using a shared barrier and
    location-specific locks.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for various locations.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        

        self.barrier = None # Global barrier for device synchronization
        self.InitializationEvent = Event() # Event to signal completion of device initialization
        self.LockLocation = None # Dictionary to store locks for different locations (shared across devices)
        self.LockDict = Lock() # Lock to protect access to LockLocation

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are assigned
        self.scripts = [] # List to store assigned scripts (script, location) tuples
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing
        self.thread = DeviceThread(self) # The main thread for this device

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs initial setup for all devices in the simulation.
        The device with ID 0 initializes the global barrier and location locks,
        then propagates them to other devices. Other devices wait for this setup.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        
        # Block Logic: Device with ID 0 acts as the master to set up global synchronization.
        if self.device_id == 0:
            
            # Inline: n represents the total number of devices.
            n = len(devices)
            self.barrier = ReusableBarrier(n)    # Initialize global barrier for all devices
            self.LockLocation = {} # Initialize dictionary for location-specific locks

            # Block Logic: Propagate the initialized synchronization objects to all other devices.
            for idx in range(len(devices)):
                d = devices[idx]
                d.LockLocation = self.LockLocation # Share the same LockLocation dictionary
                d.barrier = self.barrier           # Share the same global barrier
                if d.device_id == 0:
                    pass # Master device doesn't need to signal itself
                else:
                    d.InitializationEvent.set() # Signal that initialization is complete for other devices
        else:
            self.InitializationEvent.wait() # Wait for the master device to complete initialization

        self.thread.start() # Start the main device thread after setup

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific location.
        If a script is provided, it's added to the device's script list.
        If no script is provided (None), it signals that a timepoint is done.

        Args:
            script (Script or None): The script object to assign, or None to signal timepoint completion.
            location (str): The location associated with the script.
        """
        
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list
        else:
            self.timepoint_done.set() # Signal that processing for the current timepoint is complete

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            Any or None: The sensor data if available for the location, otherwise None.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        Args:
            location (str): The location for which to set data.
            data (Any): The new sensor data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its main device thread.
        """
        
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for orchestrating script execution
    and data sharing within the device and with neighbors. It synchronizes
    with a global barrier and manages location-specific locks.
    """
    

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent Device instance this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Executes the main logic of the DeviceThread.
        - Continuously synchronizes with the global barrier.
        - Fetches neighbor information from the supervisor.
        - If no neighbors are returned (signal for shutdown), the loop breaks.
        - Waits for scripts to be assigned for the current timepoint.
        - Acquires location-specific locks, executes scripts, and updates data on devices.
        - Releases location-specific locks.
        - Clears the timepoint_done event for the next cycle.
        """
        
        # Block Logic: Main loop for continuous processing of timepoints.
        while True:
            # Block Logic: Synchronize all devices at the global barrier.
            self.device.barrier.wait()

            # Block Logic: Fetch updated neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: If no neighbors are returned, it's a shutdown signal.
            if neighbours is None:
                break # Exit the main loop

            # Block Logic: Wait for scripts to be assigned for the current timepoint.
            self.device.timepoint_done.wait()
            dev_scripts = self.device.scripts # Retrieve the scripts assigned to this device

            # Block Logic: Process each assigned script.
            for (script, location) in self.device.scripts:
                # Block Logic: Acquire a lock for the LockDict to manage location locks.
                self.device.LockDict.acquire()

                # Block Logic: If a lock for the current location doesn't exist, create one.
                if location not in self.device.LockLocation.keys():
                    self.device.LockLocation[location] = Lock()

                # Block Logic: Acquire the specific lock for the current location.
                self.device.LockLocation[location].acquire()

                # Block Logic: Release the LockDict lock after managing the location lock.
                self.device.LockDict.release()

                script_data = []
                # Block Logic: Collect data from neighboring devices for the current location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collect data from the current device itself for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: If data was collected, execute the script and update devices.
                if script_data != []:
                    # Inline: Execute the assigned script with collected data.
                    result = script.run(script_data)

                    # Block Logic: Propagate the script's result to neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Update the current device's sensor data with the script's result.
                    self.device.set_data(location, result)

                # Block Logic: Release the lock for the current location.
                self.device.LockLocation[location].release()

            # Block Logic: Clear the timepoint_done event for the next cycle.
            self.device.timepoint_done.clear()