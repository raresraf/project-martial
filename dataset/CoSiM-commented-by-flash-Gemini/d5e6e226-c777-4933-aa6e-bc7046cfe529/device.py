"""
This module implements a device simulation framework that utilizes a multi-threaded
approach for distributed script execution and synchronization. It defines:
- ReusableBarrier: A re-usable barrier mechanism for threads using a Condition variable.
- Device: Represents a simulated device managing sensor data and script assignments,
          and orchestrating its own pool of DeviceThread workers.
- DeviceThread: Worker threads responsible for executing assigned scripts and
                synchronizing with other devices.

The system uses Events, Locks, and Condition variables for inter-thread communication and data consistency.
"""


from threading import Lock, Event, Thread, Condition

class ReusableBarrier():
    """
    Implements a reusable barrier synchronization mechanism using a condition variable.
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
        self.count_threads = self.num_threads    # Counter for threads currently waiting at the barrier.
        self.cond = Condition()                  # Condition variable for synchronization.
                                                 

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have reached this barrier.
        When the last thread arrives, all waiting threads are notified.
        """
        self.cond.acquire()                      # Acquire the condition variable's intrinsic lock.
        self.count_threads -= 1
        # Block Logic: If this is the last thread to reach the barrier,
        # reset the counter and notify all waiting threads.
        if self.count_threads == 0:
            self.cond.notify_all()               # Notify all waiting threads.
            self.count_threads = self.num_threads    # Reset counter for next use.
        else:
            self.cond.wait()                    # Wait until all threads have arrived.
        self.cond.release()                     # Release the condition variable's intrinsic lock.

class Device(object):
    """
    Represents a simulated device in a distributed system. Each device manages
    its sensor data, assigns and executes scripts via its `DeviceThread` workers,
    and interacts with a supervisor. Synchronization is handled through shared
    locks and a global barrier.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for various locations.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are assigned
        self.scripts = [] # List to store assigned scripts (script, location) tuples
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing
        self.gotneighbours = Event() # Event to signal when neighbor information is available
        self.zavor = Lock() # Lock to protect access to `neighbours` and `gotneighbours`
        self.threads = [] # List to hold DeviceThread instances (worker threads)
        self.neighbours = [] # List to store neighboring devices
        self.nthreads = 8 # Number of worker threads for this device
        self.barrier = ReusableBarrier(1) # Global barrier for device synchronization (initialized in setup_devices)
        self.lockforlocation = {} # Dictionary to store locks for different locations
        self.num_locations = supervisor.supervisor.testcase.num_locations # Total number of locations in the testcase
        # Block Logic: Create and start `nthreads` DeviceThread instances.
        for i in xrange(self.nthreads):
            self.threads.append(DeviceThread(self, i))
        for i in xrange(self.nthreads):
            self.threads[i].start()


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
        This includes initializing the global barrier and location-specific locks,
        and propagating these to all devices.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        # Block Logic: Initialize a shared barrier for all DeviceThreads across all devices.
        barrier = ReusableBarrier(devices[0].nthreads*len(devices))
        lockforlocation = {}
        # Block Logic: Initialize a lock for each possible location.
        for i in xrange(0, devices[0].num_locations):
            lock = Lock()
            lockforlocation[i] = lock
        # Block Logic: Propagate the initialized barrier and location locks to all devices.
        for i in xrange(0, len(devices)):
            devices[i].barrier = barrier
            devices[i].lockforlocation = lockforlocation


    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific location.
        If a script is provided, it's added to the device's script list and
        the `script_received` event is set. If no script is provided (None),
        it signals that a timepoint is done.

        Args:
            script (Script or None): The script object to assign, or None to signal timepoint completion.
            location (str): The location associated with the script.
        """
        
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list
            self.script_received.set() # Signal that a new script is available
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
        Shuts down the device by joining all its worker threads.
        """
        
        for i in xrange(self.nthreads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A worker thread for a `Device` instance. Each thread is responsible for
    processing a subset of assigned scripts and managing data for its device.
    It synchronizes with other `DeviceThread` instances within the same `Device`
    using a shared barrier.
    """
    
    def __init__(self, device, id_thread):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent Device instance this thread belongs to.
            id_thread (int): A unique identifier for this thread within its parent device.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id_thread = id_thread

    def run(self):
        """
        Executes the main logic of the DeviceThread.
        - Continuously fetches neighbor information from the supervisor.
        - If no neighbors are returned (signal for shutdown), the loop breaks.
        - Waits for timepoint processing to be signaled.
        - Processes a subset of assigned scripts, acquiring and releasing location-specific locks.
        - Updates sensor data based on script results.
        - Synchronizes with a global barrier after processing.
        """
        
        # Block Logic: Main loop for continuous processing of timepoints.
        while True:
            
            # Block Logic: Acquire lock to protect access to `neighbours` and `gotneighbours` event.
            self.device.zavor.acquire()
            # Block Logic: If neighbor information hasn't been fetched for this timepoint, fetch it.
            if self.device.gotneighbours.is_set() == False:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.gotneighbours.set() # Signal that neighbors are now available.
            self.device.zavor.release() # Release lock.
            
            # Block Logic: If no neighbors are returned, it's a shutdown signal.
            if self.device.neighbours is None:
                break # Exit the main loop

            # Block Logic: Wait for scripts to be assigned for the current timepoint.
            self.device.timepoint_done.wait()
            
            myscripts = [] # List to store scripts assigned to this specific thread.
            # Block Logic: Distribute assigned scripts to threads in a round-robin fashion.
            # Invariant: Each DeviceThread processes a distinct subset of the overall scripts.
            for i in xrange(self.id_thread, len(self.device.scripts), self.device.nthreads + 1):
                myscripts.append(self.device.scripts[i])

            # Block Logic: Process each script assigned to this thread.
            for (script, location) in myscripts:
                self.device.lockforlocation[location].acquire() # Acquire a lock for the specific location.
                script_data = [] # List to collect data for the current script.
                
                # Block Logic: Collect data from neighboring devices for the current location.
                for device in self.device.neighbours:
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
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    # Block Logic: Update the current device's sensor data with the script's result.
                    self.device.set_data(location, result)
                self.device.lockforlocation[location].release() # Release the lock for the current location.

            # Block Logic: Synchronize all DeviceThreads using the global barrier.
            self.device.barrier.wait()
            
            # Block Logic: Only the first thread (id_thread 0) clears events for the next cycle.
            if self.id_thread == 0:
                self.device.timepoint_done.clear() # Clear timepoint_done for the next cycle.
                self.device.gotneighbours.clear() # Clear gotneighbours for the next cycle.
            
            self.device.barrier.wait() # Final synchronization before the next timepoint.
            