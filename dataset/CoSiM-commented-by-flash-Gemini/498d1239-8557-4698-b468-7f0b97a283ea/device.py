

"""
@498d1239-8557-4698-b468-7f0b97a283ea/device.py
@brief This module implements a distributed device simulation framework,
featuring custom `ReusableBarrier` for synchronization and multi-threaded
`Device` operation using `DeviceThread` and `ThreadAux` for task execution.
"""

from threading import Event, Thread, Lock, Semaphore, Lock

class ReusableBarrier(object):
    """
    A reusable barrier synchronization primitive implementing a two-phase wait
    mechanism using semaphores and a lock.

    Threads wait in two distinct phases, allowing for efficient resetting and reuse.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                in each phase before proceeding.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for threads in phase 1.
        self.count_threads2 = self.num_threads # Counter for threads in phase 2.

        self.counter_lock = Lock() # Lock to protect access to thread counters.
        self.threads_sem1 = Semaphore(0) # Semaphore for releasing threads from phase 1.
        self.threads_sem2 = Semaphore(0) # Semaphore for releasing threads from phase 2.

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have passed both
        phase 1 and phase 2 of the barrier.
        """
        self.phase1() # Execute the first phase of the barrier.
        self.phase2() # Execute the second phase of the barrier.

    def phase1(self):
        """
        First phase of the barrier. All threads decrement a counter and wait
        on `threads_sem1` until the last thread releases them.
        """
        # Block Logic: Atomically decrement the counter and check if this is the last thread.
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # If this is the last thread, release all waiting threads from phase 1.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset counter for next use.

        self.threads_sem1.acquire() # Acquire (wait on) the semaphore until released by the last thread.

    def phase2(self):
        """
        Second phase of the barrier. Similar to phase 1, but uses `threads_sem2`
        and `count_threads2`.
        """
        # Block Logic: Atomically decrement the counter and check if this is the last thread.
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # If this is the last thread, release all waiting threads from phase 2.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset counter for next use.

        self.threads_sem2.acquire() # Acquire (wait on) the semaphore until released by the last thread.



class Device(object):
    """
    Represents a single device in a simulated distributed system.

    Each device manages its sensor data, processes scripts using a pool
    of worker threads (`ThreadAux`), and coordinates globally through
    `DeviceThread` and shared barriers.
    """
    
    # Class-level attributes for global synchronization and shared resources.
    bar1 = ReusableBarrier(1)  # Global barrier for synchronizing all devices.
    event1 = Event()           # Event to signal when global setup is complete.
    locck = []                 # List of locks for each data location, shared across devices.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary representing the sensor data this device holds.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        # Event to signal when the current timepoint's scripts have been assigned.
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = [] # List to store references to all devices in the system.

        # List of Events for inter-thread communication within this device.
        # Each event can be used to signal specific ThreadAux instances.
        self.event = []
        for _ in xrange(11): # Using xrange for Python 2 compatibility.
            self.event.append(Event())

        self.nr_threads_device = 8 # Number of auxiliary worker threads for this device.
        
        self.nr_thread_atribuire = 0 # Counter to distribute scripts among ThreadAux instances.
        
        # Barrier for synchronizing auxiliary worker threads within this device.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device+1) # +1 for DeviceThread.

        # Main thread for device operations (e.g., fetching neighbors).
        self.thread = DeviceThread(self)
        self.thread.start()

        # Block Logic: Create and start auxiliary worker threads.
        self.threads = []
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start() # Start each auxiliary thread.

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures global shared resources like the main barrier and location locks.

        This method is typically called by the supervisor during initial setup.

        Args:
            devices (list): A list of all Device instances in the system.
        """
        self.devices = devices
        
        # Block Logic: Only device 0 initializes the global location locks and the global barrier.
        if self.device_id == 0:
            for _ in xrange(30): # Initialize 30 shared locks for data locations.
                Device.locck.append(Lock())
            Device.bar1 = ReusableBarrier(len(devices)) # Initialize global barrier with total device count.
            
            Device.event1.set() # Signal that global setup is complete.

    def assign_script(self, script, location):
        """
        Assigns a script to one of the auxiliary worker threads in a round-robin fashion.

        Args:
            script (object): The script object to be executed.
            location (int): The data location relevant to the script.
        """
        if script is not None:
            # Assign the script and its location to the next available auxiliary thread.
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            
            # Move to the next auxiliary thread for the next assignment.
            self.nr_thread_atribuire = (self.nr_thread_atribuire+1)%\
            self.nr_threads_device
        else:
            self.timepoint_done.set() # Signal that script assignment for this timepoint is complete.

    def get_data(self, location):
        """
        Retrieves data from the device's sensor_data at the specified location.

        Args:
            location (int): The index or key of the data to retrieve.

        Returns:
            any: The data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates data in the device's sensor_data at the specified location.

        Args:
            location (int): The index or key of the data to set.
            data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data # Update the data if the location exists.

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device, joining all its associated threads.
        """
        self.thread.join() # Wait for the main DeviceThread to complete.
        for threadd in self.threads:
            threadd.join() # Wait for each auxiliary worker thread to complete.


class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for fetching neighbors,
    managing synchronization events, and coordinating with auxiliary worker threads.
    """
    

    def __init__(self, device):
        """
        Initializes a DeviceThread instance.

        Args:
            device (Device): The parent Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None # Stores the list of neighboring devices.
        self.contor = 0 # Counter used to manage events for auxiliary threads.

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It waits for global setup to complete, continuously fetches neighbors,
        synchronizes at various timepoints, and manages events to signal
        auxiliary threads for script execution.
        """
        # Pre-condition: Wait until global device setup (including Device.event1) is complete.
        Device.event1.wait()

        while True:
            # Functional Utility: Fetch the list of neighboring devices from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                # If no neighbors are returned (e.g., supervisor signals shutdown),
                # set the corresponding event to unblock auxiliary threads and break.
                self.device.event[self.contor].set()
                break

            # Block Logic: Wait for the current timepoint's scripts to be assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Reset the event for the next timepoint.

            # Functional Utility: Signal auxiliary threads that script assignment for this timepoint is ready.
            self.device.event[self.contor].set()
            self.contor += 1 # Increment event counter for the next timepoint.

            # Block Logic: Synchronize all worker threads within this device.
            # This barrier ensures all auxiliary threads are ready for script execution.
            self.device.bar_threads_device.wait()

            # Block Logic: Global synchronization point for all devices.
            # This barrier ensures all devices complete their operations for the current timepoint.
            Device.bar1.wait()

class ThreadAux(Thread):
    """
    An auxiliary worker thread for a Device, responsible for executing a subset of assigned scripts.

    These threads work in parallel to process scripts, collecting data from
    neighbors and local sensors, then updating relevant data.
    """
    
    def __init__(self, device):
        """
        Initializes a ThreadAux instance.

        Args:
            device (Device): The parent Device instance this thread belongs to.
        """
        Thread.__init__(self)
        self.device = device
        self.script_loc = {} # Dictionary to store scripts assigned to this auxiliary thread, mapping script to its location.
        self.contor = 0 # Counter to track which event to wait on.

    def run(self):
        """
        The main execution loop for the auxiliary worker thread.

        It waits for a signal from the main DeviceThread, processes its assigned
        scripts, and then synchronizes with other auxiliary threads within the same device.
        """
        while True:
            # Pre-condition: Wait for the main DeviceThread to signal that it's time to process scripts.
            self.device.event[self.contor].wait()
            self.contor += 1 # Increment event counter for the next timepoint.

            # Functional Utility: Get the list of neighboring devices from the main DeviceThread.
            neigh = self.device.thread.neighbours
            if neigh is None: # If neighbors are None (shutdown signal), break the loop.
                break

            # Block Logic: Execute each script assigned to this auxiliary thread.
            for script in self.script_loc:
                location = self.script_loc[script] # Get the data location for the current script.
                
                # Pre-condition: Acquire a global lock for the specific data location to ensure exclusive access.
                Device.locck[location].acquire()
                script_data = [] # List to collect data for the script.

                # Block Logic: Collect data from neighboring devices.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Block Logic: Collect data from the current device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: If there is data to process, execute the script.
                if script_data != []:
                    result = script.run(script_data) # Action: Execute the script.
                    # Block Logic: Update data in neighboring devices.
                    for device in neigh:
                        device.set_data(location, result)
                    # Block Logic: Update data in the current device.
                    self.device.set_data(location, result)

                # Post-condition: Release the global lock for the data location.
                Device.locck[location].release()

            # Block Logic: Synchronize with other auxiliary threads within this device.
            # This ensures all auxiliary threads finish script execution before proceeding.
            self.device.bar_threads_device.wait()
