"""
This module implements a multi-threaded device simulation system.

It includes a `ReusableBarrier` for thread synchronization, a `Device` class
representing individual simulated devices, a `DeviceThread` for the main logic
of each device, and `ThreadAux` for auxiliary tasks such as script execution
and inter-device data exchange. The system uses threading primitives like
Events, Locks, and Semaphores to manage concurrency and ensure data consistency.

The simulation aims to model scenarios where devices interact with their neighbors,
process sensor data using assigned scripts, and synchronize their operations
across timepoints.
"""

from threading import Event, Thread, Lock, Semaphore, Lock

class ReusableBarrier(object):
    """
    Implements a reusable barrier synchronization primitive for a fixed number of threads.
    It ensures that all participating threads wait at a specific point until every
    thread has reached that point, and then all threads are released simultaneously.
    This implementation uses two phases to allow for reuse.
    """


    def __init__(self, num_threads):
        """
        Initializes the reusable barrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                before any are released.
        """
        self.num_threads = num_threads
        # Counter for threads in the first phase.
        self.count_threads1 = self.num_threads
        # Counter for threads in the second phase.
        self.count_threads2 = self.num_threads
        # Lock to protect the counters during modifications.
        self.counter_lock = Lock()
        # Semaphore for the first phase, initially blocking all threads.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for the second phase, initially blocking all threads.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        participating threads have also called wait().
        This method orchestrates the two phases of the barrier.
        """
        
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        Manages the first synchronization phase of the barrier.
        Threads acquire the counter_lock, decrement their count, and if they
        are the last thread, release all waiting threads in phase 1.
        """
        
        with self.counter_lock:
            self.count_threads1 -= 1
            # Check if this is the last thread to reach the barrier.
            if self.count_threads1 == 0:
                # Release all threads waiting on threads_sem1.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset the counter for the next use of the barrier.
                self.count_threads1 = self.num_threads

        # All threads wait here until released by the last thread in phase 1.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        Manages the second synchronization phase of the barrier.
        Similar to phase 1, but uses a separate counter and semaphore to allow
        the barrier to be reused.
        """
        
        with self.counter_lock:
            self.count_threads2 -= 1
            # Check if this is the last thread to reach the barrier.
            if self.count_threads2 == 0:

                # Release all threads waiting on threads_sem2.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset the counter for the next use of the barrier.
                self.count_threads2 = self.num_threads

        # All threads wait here until released by the last thread in phase 2.
        self.threads_sem2.acquire()



class Device(object):
    """
    Represents a single simulated device in the system.
    Each device has a unique ID, sensor data, and communicates with a supervisor
    and other devices. It manages its own main thread (`DeviceThread`) and
    a pool of auxiliary threads (`ThreadAux`) for parallel processing.
    """
    
    # A global barrier for all devices, initialized with a placeholder value.
    # It will be re-initialized in setup_devices.
    bar1 = ReusableBarrier(1)
    # A global event to signal when all devices are set up.
    event1 = Event()
    # A list of Locks, intended to protect shared data locations across devices.
    # Initialized globally and populated by the first device (device_id == 0).
    locck = []

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary mapping locations to sensor data values.
            supervisor (Supervisor): A reference to the supervisor object managing devices.
        """
        
        
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = [] # List to hold references to other devices in the simulation.

        
        # A list of Events for inter-thread synchronization within this device or with DeviceThread.
        # The size (11) is hardcoded, suggesting a fixed number of synchronization points.
        self.event = []
        for _ in xrange(11):
            self.event.append(Event())

        
        self.nr_threads_device = 8 # Number of auxiliary threads managed by this device.
        
        self.nr_thread_atribuire = 0 # Index to round-robin assign scripts to auxiliary threads.
        
        # Barrier for synchronization among the Device's auxiliary threads and DeviceThread.
        # The +1 accounts for the DeviceThread itself participating in this barrier.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device+1)

        
        # The main thread responsible for this device's overall control and neighbor interactions.
        self.thread = DeviceThread(self)
        self.thread.start()

        
        # A list of auxiliary threads for parallel script execution.
        self.threads = []
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        # Start all auxiliary threads.
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global resources that depend on the total number of devices.
        This method is designed to be called once, typically by device with ID 0.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        self.devices = devices
        
        # Only the device with ID 0 performs global initialization.
        if self.device_id == 0:
            # Initialize a pool of global locks for various data locations.
            # The hardcoded number (30) suggests a fixed maximum number of shared locations.
            for _ in xrange(30):
                Device.locck.append(Lock())
            # Re-initialize the global barrier with the actual total number of devices.
            Device.bar1 = ReusableBarrier(len(devices))
            
            # Signal that global setup is complete.
            Device.event1.set()

    def assign_script(self, script, location):
        """
        Assigns a processing script to one of the auxiliary threads in a round-robin manner.

        Args:
            script (callable): The script (function or object with a run method) to be executed.
            location (int): The data location this script pertains to.
        """
        
        if script is not None:
            # Assigns the script and its location to the next available auxiliary thread.
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            
            # Moves to the next auxiliary thread for the subsequent assignment.
            self.nr_thread_atribuire = (self.nr_thread_atribuire+1)%\
            self.nr_threads_device
        else:
            # If no script is provided, it signals that timepoint processing is done.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (int): The location ID for which to retrieve data.

        Returns:
            Any: The data at the specified location, or None if the location is not found.
        """
        
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        Args:
            location (int): The location ID for which to set data.
            data (Any): The new data value to be set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device, joining all its managed threads.
        """
        
        self.thread.join()
        for threadd in self.threads:
            threadd.join()


class DeviceThread(Thread):
    """
    The main thread for a Device. It handles interactions with the supervisor,
    retrieves neighbors, and coordinates timepoint synchronization.
    """
    

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None # Stores the list of neighboring devices.
        self.contor = 0      # Counter used to index into the device's list of events.

    def run(self):
        """
        The main execution loop for the DeviceThread.
        It waits for global setup, then continuously processes timepoints:
        gets neighbors, synchronizes, and signals completion.
        """
        # Wait for all devices to be set up globally.
        Device.event1.wait()

        while True:
            
            # Retrieves the list of neighboring devices from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()

            # If no neighbors are returned (or a shutdown signal), exit the loop.
            if self.neighbours is None:
                self.device.event[self.contor].set() # Signal completion for the current timepoint.
                break

            
            # Waits until all scripts for the current timepoint are assigned or timepoint_done is set.
            self.device.timepoint_done.wait()
            # Clears the event for the next timepoint.
            self.device.timepoint_done.clear()

            
            # Signals to auxiliary threads that they can proceed for the current timepoint.
            self.device.event[self.contor].set()
            # Increments the counter to use the next event for the next timepoint.
            self.contor += 1

            
            # Waits for all auxiliary threads of this device to complete their current timepoint processing.
            self.device.bar_threads_device.wait()

            
            # Global barrier synchronization: Waits for all DeviceThreads across all devices
            # to complete their timepoint processing before proceeding to the next.
            Device.bar1.wait()

class ThreadAux(Thread):
    """
    An auxiliary thread for a Device, responsible for executing assigned scripts
    and managing data exchange with neighboring devices at specific locations.
    """
    
    def __init__(self, device):
        """
        Initializes an auxiliary thread.

        Args:
            device (Device): The Device instance this thread belongs to.
        """
        Thread.__init__(self)
        self.device = device
        # Dictionary to store scripts and the locations they operate on.
        self.script_loc = {}
        # Counter used to index into the device's list of events, synchronizing with DeviceThread.
        self.contor = 0

    def run(self):
        """
        The main execution loop for the auxiliary thread.
        It continuously waits for signals to process assigned scripts for a timepoint,
        exchanges data with neighbors, and synchronizes with other threads.
        """
        while True:
            
            
            # Waits for the DeviceThread to signal that a new timepoint's processing can begin.
            self.device.event[self.contor].wait()
            # Increments the counter to wait on the next event for the subsequent timepoint.
            self.contor += 1

            
            # Retrieves the list of neighbors from the DeviceThread.
            neigh = self.device.thread.neighbours
            # If no neighbors (shutdown signal), break the loop.
            if neigh is None:
                break

            # Iterates through all scripts assigned to this auxiliary thread.
            for script in self.script_loc:
                location = self.script_loc[script]
                
                
                # Acquires a global lock for the specific data location to prevent race conditions.
                Device.locck[location].acquire()
                script_data = [] # Collects data from neighbors and self for the script.

                # Gathers data from all neighboring devices for the current location.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Gathers data from its own device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # If there is data to process, execute the script.
                if script_data != []:
                    result = script.run(script_data) # Executes the script with collected data.
                    # Updates the data in neighboring devices with the script's result.
                    for device in neigh:
                        device.set_data(location, result)
                    # Updates its own device's data with the script's result.
                    self.device.set_data(location, result)

                
                # Releases the global lock for the data location.
                Device.locck[location].release()

            
            # Waits at the device-specific barrier, synchronizing with DeviceThread
            # and other auxiliary threads of this device.
            self.device.bar_threads_device.wait()
