




"""
This module implements a simulation framework for distributed devices,
focusing on concurrent execution of scripts and synchronized data processing.
It includes a reusable barrier for thread synchronization and classes for
device simulation and auxiliary script execution threads.
"""

from threading import Event, Thread, Lock, Semaphore, Lock

class ReusableBarrier(object):
    """
    A reusable barrier synchronization primitive for a fixed number of threads.

    This barrier allows a set of threads to wait for each other to reach a common
    point before any of them can proceed. It is designed to be reusable across
    multiple synchronization points.

    Algorithm: Implements a double-barrier pattern using two semaphores and a mutex.
               The double-barrier ensures that all threads have left the barrier
               before it can be reused for the next synchronization phase,
               preventing issues where a fast thread might re-enter before slow
               threads have exited.
    """
    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must arrive
                               at the barrier before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads


        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        `num_threads` threads have also called `wait()`.
        """
        
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        Manages the first phase of the double-barrier synchronization.

        Block Logic: Decrements a counter. If it reaches zero, all threads
                     have arrived, and semaphores for the first phase are released,
                     allowing threads to proceed. The counter is then reset.
        """
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):


                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """
        Manages the second phase of the double-barrier synchronization.

        Block Logic: Decrements a counter for the second phase. If it reaches zero,
                     all threads have passed through the first phase, and semaphores
                     for the second phase are released. The counter is then reset.
                     This second phase is crucial for ensuring the barrier is
                     fully reset and reusable without race conditions.
        """
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()



        self.threads_sem2.acquire()



class Device(object):
    """
    Represents a simulated device within a distributed system.

    Each device manages its sensor data, interacts with a central supervisor,
    and can execute assigned scripts using a pool of auxiliary threads.
    It handles synchronization for data processing and communication.
    """
    
    # Global barrier for synchronizing all devices in the simulation.
    bar1 = ReusableBarrier(1) # Initialized with 1 as a placeholder, will be re-initialized in setup_devices.
    # Global event to signal when initial device setup is complete.
    event1 = Event()
    # List of locks, one for each data location, to protect sensor data during concurrent access.
    locck = []

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor readings
                                (e.g., {location: data}).
            supervisor (object): An object representing the central supervisor,
                                 used for coordination (e.g., getting neighbors).
        """
        
        # Event to signal when the current timepoint's script assignments are done.
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # List to store references to other devices in the system.
        self.devices = []

        # List of Events, one for each auxiliary thread, used for signaling.
        self.event = []
        for _ in xrange(11): # Invariant: A fixed number of events (11) are pre-allocated.
            self.event.append(Event())

        # Number of auxiliary threads assigned to this device for script execution.
        self.nr_threads_device = 8
        
        # Counter for round-robin assignment of scripts to auxiliary threads.
        self.nr_thread_atribuire = 0
        
        # Barrier for synchronizing the DeviceThread and its auxiliary threads.
        # It includes an extra slot for the DeviceThread itself.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device+1)

        # Main thread for the device, responsible for supervisor interaction.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Pool of auxiliary threads for script execution.
        self.threads = []
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        # Block Logic: Starts all auxiliary threads.
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the list of all devices in the simulated system.

        This method is called once during initialization to provide a global
        view of all participating devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.devices = devices
        
        # Block Logic: For device_id 0, initializes global synchronization primitives.
        # This acts as a coordinator for the global barrier and locks.
        if self.device_id == 0:
            # Initializes a list of Locks, one for each data location, to protect sensor data during script execution.
            for _ in xrange(30): # Pre-condition: Assumes a maximum of 30 data locations.
                Device.locck.append(Lock())
            # Reinitializes the global barrier with the total number of devices.
            Device.bar1 = ReusableBarrier(len(devices))
            
            # Signals that the global device setup is complete.
            Device.event1.set()

    def assign_script(self, script, location):
        """
        Assigns a script to one of the auxiliary threads for execution at a specific data location.

        Scripts are assigned in a round-robin fashion to balance workload among auxiliary threads.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the sensor data location the script operates on.
        """
        if script is not None:
            # Assigns the script and its location to the next available auxiliary thread.
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            
            # Increments the assignment counter, wrapping around to distribute scripts evenly.
            self.nr_thread_atribuire = (self.nr_thread_atribuire+1)%\
            self.nr_threads_device
        else:
            # If script is None, it signifies the end of script assignments for the current timepoint.
            # Functional Utility: Signals that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new sensor data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Performs a graceful shutdown of all threads associated with this device.
        """
        # Waits for the main device thread to complete its execution.
        self.thread.join()
        # Waits for all auxiliary threads to complete their execution.
        for threadd in self.threads:
            threadd.join()


            threadd.join()


class DeviceThread(Thread):
    """
    A dedicated thread for a Device, managing its interaction with the supervisor
    and coordinating synchronization with its auxiliary threads.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Stores a list of neighboring devices, fetched from the supervisor.
        self.neighbours = None
        # Counter for signaling auxiliary threads.
        self.contor = 0

    def run(self):
        """
        Main execution loop for the DeviceThread.

        Architectural Intent: Continuously fetches neighbor information,
        synchronizes with its device's auxiliary threads, and participates
        in global synchronization barriers.
        """
        # Block Logic: Waits for the global device setup to be complete.
        Device.event1.wait()

        while True:
            
            # Block Logic: Fetches the current list of neighboring devices from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()

            # Termination Condition: If no neighbors are returned (None), it signals
            # its auxiliary threads to break and then terminates itself.
            if self.neighbours is None:
                self.device.event[self.contor].set() # Signals all auxiliary threads to break.
                break

            
            # Block Logic: Waits for all scripts for the current timepoint to be assigned.
            self.device.timepoint_done.wait()
            # Resets the event for the next timepoint.
            self.device.timepoint_done.clear()

            
            # Block Logic: Signals the auxiliary threads (using the current counter)
            # that new neighbor information or data is ready for processing.
            self.device.event[self.contor].set()
            # Increments the counter to use the next event for the next cycle.
            self.contor += 1

            
            
            # Block Logic: Waits at the local barrier for all auxiliary threads to complete their processing for the current cycle.
            self.device.bar_threads_device.wait()

            
            
            # Block Logic: Waits at the global barrier for all devices (including their DeviceThreads)
            # to complete their local processing before proceeding to the next simulation step.
            Device.bar1.wait()

            
            
            # Block Logic: Waits at the global barrier for all devices (including their DeviceThreads)
            # to complete their local processing before proceeding to the next simulation step.
            Device.bar1.wait()

class ThreadAux(Thread):
    """
    An auxiliary thread for a Device, responsible for executing assigned scripts
    on sensor data, potentially involving data from neighboring devices.
    """
    
    def __init__(self, device):
        """
        Initializes a ThreadAux instance.

        Args:
            device (Device): The Device instance this thread is associated with.
        """
        Thread.__init__(self)
        self.device = device
        # Dictionary to store scripts assigned to this thread and their corresponding data locations.
        self.script_loc = {}
        # Counter to track which event (signal) to wait for from the DeviceThread.
        self.contor = 0

    def run(self):
        """
        Main execution loop for the ThreadAux.

        Architectural Intent: Waits for signals from the DeviceThread,
        collects relevant sensor data (including from neighbors),
        executes assigned scripts, and updates data.
        """
        while True:
            # Block Logic: Waits for a signal from the DeviceThread, indicating a new cycle of processing.
            # The 'contor' ensures each auxiliary thread waits on a unique event to prevent race conditions.
            self.device.event[self.contor].wait()
            # Increments the counter to prepare for the next signal.
            self.contor += 1

            
            # Block Logic: Retrieves the latest neighbors information from the DeviceThread.
            neigh = self.device.thread.neighbours
            # Termination Condition: If neighbors are None, it signals that the simulation is ending for this device.
            if neigh is None:
                break

            # Block Logic: Iterates through all scripts assigned to this auxiliary thread.
            for script in self.script_loc:
                location = self.script_loc[script]
                
                # Functional Utility: Acquires a lock for the specific data location to ensure
                # atomic access and prevent race conditions during data read/write operations.
                Device.locck[location].acquire()
                script_data = []

                # Block Logic: Collects sensor data from neighboring devices at the specified location.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Block Logic: Collects sensor data from its own device at the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: If data is available, executes the script and updates data in neighbors and its own device.
                if script_data != []:
                    result = script.run(script_data) # Functional Utility: Executes the assigned script with collected data.
                    # Updates data on neighboring devices.
                    for device in neigh:
                        device.set_data(location, result)
                    # Updates data on its own device.
                    self.device.set_data(location, result)

                
                # Functional Utility: Releases the lock for the data location after processing is complete.
                Device.locck[location].release()

            
            # Block Logic: Waits at the local barrier for all auxiliary threads to complete their script execution for the current cycle.
            self.device.bar_threads_device.wait()
