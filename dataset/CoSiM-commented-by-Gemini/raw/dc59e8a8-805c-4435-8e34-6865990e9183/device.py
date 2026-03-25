from threading import Event, Thread, Semaphore, Lock


class ReusableBarrier(object):
    """
    A reusable barrier synchronization primitive for a fixed number of threads.
    This implementation uses a two-phase approach to allow the barrier to be
    used multiple times. Threads are blocked at the end of each phase until
    all participating threads have arrived.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                               before any of them can proceed.
        """
        self.num_threads = num_threads
        # Counter for threads arriving at the first phase.
        self.count_threads1 = [self.num_threads]
        # Counter for threads arriving at the second phase.
        self.count_threads2 = [self.num_threads]
        
        # Lock to ensure atomic updates to the thread counters.
        self.count_lock = Lock()
        # Semaphore for the first phase barrier.
        self.threads_sem1 = Semaphore(0)
        
        # Semaphore for the second phase barrier.
        self.threads_sem2 = Semaphore(0)
        

    def wait(self):
        """
        Causes a thread to wait until all threads have called this method.
        The barrier is composed of two distinct phases to ensure that no thread
        can start a new wait cycle before all threads have completed the
        previous one.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Implements one phase of the barrier.

        Args:
            count_threads (list[int]): A list containing the counter for the
                                       current phase. Using a list allows
                                       modification by reference.
            threads_sem (Semaphore): The semaphore used to block and release
                                     threads for this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            
            # If this is the last thread to arrive at the barrier for this phase.
            if count_threads[0] == 0:
                # Release all waiting threads.
                for i in range(self.num_threads):
                    
                    
                    threads_sem.release()
                    
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
                
        # All threads will block here until the last thread releases the semaphore.
        threads_sem.acquire()
        


class Device(object):
    """
    Represents a device in a simulated distributed system. Each device runs in
    its own thread and can execute scripts that interact with its neighbors.
    """
    
    # A class-level barrier for synchronizing all device threads.
    barrier = None

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local
                                sensor data, keyed by location.
            supervisor (Supervisor): The supervisor object that manages the
                                     network of devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal the arrival of a new script.
        self.script_received = Event()
        self.scripts = []
        # Event to signal the completion of a timepoint.
        self.timepoint_done = Event()
        # The thread that executes the device's main loop.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the class-level barrier for all devices. This should be
        called once before the simulation starts.

        Args:
            devices (list[Device]): The list of all devices in the simulation.
        """
        Device.barrier = ReusableBarrier(len(devices))

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific location.

        Args:
            script (Script): The script object to be executed. If None, it
                             signals the end of the current timepoint.
            location (any): The location context for the script execution.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If no script is provided, it indicates all scripts for the
            # current timepoint have been assigned.
            self.timepoint_done.set()
        # Signal the device thread that a new script has been received.
        self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location.

        Args:
            location (any): The location from which to retrieve data.

        Returns:
            The sensor data at the given location, or None if not available.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location.

        Args:
            location (any): The location at which to update data.
            data (any): The new data value.
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
    The main execution thread for a Device. It continuously processes assigned
    scripts and synchronizes with other devices at each timepoint.
    """

    def __init__(self, device):
        """
        Initializes the device thread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop of the device thread. It processes scripts for each
        timepoint, synchronizing with other devices using a barrier.
        """
        while True:
            
            # Get the device's neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # If get_neighbours returns None, it's a signal to terminate.
                break

            script_index = 0
            script_threads = []
            length_scripts_threads = 0
            while True:
                # This block manages the execution of scripts in parallel,
                # with a hard limit of 8 concurrent script threads.
                if script_index < len(self.device.scripts):
                    if length_scripts_threads < 8:
                        # Spawn a new thread to execute a script.
                        thread = self.call_threads(neighbours, script_index)
                        if thread.is_alive():
                            script_threads.append((thread, True))
                            length_scripts_threads += 1
                        script_index += 1
                    else:
                        # If the thread limit is reached, clean up finished threads.
                        local_index = 0
                        while local_index < len(script_threads):
                            if (not script_threads[local_index][0].isAlive()
                                    and script_threads[local_index][1] is True):
                                script_threads[local_index] = (script_threads[local_index][0], False)
                                length_scripts_threads -= 1
                            local_index += 1
                elif self.device.timepoint_done.is_set():
                    # End of the current timepoint.
                    self.device.timepoint_done.clear()
                    self.device.script_received.clear()
                    break
                else:
                    # Wait for more scripts to be assigned.
                    self.device.script_received.wait()
                    self.device.script_received.clear()

            # Synchronize with all other devices before starting the next timepoint.
            Device.barrier.wait()

    def call_threads(self, neighbours, index):
        """
        Creates and starts a MyThread to execute a script.

        Args:
            neighbours (list[Device]): The list of neighboring devices.
            index (int): The index of the script to execute.

        Returns:
            MyThread: The thread that was created and started.
        """
        thread = MyThread(self.device, neighbours, self.device.scripts[index])
        thread.start()
        thread.join()
        return thread




class MyThread(Thread):
    """
    A thread that executes a single script, gathering data from neighboring
    devices, running the script, and distributing the results.
    """
    
    # A dictionary of locks to ensure exclusive access to locations.
    locations_locks = {}

    def __init__(self, device, neighbours, script_info):
        """
        Initializes a script execution thread.

        Args:
            device (Device): The device initiating the script execution.
            neighbours (list[Device]): The list of neighboring devices.
            script_info (tuple): A tuple containing the script and its location.
        """
        Thread.__init__(self)
        script, location = script_info
        self.location, self.script = location, script


        self.device, self.neighbours = device, neighbours

        # Create a lock for the location if it doesn't exist.
        if location not in MyThread.locations_locks:
            MyThread.locations_locks[location] = Lock()

    def run(self):
        """
        Executes the script. This involves acquiring a lock for the script's
        location, gathering data from neighbors, running the script, and then
        updating the data on all relevant devices.
        """
        # Acquire a lock for the script's location to prevent race conditions.
        MyThread.locations_locks[self.location].acquire()
        
        script_data = []
        
        # Gather data from all neighbors at the specified location.
        for device in self.neighbours:
            if device.get_data(self.location) is not None:
                script_data.append(device.get_data(self.location))
        
        # Also include the current device's data.
        if self.device.get_data(self.location) is not None:
            script_data.append(self.device.get_data(self.location))

        
        if script_data:
            
            # Execute the script with the gathered data.
            result = self.script.run(script_data)
            
            # Distribute the result to all neighbors.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Update the current device's data as well.
            self.device.set_data(self.location, result)
        # Release the lock for the location.
        MyThread.locations_locks[self.location].release()