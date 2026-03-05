"""
This module defines classes for simulating a distributed system, where individual
`Device` instances process sensor data and execute scripts in a multi-threaded
environment. It includes mechanisms for inter-device communication, synchronization,
and workload distribution among worker threads.
"""


from threading import Event, Thread

from threading import Condition, RLock    




class ReusableBarrier():
    """
    A reusable barrier synchronization primitive for coordinating multiple threads.

    Threads wait at the barrier until all `num_threads` have arrived. Once all
    threads have arrived, they are all released, and the barrier resets for
    subsequent use.
    """
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                before any are released.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 
 
    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other threads
        (up to `num_threads`) have also called `wait()`.

        When the last thread arrives, all waiting threads are notified and released.
        The barrier then resets for subsequent use.
        """
        self.cond.acquire()                      # Acquire the condition variable's lock.
        self.count_threads -= 1;                 # Decrement the count of threads yet to reach the barrier.
        # Pre-condition: Check if this is the last thread to reach the barrier.
        if self.count_threads == 0:
            self.cond.notify_all()               # Notify all waiting threads.
            self.count_threads = self.num_threads    # Reset the barrier for reuse.
        else:
            self.cond.wait();                    # Wait until all other threads arrive.
        self.cond.release();                   # Release the condition variable's lock.




class DeviceThread_Worker(Thread):
    """
    Represents a worker thread dedicated to processing a subset of scripts
    for a specific device.

    These workers are typically spawned by a `DeviceThread` to parallelize
    script execution and data processing, interacting with the device's
    sensor data and its neighbors.
    """
    def __init__(self, device, neighbours, tid, scripts):
        """
        Initializes a DeviceThread_Worker.

        Args:
            device (Device): The Device instance that owns this worker.
            neighbours (list): A list of neighboring Device instances.
            tid (int): The thread ID of this worker.
            scripts (list): A subset of scripts for this worker to process.
        """
        Thread.__init__(self)
        self.neighbours = neighbours
        self.device = device
        self.scripts = scripts
        self.tid = tid 

    def run(self):
        """
        Executes the main logic of the worker thread.

        This method iterates through the assigned scripts, gathers relevant data
        from the device and its neighbors, processes the data using the script,
        and updates the device's and neighbors' sensor data. It ensures thread-safe
        access to shared data by acquiring and releasing locks.
        """
        
        # Iterate through each script assigned to this worker.
        # Invariant: Each (script, location) pair represents a task to be processed.
        for (script, location) in self.scripts:

            
            script_data = []
            index = location

            # Block Logic: Gather data from neighboring devices at the specified location.
            # It acquires and releases locks for both the specific location and the device's
            # general lock to ensure thread-safe data access.
            for device in self.neighbours:

                self.device.locks[index].acquire()
                self.device.lock.acquire()

                data = device.get_data(location)

                self.device.lock.release()
                self.device.locks[index].release()

                if data is not None:
                    script_data.append(data)

            
            # Block Logic: Gather data from the current device's own sensor at the specified location.
            # It also acquires and releases locks to ensure thread-safe data access.
            self.device.locks[index].acquire()
            self.device.lock.acquire()

            data = self.device.get_data(location)

            self.device.lock.release()
            self.device.locks[index].release()

            if data is not None:
                script_data.append(data)

            
            # Pre-condition: Check if any data was collected to process.
            if script_data != []:
                result = script.run(script_data)

                self.device.locks[index].acquire()

                # Block Logic: Update the sensor data of neighboring devices with the script's result.
                # Only updates if the new result is greater than the existing data.
                for dev in self.neighbours:
                    if result > dev.get_data(location):
                        dev.set_data(location, result)
                    
                # Update the current device's own sensor data with the script's result.
                self.device.set_data(location, result)

                self.device.locks[index].release()


class Device(object):
    """
    Represents a single device in the simulated distributed system.

    Each device has a unique ID, sensor data, and interacts with a supervisor.
    It manages scripts for processing, coordinates with other devices through
    a barrier, and uses worker threads for parallel execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor data for various locations.
            supervisor (object): The supervisor object responsible for coordinating devices.
        """
        self.lock = RLock()
        self.barrier = None
        self.devices = []
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.locations = []
        self.locks = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources and synchronization mechanisms for a collection of devices.

        This static-like method initializes a reusable barrier for all devices
        and creates a global set of location-specific locks that all devices will share.

        Args:
            devices (list): A list of Device instances to be set up.
        """
        
        # Store the list of all devices.
        self.devices = devices
        # Pre-condition: This block runs only for the device with device_id 0.
        # This is typically used for centralized initialization of shared resources.
        if self.device_id == 0:
            # Block Logic: Initializes a set of RLock objects for locations.
            # Invariant: Each lock corresponds to a potential location for sensor data.
            for num in range(0, 1000): # Assuming a maximum of 1000 locations.
                lock = RLock()
                # Assign the newly created lock to all devices for the current 'location'.
                for i in range (0, len(devices)):
                    devices[i].locks.append(lock)
            
            # Create a reusable barrier for all devices to synchronize at specific timepoints.
            barrier = ReusableBarrier(len(devices)) 
            # Assign the created barrier to each device if it hasn't been assigned yet.
            for i in range(0,len(devices)):
                if devices[i].barrier == None:
                    devices[i].barrier = barrier


    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific location.

        If a script is provided, it's added to the device's script queue.
        If `script` is None, it signals that all scripts for the current
        timepoint have been assigned, and the device thread should proceed.

        Args:
            script (object): The script object to assign, or None to signal
                             timepoint completion.
            location (str): The location associated with the script.
        """
        # Pre-condition: Check if a script is actually being assigned.
        if script is not None:
            self.scripts.append((script, location)) 
            self.timepoint_done.set() # Signal that a script has been assigned, even if not all for timepoint.
            
        else:
            # If script is None, it indicates that all scripts for the current
            # timepoint have been assigned.
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            any: The sensor data for the specified location, or None if the
                 location is not found in the device's sensor data.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        This method acquires a lock to ensure thread-safe updates to the
        device's sensor data.

        Args:
            location (str): The location for which to set data.
            data (any): The new sensor data to be set.
        """
        self.lock.acquire() # Acquire lock to ensure thread-safe update of sensor_data.
        # Pre-condition: Check if the location exists in the sensor data.
        if location in self.sensor_data:
            self.sensor_data[location] = data # Update the sensor data.
        self.lock.release() # Release the lock.
    

    def shutdown(self):
        """
        Shuts down the device's main thread.

        This method blocks until the device's associated `DeviceThread` has
        completed its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the primary execution logic for a Device in a separate thread.

    This thread is responsible for coordinating with a supervisor, dividing
    scripts into tasks for worker threads, and synchronizing with other
    `DeviceThread` instances using a barrier.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def divide_in_threads(self, neighbours):
        """
        Divides the assigned scripts into sub-tasks and distributes them among worker threads.

        This method creates and manages `DeviceThread_Worker` instances to process
        scripts in parallel, improving performance for complex workloads.

        Args:
            neighbours (list): A list of neighboring Device instances.
        """

        
        threads = [] # List to hold the worker thread objects.

        # Calculate the number of worker threads to use and the distribution of scripts.
        nr = len(self.device.scripts) # Total number of scripts.
        numar = 1 # Number of scripts per worker (default 1).
        # Pre-condition: Check if the number of scripts exceeds a threshold (e.g., 8)
        # to determine if more workers are needed.
        if nr > 8:
            numar = nr / 8 # Calculate scripts per worker.
            nr = 8 # Limit to a maximum of 8 worker threads.

        # Block Logic: Create and initialize worker threads.
        # Invariant: Each worker thread is assigned a portion of the scripts to process.
        for i in range(0,nr):
            # Pre-condition: Check if it's the last worker to assign remaining scripts.
            if i == nr - 1:
                # Assign all remaining scripts to the last worker.
                t = DeviceThread_Worker(self.device, neighbours, i, self.device.scripts[i * numar : len(self.device.scripts)])
            else:
                # Assign a fixed number of scripts to other workers.
                t = DeviceThread_Worker(self.device, neighbours, i, self.device.scripts[i * numar : i*numar + numar])
            threads.append(t) # Add the worker thread to the list.

        # Block Logic: Start all worker threads.
        # Invariant: All worker threads begin parallel execution of their assigned scripts.
        for i in range(0, nr):
            threads[i].start()

        # Block Logic: Wait for all worker threads to complete their execution.
        # Invariant: The main device thread pauses until all its spawned workers have finished.
        for i in range(0,nr):
            threads[i].join()

    def run(self):
        """
        Executes the main logic of the device thread.

        This method continuously coordinates with the supervisor, processes
        assigned scripts using worker threads, and synchronizes with other
        device threads at a barrier. It handles shutdown conditions.
        """
        # Main loop for the device thread, runs indefinitely until shutdown.
        while True:
            # Retrieve information about neighboring devices from the supervisor.
            # This is done at the beginning of each cycle to get up-to-date neighbor information.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: Check if the device has no neighbors, indicating a shutdown scenario.
            if neighbours is None:
                break # Exit the main loop, effectively shutting down the device thread.

            # Wait until a new script is assigned or a timepoint done signal is received.
            self.device.script_received.wait()

            # Block Logic: Divide the assigned scripts into sub-tasks and distribute them
            # among worker threads for parallel processing.
            self.divide_in_threads(neighbours)

            # Clear the 'script_received' event for the next timepoint.
            self.device.script_received.clear()

            # Synchronize all device threads, ensuring all devices have completed their script processing
            # for the current timepoint before proceeding to the next.
            self.device.barrier.wait()
