


from threading import Event, Thread, Lock, Condition

class Device(object):
    """
    Represents a simulated device in a distributed system, capable of processing
    sensor data, executing scripts, and synchronizing with other devices
    through a supervisor.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing the device's sensor data,
                                typically keyed by location.
            supervisor (Supervisor): The central supervisor managing the devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been received by the device.
        self.script_received = Event()
        # List to store (script, location) tuples for execution.
        self.scripts = []
        # Event to signal that the device has completed processing for a timepoint.
        self.timepoint_done = Event()
        # The dedicated thread for this device's operational logic.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures shared resources (locks and a barrier) among a group of devices.
        This method ensures that all devices have access to the same synchronization
        primitives for concurrent data access and timepoint progression.

        Args:
            devices (list): A list of Device objects that will share the barrier and lock set.
        """
        # Dictionary to hold locks for each data location, ensuring thread-safe access.
        lock_set = {}
        # A reusable barrier for synchronizing all devices.
        barrier = ReusableBarrier(len(devices))
        idx = len(devices) - 1

        # Iterate through devices (in reverse order of ID) to assign shared resources.
        while idx >= 0:
            current_device = devices[idx]
            # Assign the shared barrier to the current device.
            current_device.barrier = barrier
            # Initialize a lock for each data location in the current device's sensor data.
            for current_location in current_device.sensor_data:
                lock_set[current_location] = Lock()
            # Assign the shared lock set to the current device.
            current_device.lock_set = lock_set
            idx = idx - 1

    def assign_script(self, script, location):
        """
        Assigns a script for the device to execute at a given location.

        Args:
            script (Script): The script object to be executed. If None, it signals
                             the completion of script assignments for the current timepoint.
            location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Signal that script assignment for the current timepoint is done.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location: The key corresponding to the desired sensor data.

        Returns:
            The sensor data for the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates the sensor data for a given location.

        Args:
            location: The key for the sensor data to be updated.
            data: The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
        else:
            pass # Data is not set if location does not exist in sensor_data.

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device by joining its execution thread.
        """
        self.thread.join()

class ReusableBarrier(object):
    """
    Implements a reusable barrier synchronization mechanism using a Condition variable.
    Threads arriving at the barrier wait until all expected threads have arrived,
    after which all are released simultaneously, and the barrier is reset for reuse.
    """
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier.

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

    def print_barrier(self):
        """
        (Development/Debug) Prints the current state of the barrier.
        """
        print self.num_threads, self.count_threads

class DeviceThread(Thread):
    """
    The main execution thread for a Device object.
    It manages the lifecycle of script execution, data updates,
    and synchronization with other devices within a time-stepped simulation.
    """
    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device object that this thread controls.
        """
        self.device = device
        # Initialize the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)

    def run(self):
        """
        The main loop of the device thread. It orchestrates script execution
        and synchronization in discrete timepoints.
        """
        # Number of worker threads to spawn for concurrent script execution.
        nr_threads = 8 
        # Main simulation loop.
        while True:
            # Reset the timepoint_done event for the current cycle.
            self.device.timepoint_done.clear()
            # Get the current set of neighboring devices from the supervisor.
            neigh = self.device.supervisor.get_neighbours()
            # Wait for all device threads to reach this point before proceeding.
            self.device.barrier.wait()
            # If there are no neighbors, it signals the end of the simulation.
            if neigh is None:
                break
            # Wait until scripts for the current timepoint are assigned and ready.
            self.device.timepoint_done.wait()
            
            # Prepare scripts for concurrent execution.
            execute_script = []
            threads = []
            for script in self.device.scripts:
                execute_script.append(script)
            
            # Spawn worker threads to execute scripts in parallel.
            for i in xrange(nr_threads):
                threads.append(MakeUpdate(self.device, neigh, execute_script))
                threads[i].start()

            # Wait for all worker threads to complete their script execution.
            for t in threads:
                t.join()
            # Synchronize with other device threads after script execution is complete
            # and before moving to the next timepoint.
            self.device.barrier.wait()

class MakeUpdate(Thread):
    """
    A worker thread responsible for executing a single script, collecting data
    from neighbors and the device itself, updating data, and ensuring thread safety
    through locking mechanisms.
    """
    def __init__(self, device, neighbours, execute_script):
        """
        Initializes a MakeUpdate thread.

        Args:
            device (Device): The local device object.
            neighbours (list): A list of neighboring Device objects.
            execute_script (list): A shared list of (script, location) tuples
                                   from which scripts are popped for execution.
        """
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.execute_script = execute_script

    def run(self):
        """
        The execution logic for a MakeUpdate thread. It attempts to pop and
        execute a script from the shared list, handling data collection,
        script execution, and data propagation, all while ensuring thread-safe
        access to data locations.
        """
        # Check if there are scripts available to execute.
        if len(self.execute_script) != 0:
            collected = []
            # Atomically get a script to execute. Pop removes it from the shared list.
            (script, location) = self.execute_script.pop()
            # Acquire a lock for the specific data location to prevent race conditions.
            self.device.lock_set[location].acquire()
            # Collect data from all neighbors for the given location.
            for neigh_c in self.neighbours:
                collected.append(neigh_c.get_data(location))
            # Collect data from the device itself for the given location.
            collected.append(self.device.get_data(location))

            # If data was collected, execute the script.
            if collected != []:
                # Execute the script with the aggregated data.
                result = script.run(collected)
                # Update the device's own data with the script's result.
                self.device.set_data(location, result)
                # Propagate the result to all neighboring devices.
                for neigh_c in self.neighbours:
                    neigh_c.set_data(location, result)
            self.device.lock_set[location].release() # Release the lock for the data location.
