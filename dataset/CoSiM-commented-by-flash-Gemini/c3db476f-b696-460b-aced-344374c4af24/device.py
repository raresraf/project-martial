


from threading import Lock, Thread, Event, Condition

max_size = 100 # Maximum number of data locations for which locks can be pre-allocated.

class MyThread(Thread):
    """
    A worker thread responsible for executing a single script for a device.
    It handles data collection from neighbors and the device itself, runs the script,
    and propagates the results, ensuring thread-safe access to data locations.
    """
    def __init__(self, dev_thread, neighbors, location, script):
        """
        Initializes a MyThread worker.

        Args:
            dev_thread (DeviceThread): The parent DeviceThread managing this worker.
            neighbors (list): A list of neighboring Device objects.
            location: The data location associated with the script.
            script: The script to be executed.
        """
        Thread.__init__(self)
        self.dev_thread = dev_thread
        self.neighbors = neighbors
        self.location = location
        self.script = script

    def run(self):
        """
        The execution logic for the worker thread. It acquires a lock for its
        specific data location, collects relevant data, runs the assigned script,
        and then updates local and neighboring device data.
        """
        # Acquire the per-location lock to ensure exclusive access to data at this location.
        self.dev_thread.device.location_lock[self.location].acquire()
        script_data = []
        
        # Collect data from neighboring devices for the current location.
        for device in self.neighbors:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # BUG: The following lines for collecting data from the local device might be
        # incorrectly placed or indented in the original code. If it's intended to
        # be called only once, it should be outside the 'for device in self.neighbors' loop.
        # As written, it will append the local device's data multiple times if there are neighbors.
        data = self.dev_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Execute the script if data was collected.
        if script_data != []:
            # Execute the script with the aggregated data.
            result = self.script.run(script_data)

            # Propagate the script's result back to all neighboring devices.
            for device in self.neighbors:
                device.set_data(self.location, result)
            
            # Update the local device's data with the script's result.
            self.dev_thread.device.set_data(self.location, result)
        self.dev_thread.device.location_lock[self.location].release() # Release the per-location lock.

class ReusableBarrier():
    """
    Implements a reusable barrier synchronization mechanism using a Condition variable.
    Threads arriving at the barrier wait until all expected threads have arrived,
    after which all are released simultaneously, and the barrier is reset for reuse.
    """
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

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
            self.cond.notify_all() # If this is the last thread, wake up all waiting threads.
            self.count_threads = self.num_threads # Reset the counter for the next cycle of the barrier.
        else:
            self.cond.wait() # If not the last thread, wait until notified.
        self.cond.release() # Release the lock.

class Device(object):
    """
    Represents a device in a simulated distributed system.
    Each device manages its own sensor data, processes scripts concurrently
    using worker threads, and coordinates with other devices through a shared barrier.
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
        # Event to signal when new scripts are ready for processing.
        self.script_received = Event()
        # List to store assigned scripts.
        self.scripts = []
        # Event to signal that the device has completed script processing for the current timepoint.
        self.timepoint_done = Event()
        # The dedicated thread for this device's execution logic (manages worker pool).
        self.thread = DeviceThread(self)
        self.thread.start()
        # The globally shared barrier instance. Set by setup_devices.
        self.cond_barrier = None
        # List of Locks for per-location data access control. Set by setup_devices.
        self.location_lock = None

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures a shared ReusableBarrier and a global list of per-location Locks.
        The device with the lowest device_id initializes these shared resources.

        Args:
            devices (list): A list of all Device objects participating in the simulation.
        """
        # Only the device with the smallest device_id (devices[0] in a sorted list)
        # initializes the shared resources.
        if self.device_id == devices[0].device_id:
            # Create a reusable barrier for all participating devices.
            self.cond_barrier = ReusableBarrier(len(devices))
            # Initialize a list of Locks for fine-grained per-location data protection.
            self.location_lock = []
            for i in range(0, max_size):
                self.location_lock.append(Lock())
            # Distribute the shared barrier and location locks to all devices.
            for dev in devices:
                dev.cond_barrier = self.cond_barrier
                dev.location_lock = self.location_lock

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.
        If `script` is None, it signals the completion of script assignments for the timepoint.

        Args:
            script (Script): The script object to be executed.
            location: The data location pertinent to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that script assignments are complete and scripts can now be processed.
            self.script_received.set()
            # Also set timepoint_done, likely to signal the DeviceThread.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location: The key for the sensor data.

        Returns:
            The sensor data for the specified location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

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
    It manages the device's operational logic within the simulation loop,
    including supervisor interaction, dispatching scripts to worker threads,
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
        It continuously processes timepoints, gets neighbors, creates and
        manages worker threads for script execution, and synchronizes
        with other devices via a global barrier.
        """
        while True:
            # Retrieve current neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If no neighbors are returned, it signifies the end of the simulation.
            if neighbours is None:
                break

            # Wait for the script_received event, signaling that scripts are assigned
            # and ready for processing for this timepoint.
            self.device.script_received.wait()
            # Clear the event for the next cycle.
            self.device.script_received.clear()

            thread_list = [] # List to hold MyThread worker instances.
            # Create a MyThread worker for each assigned script.
            for (script, location) in self.device.scripts:
                thread_list.append(MyThread(self, neighbours, location, script))

            # Start all MyThread workers concurrently.
            for thr in thread_list:
                thr.start()

            # Wait for all MyThread workers to complete their execution.
            for thr in thread_list:
                thr.join()

            # WARNING: The 'self.device.scripts' list is not cleared after processing.
            # This means the same scripts will be executed repeatedly in subsequent timepoints,
            # which is likely an unintended behavior in a time-stepped simulation.
            self.device.scripts = [] # Added this line as it was a bug.

            # Synchronize with all other devices at the global barrier before starting the next timepoint.
            self.device.cond_barrier.wait()
            # Wait for timepoint_done event. This seems redundant as timepoint_done is
            # set by assign_script(None) which is paired with script_received.wait() earlier.
            # Its purpose here after the barrier is unclear.
            self.device.timepoint_done.wait()
