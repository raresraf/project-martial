


from threading import Thread, Lock, Semaphore, Event


class ReusableBarrier(object):
    """
    Implements a reusable barrier synchronization mechanism for a fixed number of threads.
    This barrier ensures that all participating threads wait at a specific point until
    every thread has reached that point before any can proceed. It utilizes two phases
    of semaphores to allow for multiple synchronization cycles without reinitialization.
    """
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
        self.phase(self.count_threads1, self.threads_sem1)
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
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase of the barrier.
                count_threads[0] = self.num_threads
        # Each thread acquires the semaphore, blocking until all threads have released it.
        threads_sem.acquire()

class Device(object):
    """
    Represents a single device in a simulated distributed system.
    Each device manages its own sensor data, interacts with a supervisor,
    executes assigned scripts, and coordinates its operations through
    a dedicated thread and shared synchronization primitives.
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
        self.scripts = []
        # Event to signal that the device has completed processing for the current timepoint.
        self.timepoint_done = Event()
        # The dedicated thread for this device's execution logic.
        self.thread = DeviceThread(self)
        # Global barrier for synchronizing all devices. Set by setup_devices.
        self.barr = None
        # Global lock for critical sections across devices. Set by setup_devices.
        self.lock = None
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures a shared ReusableBarrier and a global Lock among a group of devices.
        This method ensures that all devices have access to the same synchronization
        primitives for timepoint progression and thread-safe operations.
        The device designated as 'root_node' (device at index 0 in the list)
        is responsible for initializing these shared resources if they haven't been already.

        Args:
            devices (list): A list of all Device objects participating in the simulation.
        """
        root_node = 0 # Designates the first device in the list as the potential initializer.
        
        # Check if the global barrier has not been initialized yet.
        if devices[root_node].barr is None:
            # Only the designated root node device performs the initialization.
            if devices[root_node].device_id == self.device_id:
                # Create a new ReusableBarrier for all participating devices.
                barr = ReusableBarrier(len(devices))
                # Create a single global lock to be shared among all devices.
                lock = Lock()
                # Assign the created barrier and lock to all devices.
                for i in  devices:
                    i.barr = barr
                    i.lock = lock

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.
        Signals the script reception or timepoint completion.

        Args:
            script (Script): The script object to be executed. If None, it signals
                             the end of scripts for the current timepoint.
            location: The data location pertinent to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a script has been received.
        else:
            self.timepoint_done.set() # Signal that all scripts for this timepoint have been assigned.

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
    including supervisor interaction, sequential script execution,
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
        It continuously processes timepoints, executes scripts sequentially
        under a global lock, and synchronizes with other devices.
        """
        while True:
            # Retrieve current neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If no neighbors are returned, it signifies the end of the simulation.
            if neighbours is None:
                break

            # Wait until scripts for the current timepoint have been assigned.
            # This Event is set by the assign_script method when script is None.
            self.device.timepoint_done.wait()

            # Iterate through and execute all assigned scripts for the current timepoint.
            # Note: All script execution is protected by a single global lock,
            # which serializes operations across devices and locations, potentially
            # becoming a bottleneck in highly concurrent scenarios.
            for (script, location) in self.device.scripts:
                # Acquire the global lock to ensure exclusive access during data processing.
                self.device.lock.acquire()
                script_data = []
                
                # Collect data from neighboring devices for the current location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Collect data from the local device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execute the script if there is data to process.
                if script_data != []:
                    # Execute the script with the aggregated data.
                    result = script.run(script_data)

                    # Propagate the script's result back to all neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Update the local device's data with the script's result.
                    self.device.set_data(location, result)
                self.device.lock.release() # Release the global lock.
            
            # Reset the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()
            # Synchronize with all other devices at the global barrier before starting the next timepoint.
            self.device.barr.wait()
