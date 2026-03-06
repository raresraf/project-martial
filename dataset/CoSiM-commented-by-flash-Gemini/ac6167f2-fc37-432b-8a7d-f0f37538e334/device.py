


"""
This module implements a device simulation framework that utilizes multiple threads
and a reusable barrier for synchronization. It defines:
- ReusableBarrier: A re-usable barrier mechanism for threads.
- Device: Represents a simulated device with sensors, scripts, and multi-threaded processing.
- DeviceThread: The main thread for a Device, orchestrating job assignment and synchronization.
- DeviceSubThread: Worker threads responsible for executing individual scripts.
"""


from threading import Event, Thread, Lock, Semaphore
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
        self.count_lock = Lock()       # Protects access to the count_threads variables
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase

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
        
        with self.count_lock: # Block Logic: Ensure atomic access to the counter.
            count_threads[0] -= 1
            # Block Logic: If this is the last thread in the current phase, release all waiting threads.
            if count_threads[0] == 0:
                for _ in range(self.num_threads): # Block Logic: Release each waiting thread.
                    threads_sem.release()
                count_threads[0] = self.num_threads # Reset counter for next use
        threads_sem.acquire() # Block Logic: Wait for all threads to reach this point.

class Device(object):
    """
    Represents a simulated device in a distributed system. Each device
    manages its sensor data, executes scripts, and interacts with a supervisor.
    It uses a main device thread and multiple sub-threads for concurrent operations.
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
        self.scripts_received = Event() # Event to signal when new scripts are assigned

        self.all_devices = []           # List of all devices in the simulation
        self.scripts = []               # List to store assigned scripts (script, location) tuples
        self.data_lock = Lock()         # Lock to protect access to sensor_data
        self.thread = DeviceThread(self) # The main thread for this device
        self.thread.start()             # Start the main device thread
        self.barrier = ReusableBarrier(0) # Placeholder for the global barrier
        self.location_locks = {}        # Dictionary to store locks for different locations

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
        This includes initializing the global barrier and location-specific locks
        (done by the device with ID 0) and propagating these to other devices.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        # Block Logic: Device with ID 0 acts as the master to set up global synchronization.
        if self.device_id == 0:
            # Block Logic: Initialize locks for a predefined number of locations.
            for location in xrange(100):
                self.location_locks[location] = Lock()
            self.barrier = ReusableBarrier(len(devices)) # Initialize global barrier for all devices

        # Block Logic: Propagate the initialized global barrier and location locks to all other devices.
        for dev in devices:
            # Inline: Only the master device (device_id == 0) propagates the barrier and locks.
            if self.device_id == 0:
                dev.barrier = self.barrier
                dev.location_locks = self.location_locks

            self.all_devices.append(dev)

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific location.
        If a script is provided, it's added to the device's script list.
        If no script is provided (None), it signals that scripts have been received for the timepoint.

        Args:
            script (Script or None): The script object to assign, or None to signal script reception.
            location (str): The location associated with the script.
        """
        
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list
        else:
            # Block Logic: Signal that all scripts for this timepoint have been assigned.
            self.scripts_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            Any or None: The sensor data if available for the location, otherwise None.
        """
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.
        Ensures thread-safe access to sensor_data using a lock.

        Args:
            location (str): The location for which to set data.
            data (Any): The new sensor data to set.
        """
        
        # Block Logic: Acquire data_lock to ensure exclusive access to sensor_data.
        self.data_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.data_lock.release() # Block Logic: Release data_lock after modifying sensor_data.

    def shutdown(self):
        """
        Shuts down the device by joining its main device thread.
        """
        
        self.thread.join()

class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for orchestrating the overall
    simulation workflow for that device. It fetches neighbor information,
    distributes script execution to DeviceSubThreads, and synchronizes
    with a global barrier.
    """
    

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent Device instance this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.assigned_scripts = {} # Dictionary to store scripts assigned to this thread (not used directly in current run)

    def run(self):
        """
        Executes the main logic of the DeviceThread.
        - Continuously fetches neighbor information from the supervisor.
        - If no neighbors are returned (signal for shutdown), the loop breaks.
        - Waits for scripts to be assigned for the current timepoint.
        - Creates and starts DeviceSubThreads to execute each assigned script concurrently.
        - Waits for all DeviceSubThreads to complete.
        - Synchronizes with the global barrier after all scripts are processed.
        """
        # Block Logic: Main loop for continuous processing of timepoints.
        while True:
            # Block Logic: Fetch updated neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: If no neighbors are returned, it's a shutdown signal.
            if neighbours is None:
                break # Exit the main loop

            # Block Logic: Wait for scripts to be assigned for the current timepoint, then clear the event.
            self.device.scripts_received.wait()
            self.device.scripts_received.clear()

            # Block Logic: Create a DeviceSubThread for each script to execute it concurrently.
            device_threads = []
            for (script, location) in self.device.scripts:
                device_threads.append(
                    DeviceSubThread(self.device, script, location, neighbours)
                )

            # Block Logic: Start all DeviceSubThreads.
            for i in xrange(len(device_threads)):
                device_threads[i].start()

            # Block Logic: Wait for all DeviceSubThreads to complete their execution.
            for i in xrange(len(device_threads)):
                device_threads[i].join()

            # Block Logic: Synchronize all devices at the global barrier after all scripts are processed.
            self.device.barrier.wait()

class DeviceSubThread(Thread):
    """
    A worker thread responsible for executing a single script within a Device.
    It retrieves data, runs the assigned script, and updates relevant devices.
    """
    

    def __init__(self, device, script, location, neighbours):
        """
        Initializes a DeviceSubThread.

        Args:
            device (Device): The parent Device instance this sub-thread belongs to.
            script (Script): The script object to be executed.
            location (str): The location associated with the script.
            neighbours (list): A list of neighboring Device instances relevant to this script.
        """
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the logic of the DeviceSubThread.
        - Acquires a lock for the script's location.
        - Collects data from neighboring devices and the parent device.
        - Executes the assigned script if data is available.
        - Updates data on neighboring devices and the parent device with the script's result.
        - Releases the lock for the script's location.
        """
        
        # Block Logic: Acquire a lock for the specific location to prevent race conditions during data access.
        self.device.location_locks[self.location].acquire()
        script_data = []
        # Block Logic: Collect data from neighboring devices specified for this script.
        # Invariant: Each `device` is a neighbor from which data should be collected.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Block Logic: Collect data from the current device itself for the script's location.
        data = self.device.get_data(self.location)

        if data is not None:
            script_data.append(data)

        # Block Logic: If data was collected, execute the script and update devices.
        if script_data != []:
            # Inline: Execute the assigned script with collected data.
            result = self.script.run(script_data)

            # Block Logic: Propagate the script's result to neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)

            # Block Logic: Update the current device's sensor data with the script's result.
            self.device.set_data(self.location, result)

        # Block Logic: Release the lock for the current location.
        self.device.location_locks[self.location].release()
