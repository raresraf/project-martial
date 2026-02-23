"""
@3722b56a-c47e-4bf2-b452-4ea7f5549ce0/device.py
@brief Implements a distributed device simulation framework, including a reusable barrier for synchronization and device-specific functionalities for sensor data processing and script execution.
* Algorithm: Thread-based simulation with barrier synchronization.
* Concurrency: Uses `threading.Semaphore` and `threading.Lock` for inter-thread synchronization and mutual exclusion.
"""

from threading import Event, Thread
from threading import Semaphore, Lock

class ReusableBarrier(object):
    """
    @brief A re-usable barrier mechanism for synchronizing multiple threads.
    This barrier allows a set number of threads to wait for each other at a synchronization
    point, and once all threads have arrived, they are all released simultaneously.
    It can be used multiple times after all threads have passed through.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier with a specified number of threads.
        @param num_threads: The total number of threads that must reach the barrier
                            before any are released.
        """
        self.num_threads = num_threads
        # Two counters for a double-phased barrier to allow reusability
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]

        # Lock to protect access to the thread counters
        self.count_lock = Lock()

        # Semaphores to block and release threads in each phase
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier.
        The thread will be blocked until all `num_threads` have called this method.
        This method uses a double-phase approach to ensure reusability.
        """
        # Phase 1: Wait for all threads to arrive
        self.phase(self.count_threads1, self.threads_sem1)
        # Phase 2: Reset the barrier and wait for all threads to clear
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements a single phase of the barrier synchronization.
        @param count_threads: A list containing the current count of threads for this phase.
                              (Using a list to allow modification within the `with` block).
        @param threads_sem: The semaphore associated with this phase to release waiting threads.
        """
        # Pre-condition: `count_threads[0]` reflects the number of threads yet to reach this phase.
        with self.count_lock:
            # Decrement the count of threads waiting at this phase.
            count_threads[0] -= 1
            # Invariant: If `count_threads[0]` reaches 0, all threads have arrived.
            if count_threads[0] == 0:
                # All threads have arrived, release them by incrementing the semaphore `num_threads` times.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        
        # Block the current thread until it's released by the last thread to enter the phase.
        threads_sem.acquire()


class Device(object):
    """
    @brief Represents a simulated device in a distributed system.
    Each device can hold sensor data, interact with a supervisor,
    and execute scripts based on neighbor data.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor readings relevant to this device.
        @param supervisor: A reference to the supervisor object managing the devices.
        """
        self.barrier = None # Initialized by supervisor for synchronization
        self.InitializationEvent = Event() # Used to signal device initialization completion
        self.LockLocation = None # Dictionary to store locks for specific data locations, managed by device_id == 0
        self.LockDict = Lock() # Lock to protect access to LockLocation dictionary

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Signals when a script has been assigned
        self.scripts = [] # List to store (script, location) tuples
        self.timepoint_done = Event() # Signals when a device has completed its operations for a timepoint
        self.thread = DeviceThread(self) # The dedicated thread for this device

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the synchronization mechanisms and inter-device communication.
        This method is called once during initialization. Only device 0 acts as the coordinator.
        @param devices: A list of all Device objects in the simulation.
        """
        # Pre-condition: This block executes only for device with ID 0 to coordinate setup.
        if self.device_id == 0:
            # Invariant: Device 0 initializes the shared barrier and LockLocation dictionary.
            n = len(devices)
            self.barrier = ReusableBarrier(n)   # Initialize a barrier for all devices
            self.LockLocation = {}  # Initialize shared lock storage for data locations

            # Iterate through all devices to set up shared resources.
            for idx in range(len(devices)):
                d = devices[idx]

                # Assign shared LockLocation and barrier to all devices.
                d.LockLocation = self.LockLocation
                d.barrier = self.barrier
                # If not device 0, signal their initialization completion.
                if d.device_id == 0:
                    pass
                else:
                    d.InitializationEvent.set()
        else:
            # For non-coordinator devices, wait until device 0 has completed shared resource initialization.
            self.InitializationEvent.wait()

        # Start the dedicated thread for this device.
        self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific data location.
        @param script: The script object to be executed.
        @param location: The data location relevant to the script.
        """
        # Pre-condition: A script is provided, or None to signal end of timepoint.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Invariant: If script is None, it means the current timepoint's script assignments are done.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key for the sensor data.
        @return: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.
        @param location: The key for the sensor data.
        @param data: The new data to be set.
        """
        # Pre-condition: `location` must exist in `sensor_data` to be updated.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's associated thread.
        """
        # Wait for the device's thread to complete its execution.
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief A thread dedicated to a Device, responsible for its operational logic,
    including synchronization, data collection from neighbors, script execution,
    and data dissemination.
    """
    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device: The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the device thread.
        Handles barrier synchronization, retrieves neighbor data, executes scripts,
        and updates shared data locations.
        """
        # Invariant: Loop continuously until the simulation signals termination.
        while True:
            # Synchronize all device threads at the barrier before proceeding.
            self.device.barrier.wait()

            # Retrieve neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If `neighbours` is None, it signals the end of the simulation.
            if neighbours is None:
                break # Exit the loop and terminate the thread.

            # Wait until all scripts for the current timepoint have been assigned to this device.
            self.device.timepoint_done.wait()

            dev_scripts = self.device.scripts

            # Process each assigned script.
            # Invariant: Each script is executed in isolation with proper locking of data locations.
            for (script, location) in self.device.scripts:
                # Acquire a lock for the LockLocation dictionary to safely check/add locks.
                self.device.LockDict.acquire()

                # If no lock exists for this data `location`, create one.
                if location not in self.device.LockLocation.keys():
                    self.device.LockLocation[location] = Lock()

                # Acquire the specific lock for the current data `location` to ensure exclusive access.
                self.device.LockLocation[location].acquire()

                # Release the LockDict lock as the LockLocation dictionary is no longer being modified.
                self.device.LockDict.release()

                script_data = []
                # Collect data from neighboring devices for the current `location`.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Collect data from the current device itself for the current `location`.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: Execute the script only if relevant data was collected.
                if script_data != []:
                    # Execute the script with the collected data.
                    result = script.run(script_data)

                    # Disseminate the script result to neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Update the current device's data with the script result.
                    self.device.set_data(location, result)

                # Release the lock for the current data `location`.
                self.device.LockLocation[location].release()

            # Reset the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()