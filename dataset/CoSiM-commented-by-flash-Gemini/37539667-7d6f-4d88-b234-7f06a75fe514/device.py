"""
@37539667-7d6f-4d88-b234-7f06a75fe514/device.py
@brief Implements a distributed device simulation framework with thread-level parallelism for script execution.
This variant utilizes a reusable barrier for synchronization and introduces a new thread type (`MyThread`)
to handle script execution concurrently for different locations, enhancing the parallel processing capabilities.
* Algorithm: Thread-based simulation with barrier synchronization and fine-grained parallel script execution.
* Concurrency: Uses `threading.Semaphore`, `threading.Lock`, `threading.Event`, `threading.Thread` for
               inter-thread synchronization, mutual exclusion, and task coordination.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier():
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
        self.count_lock = Lock()                 # Lock to protect access to the thread counters.
        self.threads_sem1 = Semaphore(0)         # Semaphore for the first phase.
        self.threads_sem2 = Semaphore(0)         # Semaphore for the second phase.

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier.
        The thread will be blocked until all `num_threads` have called this method.
        This method uses a double-phase approach to ensure reusability.
        """
        # Phase 1: Wait for all threads to arrive.
        self.phase(self.count_threads1, self.threads_sem1)
        # Phase 2: Reset the barrier and wait for all threads to clear.
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
    @brief Represents a simulated device in a distributed system, capable of processing sensor data
    and executing scripts, with enhanced concurrency for script execution.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor readings relevant to this device.
        @param supervisor: A reference to the supervisor object managing the devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Signals when a script has been assigned.
        self.scripts = [] # List to store (script, location) tuples.
        self.timepoint_done = Event() # Signals when a device has completed its operations for a timepoint.

        self.thread = DeviceThread(self) # The dedicated thread for this device.
        self.thread.start() # Start the device's main thread immediately.
        self.devices = [] # List of all devices, set up by the supervisor.
        self.reusable_barrier = None # Shared barrier for device synchronization.
        self.thread_list = [] # List to hold MyThread instances for concurrent script execution.
        self.location_lock = [None] * 99 # Array to store locks for specific data locations (index by location ID).

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the synchronization mechanisms and inter-device communication.
        This method ensures a shared barrier is initialized and distributed among devices.
        @param devices: A list of all Device objects in the simulation.
        """
        # Pre-condition: If the reusable_barrier is not yet initialized for this device.
        if self.reusable_barrier is None:
            # Invariant: Create a new barrier if one doesn't exist, and assign it to all devices.
            barrier = ReusableBarrier(len(devices))
            self.reusable_barrier = barrier
            for device in devices:
                if device.reusable_barrier is None:
                    device.reusable_barrier = barrier

        # Populate the device's internal list of all devices.
        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific data location.
        Also initializes a lock for the location if not already present, ensuring thread-safe access.
        @param script: The script object to be executed.
        @param location: The data location relevant to the script.
        """
        is_location_locked = 0
        # Pre-condition: A script is provided, or None to signal end of timepoint.
        if script is not None:
            self.scripts.append((script, location))
            # Block Logic: Check if a lock for this location already exists; if not, create one.
            if self.location_lock[location] is None:
                # Invariant: Search for an existing lock for this location among other devices.
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device .location_lock[location]
                        is_location_locked = 1
                        break
                # If no existing lock was found, create a new one.
                if is_location_locked == 0:
                    self.location_lock[location] = Lock()
            self.script_received.set() # Signal that a script has been received.

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
        # Wait for the device's main thread to complete its execution.
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief A thread dedicated to a Device, responsible for its operational logic,
    including synchronizing with other devices, managing script execution in parallel,
    and handling data updates.
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
        It handles fetching neighbor data, spawning `MyThread`s for concurrent script execution
        per location, and synchronizing at the barrier.
        """
        # Invariant: Loop continuously until the simulation signals termination.
        while True:
            # Retrieve neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If `neighbours` is None, it signals the end of the simulation.
            if neighbours is None:
                break # Exit the loop and terminate the thread.

            # Wait until all scripts for the current timepoint have been assigned to this device.
            self.device.timepoint_done.wait()

            # Block Logic: For each assigned script, create and start a new `MyThread` for concurrent execution.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, location, script, neighbours)
                self.device.thread_list.append(thread)

            # Invariant: Start all `MyThread`s concurrently.
            for thread in self.device.thread_list:
                thread.start()

            # Invariant: Wait for all `MyThread`s to complete their execution.
            for thread in self.device.thread_list:
                thread.join()

            self.device.thread_list = [] # Clear the list of threads for the next timepoint.

            # Reset the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()
            # Synchronize all device threads at the barrier before proceeding to the next timepoint.
            self.device.reusable_barrier.wait()

class MyThread(Thread):
    """
    @brief A specialized thread responsible for executing a single script for a specific data location.
    It acquires a lock for the location, collects data from neighbors and itself, runs the script,
    and then disseminates the results before releasing the lock.
    """
    def __init__(self, device, location, script, neighbours):
        """
        @brief Initializes MyThread.
        @param device: The parent Device object.
        @param location: The data location pertinent to this script execution.
        @param script: The script object to be executed.
        @param neighbours: A list of neighboring devices to collect data from.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for MyThread.
        Acquires location lock, performs data collection, script execution, and data dissemination.
        """
        # Pre-condition: Acquire a lock for the specific data `location` to ensure exclusive access.
        self.device.location_lock[self.location].acquire()
        script_data = []
        # Block Logic: Collect data from neighboring devices for the current `location`.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Collect data from the current device itself for the current `location`.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Pre-condition: Execute the script only if relevant data was collected.
        if script_data != []:
            # Execute the script with the collected data.
            result = self.script.run(script_data)

            # Disseminate the script result to neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Update the current device's data with the script result.
            self.device.set_data(self.location, result)

        # Release the lock for the current data `location`, allowing other threads to access it.
        self.device.location_lock[self.location].release()