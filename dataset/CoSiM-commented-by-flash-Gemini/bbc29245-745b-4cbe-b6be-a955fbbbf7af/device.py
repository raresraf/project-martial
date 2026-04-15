
"""
@bbc29245-745b-4cbe-b6be-a955fbbbf7af/device.py
@brief Defines the Device, ScriptThread, DeviceThread, and ReusableBarrier classes for managing distributed device operations.
This module provides the core components for a distributed simulation or data processing system.
The `Device` class represents an individual processing unit, holding sensor data and
managing scripts. `ScriptThread` executes individual scripts, `DeviceThread` orchestrates
timepoint processing and script distribution, and `ReusableBarrier` ensures synchronization
across multiple device threads.

Domain: Concurrency, Distributed Systems, Simulation, Data Processing.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.
    This barrier allows a fixed number of threads to wait for each other at a
    specific point in their execution, and then proceeds together. It can be
    reused multiple times after all threads have passed through.
    Algorithm: Double-counting semaphore-based barrier.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier for a specified number of threads.
        @param num_threads: The total number of threads that will participate in the barrier.
        Functional Utility: Sets up internal counters and semaphores required for barrier synchronization.
        """
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Inline: Counter for the first phase of the barrier.
        self.count_threads2 = [self.num_threads] # Inline: Counter for the second phase of the barrier.
        
        self.count_lock = Lock()                 # Inline: A lock to protect access to the thread counters.
        
        self.threads_sem1 = Semaphore(0)         # Inline: Semaphore for releasing threads in the first phase.
        
        self.threads_sem2 = Semaphore(0)         # Inline: Semaphore for releasing threads in the second phase.

    def wait(self):
        """
        @brief Blocks the calling thread until all other threads have also called wait.
        Functional Utility: Orchestrates the two-phase synchronization mechanism, ensuring
        all participating threads reach this point before any can proceed.
        """
        # Block Logic: Executes the first phase of the barrier.
        self.phase(self.count_threads1, self.threads_sem1)
        # Block Logic: Executes the second phase of the barrier.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements one phase of the barrier synchronization.
        @param count_threads: A list containing the current count of threads waiting for this phase.
        @param threads_sem: The semaphore used to release threads once the count reaches zero.
        Block Logic: Decrements the thread count. When the count reaches zero,
        all waiting threads are released, and the count is reset for the next use.
        Invariant: At the entry, threads are waiting for their turn in the phase.
        """
        
        with self.count_lock:
            count_threads[0] -= 1
            # Block Logic: Checks if this is the last thread to reach the barrier phase.
            if count_threads[0] == 0:
                # Block Logic: Releases all waiting threads from the semaphore and resets the counter.
                for _ in range(self.num_threads):
                    threads_sem.release()
                
                count_threads[0] = self.num_threads # Inline: Resets the counter for the next use of the barrier.
        # Functional Utility: Acquires the semaphore, blocking the thread until all threads have reached the barrier.
        threads_sem.acquire()


class Device(object):
    """
    @brief Represents a single device in a distributed system, managing its sensor data and scripts.
    This class encapsulates device-specific state, including its ID, sensor readings,
    and a list of scripts to be executed. It interacts with a supervisor for global
    context and uses a barrier and shared locks for synchronization with other devices.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: Unique identifier for the device.
        @param sensor_data: A dictionary containing sensor readings, indexed by location.
        @param supervisor: A reference to the supervisor object managing all devices.
        Functional Utility: Sets up the device's state, including synchronization primitives
        and a dedicated thread for timepoint management.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []                      # Inline: List to hold (script, location) tuples assigned to the device.
        self.timepoint_done = Event()          # Inline: Event to signal the completion of a timepoint's processing.
        self.barrier = None                    # Inline: Synchronization barrier for coordinating with other devices.
        self.thread = DeviceThread(self)       # Inline: The dedicated thread for this device's timepoint management.
        self.thread.start()                    # Functional Utility: Starts the device's dedicated timepoint management thread.
        self.location_locks = None             # Inline: List of shared locks for data locations, initialized by the master device.

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        Functional Utility: Provides a human-readable identifier for the device.
        """
        # Functional Utility: Formats the device ID into a descriptive string.
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the synchronization barrier and shared locks across multiple devices.
        @param devices: A list of all participating Device objects.
        Block Logic: If this is the master device (device_id == 0), it initializes a
        ReusableBarrier and a global list of locks for data locations. It then assigns these
        synchronization primitives to all participating devices.
        Pre-condition: Called once all Device objects have been instantiated.
        """
        # Block Logic: Checks if the current device is the master device (ID 0).
        if 0 == self.device_id:
            
            # Functional Utility: Initializes a ReusableBarrier with the total number of devices.
            self.barrier = ReusableBarrier(len(devices))
            
            locations = []
            # Block Logic: Gathers all unique data locations across all devices.
            for device in devices:
                for location in device.sensor_data:
                    if location not in locations:
                        locations.append(location)
            
            # Functional Utility: Initializes a list of locks, one for each unique data location.
            # This ensures that access to sensor data is thread-safe across all devices.
            self.location_locks = []
            for _ in range(len(locations)):
                self.location_locks.append(Lock())
            
            # Block Logic: Assigns the common barrier and global location locks to all participating devices.
            for device in devices:
                device.barrier = self.barrier
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location on the device.
        @param script: The script object to execute.
        @param location: The data location on which the script will operate.
        Block Logic: Appends the script and its target location to the device's script list.
        If no script is provided, it signals that the current timepoint is done.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Functional Utility: Signals that the current timepoint has completed all script assignments.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key identifying the sensor data to retrieve.
        Returns: The sensor data at the specified location, or None if the location is not found.
        """
        # Block Logic: Checks if the requested location exists in the sensor data.
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a given location.
        @param location: The key identifying the sensor data to update.
        @param data: The new data value to set.
        Functional Utility: Writes new data to a specific location.
        """
        # Block Logic: Checks if the requested location exists in the sensor data.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Joins the device's main processing thread, effectively stopping its operation.
        Functional Utility: Ensures that the DeviceThread completes its execution before the program exits.
        """
        self.thread.join()


class ScriptThread(Thread):
    """
    @brief A dedicated thread for executing a single script on a device's data.
    This thread is responsible for acquiring the appropriate lock for the data location,
    collecting data from neighbors and itself, executing the assigned script,
    and propagating the results back to relevant devices.
    """
    

    def __init__(self, device, script, location, neighbours):
        """
        @brief Initializes a ScriptThread for a specific script execution.
        @param device: The Device object on which the script will run.
        @param script: The script object to execute.
        @param location: The data location pertinent to this script execution.
        @param neighbours: A list of neighboring Device objects to interact with.
        Functional Utility: Associates the thread with all necessary context for script execution.
        """
        
        Thread.__init__(self)
        self.device = device
        self.script = script


        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for a ScriptThread.
        Functional Utility: Manages the lifecycle of a single script execution,
        including data acquisition, locking, script execution, and result dissemination.
        """
        # Block Logic: Acquires a lock for the specific data location to ensure exclusive access during processing.
        # It uses an index into `device.location_locks` which implies `location` is an index or mapped to one.
        with self.device.location_locks[self.location]:
            script_data = []
            
            # Block Logic: Collects data from neighboring devices for the script.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Collects data from the current device for the script.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
            # Block Logic: Executes the script if there is any data to process.
            # Pre-condition: `script_data` must not be empty for script execution.
            if script_data != []:
                # Functional Utility: Runs the assigned script with the collected data.
                result = self.script.run(script_data)
                
                # Block Logic: Propagates the script result to neighboring devices.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                # Functional Utility: Updates the current device's data with the script result.
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    @brief A dedicated thread for a Device to manage its timepoints and orchestrate task distribution.
    This thread coordinates the overall process for a single device, fetching neighbor information,
    signaling timepoint completion, distributing scripts to individual ScriptThread instances,
    and synchronizing with other devices using a barrier.
    """
    

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread with its associated device.
        @param device: The Device object that this thread will manage.
        Functional Utility: Sets up the thread name and associates it with the device.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        Functional Utility: Orchestrates the processing at each timepoint for the device.
        It manages the flow of execution, including fetching data, creating and joining
        ScriptThread instances for assigned scripts, and ensuring synchronization
        with other devices via a barrier.
        """
        while True:
            # Functional Utility: Retrieves the current set of neighboring devices from the supervisor.
            # Invariant: `vecini` will be None if the supervisor signals termination.
            vecini = self.device.supervisor.get_neighbours()
            if vecini is None:
                break
            
            # Functional Utility: Waits until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()
            threads = []
            
            # Block Logic: If there are neighbors, create and start ScriptThreads for each assigned script.
            # Invariant: Each script will be executed by a dedicated thread.
            if len(vecini) != 0: # Checks if there are any neighbors to interact with.
                for (script, locatie) in self.device.scripts:
                    thread = ScriptThread(self.device, script, locatie, vecini)
                    threads.append(thread)
                    thread.start()
                # Block Logic: Waits for all ScriptThreads for the current timepoint to complete their execution.
                for thread in threads:
                    thread.join()
            
            # Functional Utility: Clears the event to prepare for the next timepoint's script assignments.
            self.device.timepoint_done.clear()
            
            # Functional Utility: Synchronizes with other device threads using the shared barrier,
            # ensuring all devices complete their current timepoint before proceeding to the next.
            self.device.barrier.wait()

