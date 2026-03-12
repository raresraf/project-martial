


"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `ReusableBarrier`: A reusable barrier synchronization mechanism using semaphores stored in lists.
- `Device`: Represents a single device, managing its sensor data, communication with a supervisor,
  and orchestrating multi-threaded script execution.
- `ScriptThread`: A dedicated thread for executing a single script, collecting data,
  and updating device states.
- `DeviceThread`: The main orchestrating thread for a `Device`, responsible for fetching
  neighbor information, spawning `ScriptThread`s for each assigned script, and
  managing timepoint synchronization.

The system utilizes `threading.Event` for signaling, `threading.Lock` for protecting
shared resources (like `location_locks`), and custom barrier implementations for coordination.
"""


from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    A reusable barrier synchronization mechanism for multiple threads using semaphores.
    This barrier allows a fixed number of threads to wait at a synchronization point,
    and once all threads arrive, they are all released simultaneously. The counters
    are stored in lists (`count_threads1`, `count_threads2`) to allow for reusability
    of the barrier instance across multiple `wait` calls.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that will participate
                               in the barrier synchronization.
        """
        self.num_threads = num_threads
        # Counters for the two phases of the barrier. Stored in lists to be mutable
        # when passed to the 'phase' method, enabling reusability.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock() # Lock to protect the shared counters during decrements and resets.
        
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase of threads to wait on.
        
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase of threads to wait on.

    def wait(self):
        """
        Causes the calling thread to wait until all threads have reached this barrier.
        This method orchestrates a two-phase synchronization to ensure reusability
        without deadlocks.
        """
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the current count of threads
                                  remaining for this phase.
            threads_sem (Semaphore): The semaphore threads wait on for this phase.
        """
        
        with self.count_lock: # Protect shared counter access.
            count_threads[0] -= 1 # Decrement the count of threads remaining.
            
            if count_threads[0] == 0: # If this is the last thread to arrive:
                for _ in range(self.num_threads):
                    threads_sem.release() # Release all waiting threads by incrementing the semaphore.
                
                count_threads[0] = self.num_threads # Reset counter for next use.
        threads_sem.acquire() # Wait (decrement) the semaphore, blocking until released by the last thread.


class Device(object):
    """
    Represents a single device within a simulated distributed environment.
    Each device manages its own sensor data, communicates with a supervisor,
    and orchestrates multi-threaded script execution. It utilizes a dedicated
    `DeviceThread` to manage `ScriptThread`s for concurrent task processing.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for the device.
            supervisor (object): A reference to a supervisor object for inter-device communication.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to hold assigned scripts (tuples of (script, location)).
        self.timepoint_done = Event() # Event to signal that the current timepoint's processing is complete.
        self.barrier = None # Placeholder for the global ReusableBarrier, set in setup_devices.
        self.thread = DeviceThread(self) # The main orchestrating thread for this device.
        self.thread.start() # Start the DeviceThread.
        self.location_locks = None # Placeholder for shared location locks, set in setup_devices.

    def __str__(self):
        """
        Returns a string representation of the device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes a global barrier and shared location-specific locks,
        then distributes them to all devices. This method is designed to be
        called only by the device with `device_id == 0`. It creates a single
        `ReusableBarrier` and a list of `Lock`s for each unique data location
        found across all devices.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        # Block Logic: Only the device with device_id 0 performs this setup.
        if 0 == self.device_id:
            # Inline: Creates a global ReusableBarrier for synchronization among all DeviceThreads.
            self.barrier = ReusableBarrier(len(devices))
            
            locations = [] # List to collect all unique data locations across all devices.
            # Block Logic: Identify all unique data locations across all devices.
            for device in devices:
                for location in device.sensor_data:
                    if location not in locations:
                        locations.append(location)
            
            # Block Logic: Create a Lock for each unique data location.
            self.location_locks = []
            for _ in range(len(locations)):
                self.location_locks.append(Lock())
            
            # Inline: Distributes the created global barrier and location locks to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals timepoint completion.
        If a script is provided, it's appended to the device's internal script list.
        If `script` is None, it signals that script assignments for the current
        timepoint are complete by setting `timepoint_done`.

        Args:
            script (object): The script object to be executed, or None to signal timepoint completion.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
        else:
            # Inline: If script is None, signal that script assignments for the timepoint are done.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's `sensor_data` dictionary.

        Args:
            location (int): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location in this device's `sensor_data` dictionary.
        The data is updated only if the location exists in the `sensor_data`.

        Args:
            location (int): The location identifier for which to set data.
            data (any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device by waiting for its main `DeviceThread` to complete.
        This implicitly triggers the shutdown of associated `ScriptThread`s managed by `DeviceThread`.
        """
        self.thread.join() # Wait for the DeviceThread to finish its execution.


class ScriptThread(Thread):
    """
    A worker thread responsible for executing a single script within a distributed device system.
    It collects necessary data from the parent device and its neighbors, runs the assigned script,
    and updates the sensor data on relevant devices, all while acquiring a location-specific lock.
    """

    def __init__(self, device, script, location, neighbours):
        """
        Initializes a `ScriptThread` instance.

        Args:
            device (Device): The parent `Device` object that spawned this thread.
            script (object): The script object to be executed.
            location (int): The data location identifier to which the script applies.
            neighbours (list): A list of Device objects representing neighboring devices.
        """
        
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

        def run(self):

            """

            The main execution method for the `ScriptThread`.

            It acquires the location-specific lock, collects data from neighbors and

            its parent device, executes the assigned script, and then updates the

            sensor data on relevant devices with the script's result, before releasing the lock.

            """

            # Block Logic: Acquire the location-specific lock to ensure exclusive access

            # to data at this `location` during script execution and data update.

            with self.device.location_locks[self.location]:

                script_data = [] # List to collect input data for the script.

                

                # Block Logic: Collect data from all neighboring devices at the specified location.

                for device in self.neighbours:

                    data = device.get_data(self.location) # Get data from the neighbor.

                    if data is not None:

                        script_data.append(data) # Add to script input if available.

                

                # Block Logic: Collect data from this worker's own parent device at the specified location.

                data = self.device.get_data(self.location)

                if data is not None:

                    script_data.append(data) # Add to script input if available.

                

                # Block Logic: If input data is available, execute the script and update device data.

                if script_data != []:

                    # Inline: Execute the script's `run` method with the collected data.

                    result = self.script.run(script_data)

                    

                    # Block Logic: Update sensor data for all involved devices (neighbors and self) with the result.

                    for device in self.neighbours:

                        device.set_data(self.location, result) # Update neighbor's data.

                    

                    self.device.set_data(self.location, result) # Update this device's own data.

    

    class DeviceThread(Thread):

    

        """

    

        The main orchestrating thread for a `Device`.

    

        This thread is responsible for fetching neighbor information from the supervisor,

    

        waiting for timepoint completion signals, and then spawning individual `ScriptThread`s

    

        for each assigned script. It manages the execution and joining of these worker threads

    

        and participates in global synchronization.

    

        """

    

    

    

        def __init__(self, device):

    

            """

    

            Initializes a `DeviceThread` instance.

    

    

    

            Args:

    

                device (Device): The parent `Device` object this thread belongs to.

    

            """

    

            

    

            Thread.__init__(self, name="Device Thread %d" % device.device_id)

    

            self.device = device

    

    

    

            def run(self):

    

    

    

                """

    

    

    

                The main execution loop for the `DeviceThread`.

    

    

    

                It continuously fetches neighbor data, waits for timepoint readiness,

    

    

    

                spawns `ScriptThread`s for each assigned script, waits for their completion,

    

    

    

                and participates in global synchronization.

    

    

    

                """

    

    

    

                while True:

    

    

    

                    # Block Logic: Fetch neighbor information from the supervisor.

    

    

    

                    vecini = self.device.supervisor.get_neighbours() # `vecini` is a variable name for neighbors.

    

    

    

                    # Inline: If `vecini` is None, it signals termination for the device.

    

    

    

                    if vecini is None:

    

    

    

                        break # Exit the main loop, terminating the DeviceThread.

    

    

    

                    

    

    

    

                    # Block Logic: Wait for the `timepoint_done` event to be set,

    

    

    

                    # indicating that all scripts for the current timepoint have been assigned.

    

    

    

                    self.device.timepoint_done.wait()

    

    

    

                    threads = [] # List to hold spawned `ScriptThread` instances.

    

    

    

                    

    

    

    

                    # Block Logic: If there are neighbors (i.e., not a single-device simulation),

    

    

    

                    # spawn a `ScriptThread` for each assigned script.

    

    

    

                    if len(vecini) != 0: # This condition might be problematic if 'neighbours' can be an empty list.

    

    

    

                        # Inline: Iterate through each assigned script (script, location) pair.

    

    

    

                        for (script, locatie) in self.device.scripts:

    

    

    

                            # Inline: Create and start a new `ScriptThread` for each script.

    

    

    

                            thread = ScriptThread(self.device, script, locatie, vecini)

    

    

    

                            threads.append(thread) # Add to the list of managed script threads.

    

    

    

                            thread.start() # Start the `ScriptThread`.

    

    

    

                        # Inline: Wait for all spawned `ScriptThread`s to complete their execution.

    

    

    

                        for thread in threads:

    

    

    

                            thread.join()

    

    

    

                    

    

    

    

                    # Block Logic: Clear the `timepoint_done` event for the next cycle.

    

    

    

                    self.device.timepoint_done.clear()

    

    

    

                    

    

    

    

                    # Block Logic: Synchronize with other devices at the global barrier.

    

    

    

                    # This ensures all devices have completed their script processing for the timepoint.

    

    

    

                    self.device.barrier.wait()

