"""
This module implements a device simulation framework that utilizes a multi-threaded
approach for distributed script execution and synchronization. It defines:
- Barrier: A re-usable barrier mechanism for threads using a Condition variable.
- Device: Represents a simulated device managing sensor data and script assignments,
          and orchestrating its own pool of DeviceThread workers.
- DeviceThread: Worker threads responsible for executing assigned scripts and
                synchronizing with other devices.

The system uses Events, Locks, and Condition variables for inter-thread communication and data consistency.
"""


from threading import Event, Thread, Condition, Lock


class Barrier(object):
    """
    Implements a reusable barrier synchronization mechanism using a condition variable.
    This barrier allows a fixed number of threads to wait until all have reached a certain point
    before any are allowed to proceed, and can then be reset for subsequent synchronizations.
    """
    

    def __init__(self, num_threads=0):
        """
        Initializes the reusable barrier.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        
        self.cond = Condition()

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have reached this barrier.
        When the last thread arrives, all waiting threads are notified.
        """
        
        
        self.cond.acquire() # Acquire the condition variable's intrinsic lock.
        self.count_threads -= 1
        # Block Logic: If this is the last thread to reach the barrier,
        # reset the counter and notify all waiting threads.
        if self.count_threads == 0:
            
            self.cond.notify_all() # Notify all waiting threads.
            self.count_threads = self.num_threads # Reset counter for next use.
        else:
            
            self.cond.wait() # Wait until all threads have arrived.
        
        self.cond.release() # Release the condition variable's intrinsic lock.

class Device(object):
    """
    Represents a simulated device in a distributed system. Each device manages
    its sensor data, assigns and executes scripts via its `DeviceThread` workers,
    and interacts with a supervisor. Synchronization is handled through shared
    locks and a global barrier.
    """
    
    
    bariera_devices = Barrier() # Class-level global barrier for synchronizing all devices.
    locks = [] # Class-level list to store locks for different locations.

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

        
        
        self.scripts = [] # List to store assigned scripts.
        self.locations = [] # List to store locations corresponding to assigned scripts.
        
        self.nr_scripturi = 0 # Counter for the number of scripts assigned to this device.
        
        self.script_crt = 0 # Index of the current script being processed.

        
        
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.

        
        self.neighbours = [] # List to store neighboring devices.
        self.event_neighbours = Event() # Event to signal when neighbor information is available.
        self.lock_script = Lock() # Lock to protect access to `script_crt` and `nr_scripturi`.
        self.bar_thr = Barrier(8) # Barrier for synchronizing worker threads within this device (1 master + 7 workers).

        
        self.thread = DeviceThread(self, 1) # The master thread for this device (first worker).
        self.thread.start() # Start the master thread.
        self.threads = [] # List to hold additional worker DeviceThread instances.
        # Block Logic: Create and start 7 additional DeviceThread worker instances.
        for _ in range(7):
            tthread = DeviceThread(self, 0) # Create worker threads (non-master).


            self.threads.append(tthread)
            tthread.start()

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
        This includes initializing the class-level global barrier (`bariera_devices`)
        and location-specific locks (`locks`), which are then shared across all devices.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        # Block Logic: Initialize the global barrier for all devices with the total number of devices.
        Device.bariera_devices = Barrier(len(devices))
        
        # Block Logic: If the class-level locks list is empty, initialize it with a lock for each location.
        if Device.locks == []:
            # Inline: The number of locations is obtained from the supervisor's testcase.
            for _ in range(self.supervisor.supervisor.testcase.num_locations):
                Device.locks.append(Lock())

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific location.
        If a script is provided, it's added to the device's script list and
        the `nr_scripturi` counter is incremented. If no script is provided (None),
        it signals that a timepoint is done.

        Args:
            script (Script or None): The script object to assign, or None to signal timepoint completion.
            location (str): The location associated with the script.
        """
        
        if script is not None:
            self.scripts.append(script) # Add the script to the list of scripts.
            self.locations.append(location) # Add the location to the list of locations.
            
            self.nr_scripturi += 1 # Increment the count of assigned scripts.
        else:
            # Block Logic: If no script is provided, signal that processing for the current timepoint is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            Any or None: The sensor data if available for the location, otherwise None.
        """
        
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        Args:
            location (str): The location for which to set data.
            data (Any): The new sensor data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its master `DeviceThread` and all other
        worker `DeviceThread` instances.
        """
        
        self.thread.join()
        for tthread in self.threads:
            tthread.join()


class DeviceThread(Thread):
    """
    A worker thread for a `Device` instance. Each thread is responsible for
    processing a subset of assigned scripts and managing data for its device.
    It synchronizes with other `DeviceThread` instances within the same `Device`
    using a shared barrier.
    """
    

    def __init__(self, device, first):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent Device instance this thread belongs to.
            first (int): A flag indicating if this is the "first" (master) thread (1) or a worker (0).
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.first = first

    def run(self):
        """
        Executes the main logic of the DeviceThread.
        - The "first" thread fetches neighbor information and resets script counters.
        - All threads wait for neighbor information to be available.
        - If no neighbors are returned (signal for shutdown), the loop breaks.
        - All threads wait for scripts to be assigned for the current timepoint.
        - Each thread processes a portion of the assigned scripts (round-robin).
        - Acquires location-specific locks, executes scripts, and updates data on devices.
        - Releases location-specific locks.
        - Synchronizes with a local barrier after processing its assigned scripts.
        - The "first" thread clears events for the next cycle.
        - The "first" thread synchronizes with the global barrier.
        """
        while True:
            # Block Logic: The "first" thread (master) is responsible for fetching neighbor information
            # and resetting script counters for the next timepoint.
            if self.first == 1:
                
                self.device.neighbours = self.device.supervisor.get_neighbours() # Fetch neighbors.
                self.device.script_crt = 0 # Reset current script index.
                self.device.event_neighbours.set() # Signal that neighbors are available.

            # Block Logic: All threads wait for the neighbor information to be available.
            self.device.event_neighbours.wait()

            # Block Logic: If no neighbors are returned (shutdown signal), break the loop.
            if self.device.neighbours is None:
                break

            # Block Logic: All threads wait for scripts to be assigned for the current timepoint.
            self.device.timepoint_done.wait()

            # Block Logic: Loop to continuously fetch and process scripts assigned to this device.
            while True:
                # Block Logic: Acquire lock to safely access and increment `script_crt`.
                self.device.lock_script.acquire()
                index = self.device.script_crt # Get the index of the current script.
                self.device.script_crt += 1 # Increment for the next thread.
                self.device.lock_script.release() # Release lock.

                # Block Logic: If all scripts have been distributed, break the inner loop.
                if index >= self.device.nr_scripturi:
                    break

                # Block Logic: Retrieve the script and location for the current index.
                location = self.device.locations[index]
                script = self.device.scripts[index]

                # Block Logic: Acquire a lock for the specific location to prevent race conditions during data access.
                Device.locks[location].acquire()

                script_data = [] # List to collect data for the current script.
                
                # Block Logic: Collect data from neighboring devices for the current location.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collect data from the current device itself for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: If data was collected, execute the script and update devices.
                if script_data != []:
                        
                    result = script.run(script_data) # Inline: Execute the assigned script with collected data.

                    
                    # Block Logic: Propagate the script's result to neighboring devices.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    # Block Logic: Update the current device's sensor data with the script's result.
                    self.device.set_data(location, result)

                # Block Logic: Release the lock for the current location.
                Device.locks[location].release()

            # Block Logic: Synchronize all worker threads within this device.
            self.device.bar_thr.wait()
            
            # Block Logic: The "first" thread clears events for the next cycle.
            if self.first == 1:
                self.device.event_neighbours.clear() # Clear neighbor event.
                self.device.timepoint_done.clear() # Clear timepoint_done event.
            self.device.bar_thr.wait() # Final synchronization with other worker threads.
            
            # Block Logic: The "first" thread synchronizes with the global barrier, including all other devices.
            if self.first == 1:
                Device.bariera_devices.wait()

