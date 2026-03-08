"""
@4a2b6716-6b41-48b5-99ca-d5e81388b499/device.py
@brief This module implements a distributed device simulation framework,
featuring `Device` for managing sensors and scripts, `DeviceThread` for
coordination, `SingleDeviceThread` for parallel script execution, and a
`ReusableBarrierSem` for synchronization.
"""

from threading import Event, Thread, Lock, Semaphore


class Device(object):
    """
    Represents a single device in a simulated distributed system.

    Each device manages its own sensor data, processes scripts, and
    coordinates with a supervisor and other devices using threads and barriers.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary representing the sensor data this device holds.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when scripts have been assigned for the current timepoint.
        self.scripts = [] # List to hold assigned scripts.
        self.timepoint_done = Event() # Event to signal when script assignments for a timepoint are complete.
        # Main thread for device operations.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures shared synchronization primitives (barrier and location-specific locks)
        across all devices in the system. This method is typically called by the supervisor.

        Args:
            devices (list): A list of all Device instances in the simulated system.
        """
        # Flag to identify the 'master' device (the one with the smallest ID) for initialization.
        flag = True
        device_number = len(devices) # Total number of devices in the system.

        # Block Logic: Determine if this device has the smallest ID.
        for dev in devices:
            if self.device_id > dev.device_id:
                flag = False # If any device has a smaller ID, this is not the master.

        if flag == True: # Only the master device performs this initialization.
            # Create a reusable barrier for all devices.
            barrier = ReusableBarrierSem(device_number)
            map_locations = {} # Dictionary to hold locks for specific data locations, shared across devices.
            tmp = {} # Temporary dictionary (appears unused or for debugging).
            for dev in devices:
                dev.barrier = barrier # Assign the shared barrier to all devices.
                # Block Logic: Initialize a lock for each unique data location encountered across devices.
                tmp = list(set(dev.sensor_data) - set(map_locations)) # Find new locations.
                for i in tmp:
                    map_locations[i] = Lock() # Assign a new lock to each new location.
                dev.map_locations = map_locations # Assign the shared location locks to all devices.
                tmp = {} # Reset temporary dictionary.

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific data location.

        Args:
            script (object): The script object to be executed.
            location (int): The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script to the device's list.
            self.script_received.set() # Signal that scripts have been received (currently unused in run logic).
        else:
            self.timepoint_done.set() # If no script, signal that script assignment for this timepoint is complete.

    def get_data(self, location):
        """
        Retrieves data from the device's sensor_data at the specified location.

        Args:
            location (int): The index or key of the data to retrieve.

        Returns:
            any: The data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates data in the device's sensor_data at the specified location.

        Args:
            location (int): The index or key of the data to set.
            data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data # Update the data if the location exists.

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device, waiting for its main thread to finish.
        """
        self.thread.join() # Wait for the main DeviceThread to complete its execution.


class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for fetching neighbors,
    managing timepoint events, and coordinating script execution
    through a pool of `SingleDeviceThread` workers.
    """
    

    def __init__(self, device):
        """
        Initializes a DeviceThread instance.

        Args:
            device (Device): The parent Device instance this thread belongs to.
        """
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously fetches neighbors, waits for scripts to be assigned,
        distributes these scripts to `SingleDeviceThread` workers, and then
        synchronizes all devices using a global barrier.
        """
        while True:
            # Pre-condition: Clear the timepoint_done event for the new timepoint.
            self.device.timepoint_done.clear()
            # Functional Utility: Fetch the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # If no neighbors are returned (e.g., supervisor signals shutdown), break the loop.
            # Block Logic: Wait until scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()
            
            script_list = [] # List to hold scripts for distribution.
            thread_list = [] # List to hold `SingleDeviceThread` instances.
            index = 0 # (This variable `index` is initialized but not used in this loop for script distribution, only passed to workers).
            
            # Block Logic: Copy assigned scripts into a local list.
            for script in self.device.scripts:
                script_list.append(script)
            
            # Block Logic: Create and start `SingleDeviceThread` workers for each script.
            # Note: The current implementation creates 8 threads regardless of the number of scripts,
            # and each thread attempts to pop from the `script_list` using `self.index`.
            # This logic appears problematic as `script_list.pop(self.index)` would remove
            # an element at a fixed index, likely leading to `IndexError` or incorrect distribution.
            for i in xrange(8):
                thread = SingleDeviceThread(self.device, script_list, neighbours, index) # 'index' should probably be 'i' if intended for round-robin.
                thread.start()
                thread_list.append(thread)
            
            # Block Logic: Wait for all `SingleDeviceThread` workers to complete.
            for i in xrange(len(thread_list)):
                thread_list[i].join()
            
            # Block Logic: Synchronize all devices at the global barrier after script execution.
            self.device.barrier.wait()

class SingleDeviceThread(Thread):
    """
    A worker thread responsible for executing a single script within a Device's context.

    This thread collects data from neighbors and the local device, runs the script,
    and updates data based on the script's result.
    """
    
    def __init__(self, device, script_list, neighbours, index):
        """
        Initializes a SingleDeviceThread instance.

        Args:
            device (Device): The parent Device instance this thread belongs to.
            script_list (list): The list of all scripts assigned to the parent Device.
                                Note: The current implementation pops an element by `index`,
                                which might lead to unexpected behavior if not carefully managed.
            neighbours (list): A list of neighboring Device instances.
            index (int): The index of the script to be processed from `script_list`.
        """
        Thread.__init__(self)
        self.device = device
        self.script_list = script_list
        self.neighbours = neighbours
        self.index = index

    def run(self):
        """
        The main execution method for the SingleDeviceThread.

        It attempts to retrieve and execute a single script from the `script_list`
        based on its `index`, if the list is not empty.
        """
      # Pre-condition: Check if there are scripts in the list to process.
        if self.script_list != []:
            # Functional Utility: Pop the script at the specified index and its location.
            # This operation modifies the shared `script_list`, which could be a race condition
            # if `script_list` is modified by multiple `SingleDeviceThread` instances concurrently.
            (script, location) = self.script_list.pop(self.index)
            self.compute(script, location) # Execute the script.

    def update(self, result, location):
        """
        Updates the data at a specific location on neighboring devices and the current device.

        Args:
            result (any): The result of the script execution to update with.
            location (int): The data location to update.
        """
        # Block Logic: Update data on all neighboring devices.
        for device in self.neighbours:
            device.set_data(location, result)
        # Block Logic: Update data on the current device.
        self.device.set_data(location, result)

    def collect(self, location, neighbours, script_data):
        """
        Collects data from neighboring devices and the current device for a given location.

        Args:
            location (int): The data location to collect data from.
            neighbours (list): A list of neighboring Device instances.
            script_data (list): The list to append collected data to.
        """
        # Pre-condition: Acquire a lock for the specific data location to ensure exclusive access during data collection.
        self.device.map_locations[location].acquire()
        
        # Block Logic: Collect data from neighboring devices.
        for device in self.neighbours:
            # Data from neighbors is retrieved without explicit locking on the neighbor's side
            # for `get_data`, but `map_locations` lock protects the overall data collection process.
            data = device.get_data(location)
            if data is None:
                pass # Skip if data is not available.
            else:
                script_data.append(data) # Add collected data.

        # Block Logic: Collect data from the current device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data) # Add local data.

    def compute(self, script, location):
        """
        Executes a script by first collecting data, then running the script,
        and finally updating the relevant data.

        Args:
            script (object): The script object to execute.
            location (int): The data location relevant to the script.
        """
        script_data = [] # List to hold collected data.
        self.collect(location, self.neighbours, script_data) # Collect data from neighbors and local device.

        # Pre-condition: If there is data to process, execute the script.
        if script_data == []:
            pass # No data, so do nothing.
        else:
            # Action: Execute the script.
            result = script.run(script_data)
            self.update(result, location) # Update devices with the script's result.

        # Post-condition: Release the lock for the data location.
        self.device.map_locations[location].release()

class ReusableBarrierSem():
    """
    A reusable barrier synchronization primitive implementing a two-phase wait
    mechanism using semaphores and a lock.

    Threads wait in two distinct phases, allowing for efficient resetting and reuse.
    """
    

    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrierSem.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                in each phase before proceeding.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for threads in phase 1.
        self.count_threads2 = self.num_threads # Counter for threads in phase 2.

        self.counter_lock = Lock() # Lock to protect access to thread counters.
        self.threads_sem1 = Semaphore(0) # Semaphore for releasing threads from phase 1.
        self.threads_sem2 = Semaphore(0) # Semaphore for releasing threads from phase 2.

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have passed both
        phase 1 and phase 2 of the barrier.
        """
        self.phase1() # Execute the first phase of the barrier.
        self.phase2() # Execute the second phase of the barrier.

    def phase1(self):
        """
        First phase of the barrier. All threads decrement a counter and wait
        on `threads_sem1` until the last thread releases them.
        """
        # Block Logic: Atomically decrement the counter and check if this is the last thread.
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # If this is the last thread, release all waiting threads from phase 1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset counter for next use.

        self.threads_sem1.acquire() # Acquire (wait on) the semaphore until released by the last thread.

    def phase2(self):
        """
        Second phase of the barrier. Similar to phase 1, but uses `threads_sem2`
        and `count_threads2`.
        """
        # Block Logic: Atomically decrement the counter and check if this is the last thread.
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # If this is the last thread, release all waiting threads from phase 2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset counter for next use.

        self.threads_sem2.acquire() # Acquire (wait on) the semaphore until released by the last thread.