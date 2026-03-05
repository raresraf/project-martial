"""
This module defines classes for simulating a distributed system, where individual
`Device` instances process sensor data and execute scripts in a multi-threaded
environment. It includes a custom two-phase barrier for synchronization and
a dedicated thread for script execution.
"""


from threading import Thread, Event, Lock, Semaphore

from threading import Condition, RLock    




class ReusableBarrier():
    """
    A reusable two-phase barrier synchronization primitive for coordinating multiple threads.

    Threads wait at the barrier in two distinct phases. All `num_threads` must
    arrive in the first phase before proceeding to the second, and similarly for
    the second phase. This ensures all threads complete one round of work before
    any start the next.
    """
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                in each phase before any are released.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other threads
        (up to `num_threads`) have completed both phases.

        This method coordinates the two-phase synchronization, ensuring all threads
        pass through the first phase before all are released to the second, and so on.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Manages a single phase of the barrier synchronization.

        A thread decrements a counter and either notifies all waiting threads
        (if it's the last to arrive) or waits itself.

        Args:
            count_threads (list): A list containing the current count of threads
                                  yet to reach this phase of the barrier.
            threads_sem (Semaphore): The semaphore used to release threads
                                     waiting in this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # Pre-condition: Check if this is the last thread to reach this phase of the barrier.
            if count_threads[0] == 0:
                # Block Logic: If the last thread arrives, release all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads # Reset the counter for the next use of this phase.
        threads_sem.acquire() # Acquire the semaphore, blocking if not released by the last thread.

class Device(object):
    """
    Represents a single device in the simulated distributed system.

    Each device has a unique ID, sensor data, and interacts with a supervisor.
    It manages scripts for processing, coordinates with other devices through
    a barrier, and uses worker threads for parallel execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor data for various locations.
            supervisor (object): The supervisor object responsible for coordinating devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barrier = None
        self.list_thread = []
        self.thread.start()
        self.location_lock = [None] * 100

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources and synchronization mechanisms for a collection of devices.

        This method initializes a reusable barrier for all devices, if one hasn't
        been set, and populates the device's list of all other devices.

        Args:
            devices (list): A list of Device instances to be set up.
        """
        
        # Pre-condition: Check if the barrier for this device has not been initialized.
        if self.barrier is None:
            # Block Logic: Initializes a shared ReusableBarrier for all devices if not already done.
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            # Invariant: All devices in the list will share the same ReusableBarrier instance.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        # Block Logic: Populates the device's internal list of all other devices in the system.
        # Invariant: 'self.devices' contains a reference to every other device.
        for device in devices:
            if device is not None:
                self.devices.append(device)


    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific location.

        If a script is provided, it's added to the device's script queue, and
        a location-specific lock is ensured to exist. If `script` is None,
        it signals that all scripts for the current timepoint have been assigned.

        Args:
            script (object): The script object to assign, or None to signal
                             timepoint completion.
            location (str): The location associated with the script.
        """
        flag = 0 # Flag to indicate if a location lock was found from another device.
        # Pre-condition: Check if a script is actually being assigned.
        if script is not None:
            self.scripts.append((script, location)) # Add the script to the device's queue.
            # Pre-condition: Check if a lock for this location has not been initialized on this device.
            if self.location_lock[location] is None:
                # Block Logic: Search for an existing lock for this location among other devices.
                # Invariant: If found, this device will use the same lock; otherwise, it creates a new one.
                for device in self.devices:
                    # If another device has a lock for this location, use it.
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        flag = 1 # Set flag to indicate a lock was found.
                        break
                # If no existing lock was found, create a new one for this location.
                if flag == 0:
                    self.location_lock[location] = Lock()
            self.script_received.set() # Signal that a script has been assigned.
        else:
            # If script is None, it indicates that all scripts for the current
            # timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            any: The sensor data for the specified location, or None if the
                 location is not found in the device's sensor data.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        This method acquires a lock to ensure thread-safe updates to the
        device's sensor data.

        Args:
            location (str): The location for which to set data.
            data (any): The new sensor data to be set.
        """
        # Pre-condition: Check if the location exists in the sensor data.
        if location in self.sensor_data:
            self.sensor_data[location] = data # Update the sensor data.


    def shutdown(self):
        """
        Shuts down the device's main thread.

        This method blocks until the device's associated `DeviceThread` has
        completed its execution.
        """
        self.thread.join()


class MyThread(Thread):
    """
    Represents a worker thread that processes a single script for a device.

    These threads are spawned by a `DeviceThread` to execute scripts in parallel.
    They gather data from the device and its neighbors, run the script, and
    update sensor data while ensuring thread-safe access.
    """
    def __init__(self, device, location, script, neighbours):
        """
        Initializes a MyThread worker.

        Args:
            device (Device): The Device instance that owns this worker.
            location (str): The location associated with the script to process.
            script (object): The script object to execute.
            neighbours (list): A list of neighboring Device instances.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Executes the main logic of the worker thread.

        This method acquires a lock for the specific location, gathers data
        from the device and its neighbors, processes the script, and then
        updates the sensor data for both the device and its neighbors.
        """
        # Acquire lock for the specific location.
        if self.device.location_lock[self.location] is not None:
            self.device.location_lock[self.location].acquire()
        
        script_data = [] # List to collect data for the script.
        
        # Block Logic: Gather data from neighboring devices at the specified location.
        # Invariant: 'script_data' collects all available data for the location from neighbors.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
        # Block Logic: Gather data from the current device's own sensor at the specified location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Pre-condition: Check if any data was collected to process.
        if script_data != []:
            # Execute the script with the collected data.
            result = self.script.run(script_data)
            
            # Acquire lock for the specific location for updating data.
            if self.device.location_lock[self.location] is not None:
                self.device.location_lock[self.location].acquire()

            # Block Logic: Update the sensor data of neighboring devices with the script's result.
            # Only updates if the new result is greater than the existing data.
            for device in self.neighbours:
                device.set_data(self.location, result)
                
            # Update the current device's own sensor data with the script's result.
            self.device.set_data(self.location, result)
        
        if self.device.location_lock[self.location] is not None:
            self.device.location_lock[self.location].release() # Release lock for the specific location.


class DeviceThread(Thread):
    """
    Manages the primary execution logic for a Device in a separate thread.

    This thread is responsible for coordinating with a supervisor, dividing
    scripts into tasks for worker threads, and synchronizing with other
    `DeviceThread` instances using a barrier.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def divide_in_threads(self, neighbours):
        """
        Divides the assigned scripts into sub-tasks and distributes them among worker threads.

        This method creates and manages `MyThread` instances to process
        scripts in parallel, improving performance for complex workloads.

        Args:
            neighbours (list): A list of neighboring Device instances.
        """

        
        threads = [] # List to hold the worker thread objects.

        
        nr = len(self.device.scripts) # Total number of scripts.
        numar = 1 # Number of scripts per worker (default 1).
        # Pre-condition: Check if the number of scripts exceeds a threshold (e.g., 8)
        # to determine if more workers are needed.
        if nr > 8:
            numar = nr // 8 # Calculate scripts per worker. Use integer division.
            # The original code had `nr = 8` which seems incorrect as it limits workers but not script count.
            # This logic seems to distribute scripts across at most 8 workers if more than 8 scripts exist.
            nr = 8 # Limit to a maximum of 8 worker threads.

        # Block Logic: Create and initialize worker threads.
        # Invariant: Each worker thread is assigned a portion of the scripts to process.
        for i in range(0,nr):
            # Pre-condition: Check if it's the last worker to assign remaining scripts.
            if i == nr - 1:
                # Assign all remaining scripts to the last worker.
                t = MyThread(self.device, self.device.scripts[i * numar][1], self.device.scripts[i * numar][0], neighbours)
            else:
                # Assign a fixed number of scripts to other workers.
                t = MyThread(self.device, self.device.scripts[i * numar][1], self.device.scripts[i * numar][0], neighbours)
            threads.append(t) # Add the worker thread to the list.

        # Block Logic: Start all worker threads.
        # Invariant: All worker threads begin parallel execution of their assigned scripts.
        for i in range(0, nr):
            threads[i].start()

        # Block Logic: Wait for all worker threads to complete their execution.
        # Invariant: The main device thread pauses until all its spawned workers have finished.
        for i in range(0,nr):
            threads[i].join()

    def run(self):
        """
        Executes the main logic of the device thread.

        This method continuously coordinates with the supervisor, processes
        assigned scripts using worker threads, and synchronizes with other
        device threads at a barrier. It handles shutdown conditions.
        """
        # Main loop for the device thread, runs indefinitely until shutdown.
        while True:
            # Retrieve information about neighboring devices from the supervisor.
            # This is done at the beginning of each cycle to get up-to-date neighbor information.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: Check if the device has no neighbors, indicating a shutdown scenario.
            if neighbours is None:
                break # Exit the main loop, effectively shutting down the device thread.

            self.device.timepoint_done.wait() # Wait for scripts to be assigned for the current timepoint.

            # Block Logic: Divide the assigned scripts into sub-tasks and distribute them
            # among worker threads for parallel processing.
            self.divide_in_threads(neighbours)

            self.device.scripts = [] # Clear the scripts for the next timepoint.
            self.device.timepoint_done.clear() # Clear the timepoint done signal for the next timepoint.
            self.device.barrier.wait() # Synchronize all device threads before proceeding to the next timepoint.
