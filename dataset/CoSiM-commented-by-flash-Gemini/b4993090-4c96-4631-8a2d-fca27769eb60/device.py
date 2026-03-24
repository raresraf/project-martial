"""
This module defines a simulation framework for distributed sensor data processing.

It includes:
- ReusableBarrier: A synchronization primitive for coordinating multiple threads.
- Device: Represents a simulated device with sensor data, a supervisor, and script execution capabilities.
- DeviceThread: A thread that runs on each Device to perform simulation tasks.
"""


from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    A reusable barrier synchronization primitive for coordinating multiple threads.
    Threads wait at the barrier until all 'num_threads' have arrived, then all are released.
    It supports two phases to allow for continuous synchronization without busy-waiting.
    """
    

    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait until all threads have reached this point.
        This method orchestrates the two phases of the barrier.
        """
        
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        First phase of the barrier. Threads decrement a counter and the last thread
        to reach the barrier releases all waiting threads for phase 1.
        """
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """
        Second phase of the barrier. Similar to phase 1, threads decrement a counter
        and the last thread to reach releases all waiting threads for phase 2.
        This completes one full cycle of the reusable barrier.
        """
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a simulated computational device in a distributed system.
    Each device has a unique ID, sensor data, interacts with a supervisor,
    and can execute scripts at specific data locations.
    It manages its own threads and synchronization primitives.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping data locations to sensor readings.
            supervisor (Supervisor): The supervisor object responsible for coordinating devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.timepoint_done = Event()
        self.scripts = []

        self.barrier_worker = ReusableBarrier(8)
        self.setup_event = Event()
        self.devices = []
        self.locks = None
        self.neighbours = []
        self.barrier = None
        self.threads = []

        # Block Logic: Creates and starts a fixed number of DeviceThread instances.
        # Each thread is associated with this device and given a unique ID.
        for i in range(8):
            self.threads.append(DeviceThread(self, i))

        for thr in self.threads:
            thr.start()

        self.location_lock = []

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device {device_id}".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the devices in the simulation, particularly setting up the shared barrier
        and location locks. This method is typically called by the supervisor.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        
        # Block Logic: Only the device with device_id 0 (acting as coordinator) performs initial setup.
        # This includes creating a shared barrier for all threads across all devices,
        # determining the maximum location index, and initializing a shared list of location locks.
        # It also signals each device that the initial setup phase is complete.
        if self.device_id == 0:
            # Create a barrier for all threads (8 threads per device).
            barrier = ReusableBarrier(len(devices)*8)
            self.barrier = barrier
            location_max = 0
            # Distribute the barrier and find the maximum sensor data location across all devices.
            for device in devices:
                device.barrier = barrier
                for location, data in device.sensor_data.iteritems():
                    if location > location_max:
                        location_max = location
                device.setup_event.set() # Signal completion of initial barrier distribution.
            self.setup_event.set() # Signal coordinator's own setup_event.

            # Initialize a shared list of locks for each location.
            self.location_lock = [None] * (location_max + 1)

            # Distribute the shared location_lock list to all devices.
            for device in devices:
                device.location_lock = self.location_lock
                device.setup_event.set() # Signal completion of location_lock distribution.
            self.setup_event.set() # Signal coordinator's own setup_event.


    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific data location on this device.
        Manages the acquisition of a lock for the given location if it's not already acquired.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the data location where the script should run.
        """
        
        # Initialize busy flag.
        busy = 0
        # Block Logic: If a script is provided, it's added to the device's script list.
        # A lock for the script's location is then acquired or initialized.
        # If no script is provided (script is None), it signifies the end of a timepoint.
        if script is not None:
            self.scripts.append((script, location))
            # If no lock exists for this location, try to find one from other devices.
            if self.location_lock[location] is None:
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        # Use an existing lock for this location found on another device.
                        self.location_lock[location] = device.location_lock[location]
                        busy = 1
                        break

                if busy == 0:
                    # If no existing lock, create a new one for this location.
                    self.location_lock[location] = Lock()
            # Signal that a script has been received.
            self.script_received.set()
        else:
            # If script is None, signal that the current timepoint is done.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (int): The identifier for the data location.

        Returns:
            Any: The sensor data at the specified location, or None if the location is not found.
        """

        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location. Data is updated only if the location exists.

        Args:
            location (int): The identifier for the data location.
            data (Any): The new sensor data to set.
        """

        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining all its associated DeviceThread instances,
        ensuring all threads complete their execution gracefully.
        """

        
        for thr in self.threads:
            thr.join()


class DeviceThread(Thread):
    """
    Represents a worker thread associated with a Device.
    These threads perform the actual simulation tasks, including fetching neighbor information,
    synchronizing with other threads, and executing assigned scripts on sensor data.
    """
    

    def __init__(self, device, idd):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device instance this thread is associated with.
            idd (int): A unique identifier for this thread within its parent device.
        """

        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.idd = idd

    def run(self):
        """
        The main execution loop for the DeviceThread.
        It orchestrates the simulation process, including setup, neighbor discovery,
        script execution, and synchronization across timepoints.
        """
        # Block Logic: Waits for the device to complete its initial setup phase.
        # This ensures that shared resources like barriers and locks are initialized.
        self.device.setup_event.wait()

        while True:
            # Block Logic: Only thread with idd 0 queries the supervisor for neighbors
            # to avoid redundant calls. This information is then shared across the device.
            if self.idd == 0:
                neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours = neighbours

            # Block Logic: All worker threads wait at this barrier to synchronize
            # before proceeding to the next phase of the timepoint.
            self.device.barrier_worker.wait()

            # Block Logic: If no neighbors are assigned (e.g., simulation termination signal),
            # the thread breaks out of the main loop, effectively shutting down.
            if self.device.neighbours is None:
                break

            # Block Logic: Waits for a signal that a new timepoint has begun and scripts are assigned.
            self.device.timepoint_done.wait()
            # Block Logic: All worker threads wait at this barrier to synchronize
            # after processing the timepoint_done event.
            self.device.barrier_worker.wait()

            i = 0
            # Block Logic: Iterates through assigned scripts and executes those
            # that are designated for this specific thread (idd).
            for (script, location) in self.device.scripts:
                # Distributes script execution among threads based on thread ID.
                if i % 8 == self.idd:
                    # Block Logic: Acquires a lock for the specific data location
                    # to ensure exclusive access during script execution and data modification.
                    with self.device.location_lock[location]:
                        script_data = []
                        # Block Logic: Gathers sensor data from neighboring devices for the current location.
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        # Block Logic: Gathers sensor data from the current device for the current location.
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        # Block Logic: If relevant script data is available, executes the script.
                        if script_data != []:
                            # Execute the assigned script with the collected data.
                            result = script.run(script_data)

                            # Block Logic: Propagates the script's result to neighboring devices.
                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            
                            # Block Logic: Updates the current device's sensor data with the script's result.
                            self.device.set_data(location, result)
                i = i + 1

            # Block Logic: Clears the timepoint_done event to reset it for the next timepoint.
            self.device.timepoint_done.clear()
            # Block Logic: All threads synchronize at the main barrier before completing the timepoint,
            # ensuring all scripts have finished processing and data is consistent.
            self.device.barrier.wait()

