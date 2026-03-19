"""
This module implements a multi-threaded device simulation system,
featuring inter-device communication, script execution, and robust
synchronization mechanisms. It utilizes a reentrant barrier for
coordinated thread execution across multiple devices and within
each device.

The system aims to simulate devices that interact with their neighbors,
process sensor data using assigned scripts, and synchronize their operations
across discrete timepoints.
"""

from threading import Event, Thread, Lock, Condition


class ReentrantBarrier(object):
    """
    Implements a reentrant barrier synchronization primitive.
    It allows a fixed number of threads to wait until all threads have
    reached the barrier point before proceeding. This barrier is designed
    to be reusable after all threads have passed through it.
    """


    def __init__(self, num_threads):
        """
        Initializes the reentrant barrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                before any are released.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads  # Current count of threads waiting.
        self.cond = Condition()                # Condition variable for thread waiting/notification.


    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        participating threads have also called wait().
        Once all threads arrive, they are all released.
        """
        self.cond.acquire()  # Acquire the lock associated with the Condition.
        self.count_threads -= 1
        if self.count_threads == 0:
            # If this is the last thread, notify all waiting threads.
            self.cond.notify_all()
            self.count_threads = self.num_threads  # Reset for next use.
        else:
            # If not the last thread, wait until notified.
            self.cond.wait()
        self.cond.release()  # Release the lock.


class Device(object):
    """
    Represents a single simulated device in the system.
    Each device has a unique ID, sensor data, and interacts with a supervisor
    and other devices. It manages its own pool of auxiliary threads (`DeviceThread`)
    for parallel processing of scripts.
    """

    # Global synchronization primitives shared across all Device instances.
    barrier = None          # A global ReentrantBarrier for all DeviceThread instances.
    devices_lock = Lock()   # A global lock to protect shared Device-related static members.
    locations = []          # A global list of Locks, one for each data location, to protect
                            # concurrent access to sensor data at a specific location across devices.
    nrloc = 0               # Tracks the maximum number of data locations encountered.


    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary mapping data locations (int) to their values.
            supervisor (Supervisor): A reference to the supervisor object managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.sensor_data_lock = Lock()  # Lock to protect this device's sensor_data.

        self.supervisor = supervisor
        self.gen_lock = Lock()          # General purpose lock for this device.

        self.script_lock = Lock()       # Lock to protect script lists.
        self.script_event = Event()     # Event to signal availability of new scripts.
        self.scripts = []               # List of all scripts assigned to this device.
        self.working_scripts = []       # Scripts currently being processed in a timepoint.

        self.neighbour_request = False  # Flag to ensure neighbors are fetched only once per timepoint.
        self.neighbour = None           # Stores the list of neighboring devices.
        self.timepoint_done = False     # Flag to indicate if script assignment for current timepoint is done.
        self.reinit_barrier = None      # Barrier for synchronization among this device's own DeviceThreads.

        # Initialize and store the auxiliary threads for this device.
        self.threads_num = 8  # Number of auxiliary threads per device.
        self.threads = []
        for i in xrange(self.threads_num):
            self.threads.append(DeviceThread(self, i))


    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id


    def setup_devices(self, devices):
        """
        Sets up global and device-specific resources that depend on the total
        number of devices or the full set of data locations. This method ensures
        that global resources like `Device.barrier` and `Device.locations`
        are properly initialized once.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        with self.gen_lock:
            # Initialize the device-specific barrier for its own threads.
            self.reinit_barrier = ReentrantBarrier(self.threads_num)

        with Device.devices_lock:
            # Update the maximum number of locations and expand the global locations lock list if needed.
            # Invariant: Device.locations will have a Lock for each possible data location.
            if self.sensor_data: # Ensure sensor_data is not empty before attempting max.
                Device.nrloc = max(Device.nrloc, (max(self.sensor_data.keys())+1))
            while Device.nrloc != len(Device.locations):
                Device.locations.append(Lock())

            # Initialize the global barrier for all threads from all devices if it hasn't been yet.
            if Device.barrier is None:
                Device.barrier = ReentrantBarrier((len(devices) * self.threads_num))
        
        # Start all auxiliary threads associated with this device.
        for i in xrange(self.threads_num):
            self.threads[i].start()


    def assign_script(self, script, location):
        """
        Assigns a processing script and its associated data location to this device.
        If `script` is None, it signals that script assignment for the current
        timepoint is complete.

        Args:
            script (callable): The script (function or object with a run method)
                                to be executed, or None to signal timepoint completion.
            location (int): The data location this script pertains to.
        """
        with self.script_lock:
            if script is not None:
                # Add the script to both lists for current and future processing.
                self.scripts.append((script, location))
                self.working_scripts.append((script, location))
            else:
                # Signal that all scripts for the current timepoint have been assigned.
                self.timepoint_done = True
            
            # Notify any waiting DeviceThreads that a script is available or timepoint is done.
            self.script_event.set()


    def get_data(self, location):
        """
        Retrieves sensor data for a specific location within this device.
        The device's sensor_data_lock is used to ensure thread-safe access.

        Args:
            location (int): The location ID for which to retrieve data.

        Returns:
            Any: The data at the specified location, or None if the location is not found.
        """
        with self.sensor_data_lock:
            return self.sensor_data[location] \
                    if location in self.sensor_data else None


    def set_data(self, location, data):
        """
        Sets sensor data for a specific location within this device.
        The device's sensor_data_lock is used to ensure thread-safe modification.

        Args:
            location (int): The location ID for which to set data.
            data (Any): The new data value to be set.
        """
        with self.sensor_data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data


    def shutdown(self):
        """
        Initiates the graceful shutdown sequence for the device, joining all
        its managed auxiliary threads.
        """
        for i in xrange(self.threads_num):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    An auxiliary thread for a Device. Each Device has a pool of these threads
    to execute assigned scripts in parallel, manage data collection from
    neighbors, and update shared data locations.
    """


    def __init__(self, device, thread_nr):
        """
        Initializes a DeviceThread instance.

        Args:
            device (Device): The Device instance this thread belongs to.
            thread_nr (int): The unique number (ID) of this thread within its device's pool.
        """
        Thread.__init__(self, name="Device Thread %d-%d" % (device.device_id, thread_nr))
        self.device = device
        self.t_num = thread_nr


    def run_script(self, script, location):
        """
        Executes a given script at a specific data location.
        This method handles collecting relevant data from neighboring devices
        and the local device, running the script, and then disseminating
        the results back to neighbors and the local device.
        Global locks for the specific location are used to prevent data races.

        Args:
            script (callable): The script (function or object with a run method) to execute.
            location (int): The data location relevant to this script.
        """
        # Acquire a global lock for the specific data location to ensure atomic access.
        with Device.locations[location]:
            script_data = []  # List to collect all relevant data for the script.
            
            # Gathers data from all neighboring devices for the current location.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Gathers data from its own device for the current location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # If any data was collected, run the script and update devices.
            if script_data: # Equivalent to script_data != []
                result = script.run(script_data)  # Execute the script.

                # Update the data in neighboring devices with the script's result.
                for device in self.device.neighbours:
                    device.set_data(location, result)
                # Update its own device's data with the script's result.
                self.device.set_data(location, result)


    def run(self):
        """
        The main execution loop for the DeviceThread.
        It continuously participates in global and device-local synchronizations,
        fetches neighbor information, and processes assigned scripts for timepoints.
        """
        while True:
            # Global synchronization barrier: All DeviceThreads (from all devices) wait here
            # until everyone is ready for the next timepoint.
            Device.barrier.wait()
            
            # Acquire script_lock to safely update working_scripts and flags for the new timepoint.
            with self.device.script_lock:
                if not self.device.working_scripts: # Check if working_scripts is empty.
                    # If empty, a new timepoint has likely started, so reset state.
                    self.device.working_scripts = list(self.device.scripts) # Reload all scripts.
                    self.device.timepoint_done = False # Reset timepoint completion flag.
                    self.device.neighbour_request = False # Reset neighbor fetch flag.

            # Device-local synchronization barrier: All auxiliary threads within this specific
            # device wait here until all of them are ready to start processing scripts for the timepoint.
            self.device.reinit_barrier.wait()
            
            # Acquire global devices_lock to safely update neighbor information.
            Device.devices_lock.acquire()
            if not self.device.neighbour_request: # Ensure neighbors are fetched only once per device.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbour_request = True
            Device.devices_lock.release()

            # If no neighbors are returned (e.g., simulation end signal from supervisor),
            # this device's threads can break their main loop.
            if self.device.neighbours is None:
                break

            # Loop to continuously fetch and run scripts assigned to this device.
            while True:
                self.device.script_lock.acquire()
                if self.device.working_scripts:
                    # If there are scripts to process, pop one.
                    (script, location) = self.device.working_scripts.pop()
                    self.device.script_lock.release()

                elif self.device.timepoint_done:
                    # If no scripts left and script assignment is done for this timepoint, break.
                    self.device.script_lock.release()
                    break

                else:
                    # If no scripts are available yet and assignment is not done, wait.
                    self.device.script_event.clear()      # Clear event before waiting.
                    self.device.script_lock.release()
                    self.device.script_event.wait()       # Wait until new scripts are assigned or timepoint done is signaled.
                    continue # Re-check conditions after waking up.
                
                # Execute the fetched script.
                if script is not None:
                    self.run_script(script, location)

