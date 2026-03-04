"""
This module implements a simulated distributed device system where each Device
runs scripts, interacts with neighbors, and synchronizes operations using a
reusable barrier. It leverages threading for concurrent execution of device logic
and script processing.
"""


from threading import Event, Lock, Thread, Condition


class ReusableBarrierCond:
    """
    A reusable barrier synchronization mechanism using a Condition variable.
    Allows multiple threads to wait until all participate before proceeding.
    """

    def __init__(self, num_threads):
        """
        Initializes the reusable barrier.

        Args:
            num_threads (int): The total number of threads expected to reach this barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()


    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all `num_threads`
        have also called `wait()`. Once all threads arrive, they are all released.
        The barrier then resets for future use.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # Block Logic: Checks if the current thread is the last one to reach the barrier.
        # Invariant: 'self.count_threads' accurately reflects the number of threads still expected.
        if self.count_threads == 0:
            self.cond.notify_all() # Functional Utility: Releases all waiting threads.

            self.count_threads = self.num_threads # Functional Utility: Resets the barrier counter for reuse.
        else:
            self.cond.wait() # Functional Utility: Causes the thread to wait until notified.
        self.cond.release()


class Device(object):
    """
    Represents a single device in the simulated distributed system.
    Each device has a unique ID, sensor data, interacts with a supervisor,
    and can run assigned scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing sensor readings at different locations.
            supervisor (Supervisor): A reference to the central supervisor managing device interactions.
        """

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when new scripts are assigned.
        self.scripts = [] # List to store assigned scripts and their locations.
        self.timepoint_done = Event() # Event to signal when processing for a timepoint is complete.
        self.thread = DeviceThread(self) # Dedicated thread for device's continuous operation.
        self.thread.start()

        self.master = None # Reference to the master device (for barrier synchronization).
        self.bariera = None # Barrier instance, initialized only for the master.
        self.lock = Lock() # Generic lock for device-specific critical sections.
        self.lacate = {} # Dictionary to store locks per location for data access synchronization.

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the master device and initializes the reusable barrier for all devices.

        Args:
            devices (list): A list of all Device instances in the system.
        """
        self.master = devices[0] # Assigns the first device in the list as the master.
        self.master.bariera = ReusableBarrierCond(len(devices)) # Functional Utility: Initializes the barrier with the total number of devices.


    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.
        If script is None, it signals the end of a timepoint.

        Args:
            script (object or None): The script object to execute, or None to signal completion.
            location (str): The data location relevant to the script.
        """
        # Block Logic: If a script is provided, add it to the list; otherwise, signal completion.
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set() # Functional Utility: Signals that the current timepoint is done.
            self.script_received.set() # Functional Utility: Signals that script assignments are complete for this cycle.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (str): The specific location for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location (str): The specific location to update.
            data (any): The new data value for the location.
        """
        # Block Logic: Updates data only if the location already exists in sensor_data.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device, waiting for its thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device, responsible for continuously
    processing scripts, interacting with neighbors, and synchronizing with
    other devices.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance that this thread controls.
        """

        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device # Reference to the associated Device instance.

    def run(self):
        """
        The main loop for the device thread. It continuously fetches neighbor
        information, waits for script assignments, executes scripts, and
        synchronizes with other devices using a barrier.
        """

        
        # Block Logic: Main operational loop for the device.
        # Invariant: Continues indefinitely until a specific break condition (neighbours is None).
        while True:
            # Functional Utility: Retrieves information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()


            # Block Logic: Exits the loop if no neighbors are returned (signaling system shutdown or completion).
            # Pre-condition: 'neighbours' being None indicates a termination signal.
            if neighbours is None:
                break

            # Functional Utility: Waits until scripts have been assigned for the current timepoint.
            self.device.script_received.wait()

            tlist = [] # List to hold threads executing scripts.
            
            # Block Logic: Iterates through each assigned script to prepare for execution.
            # Invariant: Each tuple contains ('script_object', 'location').
            for (script, location) in self.device.scripts:

                script_data = [] # Data collected for the current script's execution.
                
                # Block Logic: Collects data from neighboring devices for the current script and location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Functional Utility: Collects the device's own data for the script.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: If there is data to process, a new thread is created to run the script.
                if script_data != []:

                    my_thread = MyThread(script, script_data, location,
                    neighbours, self.device, self.device.master.lock,
                    self.device.master.lacate) # Functional Utility: Creates a new thread for script execution.
                    my_thread.start() # Functional Utility: Starts the script execution thread.
                    tlist.append(my_thread) # Adds the thread to a list for later joining.

            # Block Logic: Waits for all script execution threads to complete for the current timepoint.
            for my_thread in tlist:
                my_thread.join() # Functional Utility: Waits for the script thread to finish.

            # Functional Utility: All devices wait at the barrier until every device has finished script execution.
            self.device.master.bariera.wait()
            # Functional Utility: Waits until the 'timepoint_done' event is set, signaling end of timepoint processing.
            self.device.timepoint_done.wait()
            # Functional Utility: Clears the 'script_received' event for the next timepoint.
            self.device.script_received.clear()

class MyThread(Thread):
    """
    A worker thread responsible for executing a single script and updating
    sensor data on the own device and its neighbors. It uses a per-location
    lock to ensure data consistency during updates.
    """

    def __init__(self, script, script_data, location, neighbours,
                own_device, mlock, lacate):
        """
        Initializes a MyThread instance for script execution.

        Args:
            script (object): The script object to run.
            script_data (list): The collected data for the script.
            location (str): The data location relevant to the script.
            neighbours (list): A list of neighboring Device instances.
            own_device (Device): Reference to the current device.
            mlock (Lock): A master lock, likely for broader synchronization (unused in run).
            lacate (dict): A dictionary of locks, keyed by location, for data access.
        """

        Thread.__init__(self)
        self.script = script
        self.script_data = script_data
        self.location = location
        self.neighbours = neighbours
        self.own_device = own_device
        self.mlock = mlock
        self.lacate = lacate

    def run(self):
        """
        Executes the assigned script, updates relevant sensor data, and handles
        synchronization for data modifications.
        """
        # Functional Utility: Executes the script with the collected data.
        result = self.script.run(self.script_data)

        # Block Logic: Acquires a lock specific to the data location to ensure atomic updates.
        # Pre-condition: 'self.location' is a string key for 'self.lacate'.
        if self.location in self.lacate:
            self.lacate[self.location].acquire()
        else:
            # Block Logic: If no lock exists for this location, create one and acquire it.
            self.lacate[self.location] = Lock() # Functional Utility: Creates a new lock for the location.


            self.lacate[self.location].acquire() # Functional Utility: Acquires the newly created lock.

        
        # Block Logic: Updates the sensor data on all neighboring devices.
        # Invariant: 'device' is a Device instance from the neighbors list.
        for device in self.neighbours:
            device.set_data(self.location, result)
        # Functional Utility: Updates the sensor data on the current device.
        self.own_device.set_data(self.location, result)

        # Functional Utility: Releases the lock for the data location.
        self.lacate[self.location].release()
