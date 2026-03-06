"""
This module implements a device simulation framework that utilizes a thread pool
for script execution and a reusable barrier for synchronization. It defines:
- Device: Represents a simulated device managing sensor data and script assignments.
- DeviceThread: The main thread for a Device, coordinating script execution and data sharing.
- ScriptThread: Worker threads responsible for executing individual scripts.
- MyObjects: A simple data structure to pass job details to ScriptThreads.

The system uses events, locks, and a Queue for inter-thread communication and data consistency.
"""


from threading import Event, Thread, Lock
from reusable_barrier_semaphore import ReusableBarrier
import Queue
NUMBER_OF_THREADS = 8

class Device(object):
    """
    Represents a simulated device in a distributed system. Each device manages
    its sensor data, assigns and executes scripts using a pool of worker threads,
    and interacts with a supervisor. It coordinates with other devices through
    a global barrier.
    """
    

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
        self.script_received = Event() # Event to signal when new scripts are assigned
        self.scripts = [] # List to store assigned scripts (script, location) tuples
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing
        self.thread = DeviceThread(self) # The main thread for this device
        self.thread.start() # Start the main device thread
        self.barrier = None # Global barrier for device synchronization (initialized by setup_devices)
        self.data_lock = Lock() # Lock to protect access to sensor_data

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the global barrier for synchronizing DeviceThreads across all devices.
        This method is typically called once by a supervisor or a designated master device
        to initialize and distribute the shared barrier object.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        
        barrier = ReusableBarrier(len(devices)) # Create a new barrier with the total number of devices

        # Block Logic: If the device's barrier is not yet set, initialize it with the new barrier.
        if self.barrier is None:
            self.barrier = barrier

        # Block Logic: Propagate the initialized barrier to all other devices that don't have one.
        for device in devices:

            if device.barrier is None:
                device.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific location.
        If a script is provided, it's added to the device's script list and
        the `script_received` event is set. If no script is provided (None),
        it signals that a timepoint is done.

        Args:
            script (Script or None): The script object to assign, or None to signal timepoint completion.
            location (str): The location associated with the script.
        """
        
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list
            self.script_received.set()              # Signal that a new script is available
        else:
            self.timepoint_done.set()               # Signal that processing for the current timepoint is complete

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            Any or None: The sensor data if available for the location, otherwise None.
        """
        
        return self.sensor_data[location] if location in self.sensor_data \
            else None

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
        Shuts down the device by joining its main device thread.
        """
        
        self.thread.join()

class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for orchestrating the overall
    simulation workflow. It manages a pool of `ScriptThread` workers, fetches
    neighbor information, distributes scripts, and synchronizes with a global barrier.
    """
    
    location_locks = {} # Class-level dictionary to store locks for different locations

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent Device instance this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = [] # List to hold ScriptThread instances
        self.scripts_queue = Queue.Queue() # Queue for jobs to be processed by ScriptThreads

    def run(self):
        """
        Executes the main logic of the DeviceThread.
        - Initializes and starts `NUMBER_OF_THREADS` ScriptThread workers.
        - Continuously fetches neighbor information from the supervisor.
        - If no neighbors are returned (signal for shutdown), it sends stop signals to workers and breaks.
        - Waits for scripts to be assigned for the current timepoint.
        - Puts script execution jobs into the queue for worker threads.
        - Clears the `timepoint_done` event for the next cycle.
        - Synchronizes with the global barrier after all scripts are processed.
        """
        
        # Block Logic: Initialize and start worker threads.
        # Invariant: Each DeviceThread manages a fixed number of ScriptThread workers.
        for _ in range(NUMBER_OF_THREADS):
            self.threads.append(ScriptThread(self.scripts_queue))

        for script_thread in self.threads:
            script_thread.start()

        # Block Logic: Main loop for continuous processing of timepoints.
        while True:
            
            # Block Logic: Fetch updated neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: If no neighbors are returned, it's a shutdown signal.
            if neighbours is None:
                # Block Logic: Send stop signals to all worker threads.
                for script_thread in self.threads:
                    self.scripts_queue.put(MyObjects(None, None, None, None,
                                                     False, None)) # MyObjects with stop=False acts as shutdown.
                break # Exit the main loop

            # Block Logic: Wait for scripts to be assigned for the current timepoint.
            self.device.timepoint_done.wait()
            # Block Logic: Iterate through assigned scripts and put them into the queue for workers.
            for (script, location) in self.device.scripts:

                # Block Logic: If a lock for the current location doesn't exist, create one.
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()

                # Block Logic: Put a new job (MyObjects instance) into the scripts queue.
                self.scripts_queue.put(MyObjects(self.device, location, script,
                                                 neighbours, True,
                                                 self.location_locks),
                                       block=True, timeout=None)
            self.device.timepoint_done.clear() # Clear the timepoint_done event for the next cycle.

            # Block Logic: Synchronize all devices at the global barrier after all scripts are processed.
            self.device.barrier.wait()

        # Block Logic: Wait for all worker threads to finish.
        for script_thread in self.threads:
            script_thread.join()

class ScriptThread(Thread):
    """
    A worker thread that continuously fetches and executes scripts from a shared queue.
    It retrieves data from devices, runs the assigned script, and updates data on relevant devices.
    """
    

    def __init__(self, queue):
        """
        Initializes a ScriptThread.

        Args:
            queue (Queue.Queue): The shared queue from which the worker fetches `MyObjects` jobs.
        """
        
        Thread.__init__(self, name="Script Thread")
        self.queue = queue

    def run(self):
        """
        The main execution loop for the script worker thread.
        It continuously fetches `MyObjects` jobs from the queue, executes the script
        within the job, updates data on devices, and releases location locks.
        The worker terminates if it receives a `MyObjects` with `stop` attribute set to False.
        """
        
        # Block Logic: Main loop for continuous processing of script jobs.
        while True:

            # Block Logic: Fetch a job (`MyObjects` instance) from the queue.
            my_objects = self.queue.get(block=True, timeout=None)

            # Block Logic: Check for a shutdown signal (`stop` attribute set to False).
            if my_objects.stop == False:
                break # Exit the worker loop

            # Block Logic: Acquire a lock for the specific location to prevent race conditions during data access.
            my_objects.location_locks[my_objects.location].acquire()

            script_data = []
            # Block Logic: Collect data from neighboring devices specified in the job.
            for device in my_objects.neighbours:
                data = device.get_data(my_objects.location)
                if data is not None:
                    script_data.append(data)

            # Block Logic: Acquire data_lock before accessing parent device's data.
            my_objects.device.data_lock.acquire()
            # Block Logic: Collect data from the current device itself for the job's location.
            data = my_objects.device.get_data(my_objects.location)
            my_objects.device.data_lock.release() # Release data_lock after accessing data.

            if data is not None:
                script_data.append(data)

            # Block Logic: If data was collected, execute the script and update devices.
            if script_data != []:
                # Inline: Execute the assigned script with collected data.
                result = my_objects.script.run(script_data)

                # Block Logic: Propagate the script's result to neighboring devices.
                for device in my_objects.neighbours:
                    device.data_lock.acquire() # Acquire data_lock for neighbor device.
                    device.set_data(my_objects.location, result)
                    device.data_lock.release() # Release data_lock for neighbor device.

                # Block Logic: Update the current device's sensor data with the script's result.
                my_objects.device.data_lock.acquire() # Acquire data_lock for parent device.
                my_objects.device.set_data(my_objects.location, result)
                my_objects.device.data_lock.release() # Release data_lock for parent device.

            # Block Logic: Release the lock for the current location after processing.
            my_objects.location_locks[my_objects.location].release()

class MyObjects():
    """
    A simple data class used to encapsulate job details passed to worker threads.
    It holds references to the device, script, location, neighbors, and synchronization objects.
    """
    

    def __init__(self, device, location, script, neighbours, stop, location_locks):
        """
        Initializes a MyObjects instance.

        Args:
            device (Device): The parent Device instance.
            location (str): The location associated with the script.
            script (Script): The script object to be executed.
            neighbours (list): A list of neighboring Device instances relevant to this job.
            stop (bool): A flag indicating whether the worker thread should stop after processing this job.
            location_locks (dict): A dictionary of location-specific locks.
        """
        
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours
        self.stop = stop

        # Dictionary of location-specific locks, shared across all ScriptThreads
        self.location_locks = location_locks