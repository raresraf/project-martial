
"""
@47e1e42f-4307-4a1b-867d-76fac4c433a3/device.py
@brief This module defines the `Device` class, which simulates a node in a distributed
system, handling sensor data, script execution, and synchronization. It integrates
a `ThreadPool` for managing script execution across worker threads.
"""

from threading import Event, Thread, Lock

from reusable_barrier import ReusableBarrier
from thread_pool import ThreadPool

class Device(object):
    """
    Represents a single device in a simulated distributed system.

    Each device manages its own sensor data, processes scripts, and interacts
    with other devices and a supervisor for synchronization and data exchange.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary representing the sensor data this device holds,
                                where keys are locations and values are data.
            supervisor (object): An object representing the supervisor managing the devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to store assigned scripts for this timepoint.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's script assignments.
        self.thread = DeviceThread(self) # The main thread for this device's operations.
        self.thread.start() # Start the device's main operational thread.

        self.barrier = None # Will be set by the supervisor for global synchronization.

        self.locks = {} # Dictionary to store locks for each sensor data location.
        # Initialize a lock for each location in the sensor data to ensure thread-safe access.
        for location in sensor_data:
            self.locks[location] = Lock()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device's synchronization mechanisms (e.g., barrier) among all devices.

        This method is typically called by the supervisor to establish inter-device
        communication and synchronization.

        Args:
            devices (list): A list of all Device instances in the simulated system.
        """
        
        # Invariant: Each device needs to know the total number of devices for barrier initialization.
        nr_devices = len(devices)
        # Block Logic: Only the device with device_id 0 (the 'master' device) initializes the barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(nr_devices) # Initialize a reusable barrier for all devices.
            
            # The master device assigns the initialized barrier to all other devices.
            for device in devices:
                if device.device_id: # Avoid re-assigning to itself (device_id 0).
                    device.barrier = self.barrier
        

    def assign_script(self, script, location):
        """
        Assigns a script and its associated data location to the device for later execution.

        Args:
            script (object): The script object to be executed.
            location (int): The specific data location relevant to this script.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the device's script list.
        else:
            self.timepoint_done.set() # If no script is assigned, signal that this timepoint is done for this device.

    def get_data(self, location):
        """
        Retrieves data from the specified location in the device's sensor_data.

        Ensures thread-safe access to the data by acquiring a lock for the specific location.

        Args:
            location (any): The key or index of the data to retrieve.

        Returns:
            any: The data at the specified location, or None if the location does not exist.
        """
        if location in self.sensor_data:
            self.locks[location].acquire() # Acquire lock for the specific location to prevent race conditions.

        return self.sensor_data[location] if location in self.sensor_data else None


    def set_data(self, location, data):
        """
        Sets or updates data at the specified location in the device's sensor_data.

        This method also releases the lock for the specific location, assuming
        `get_data` or another method acquired it.

        Args:
            location (any): The key or index of the data to update.
            data (any): The new value for the data at the specified location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data # Update the data at the specified location.
            self.locks[location].release() # Release the lock for the specific location.

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device by joining its main thread.
        """
        self.thread.join() # Wait for the device's main thread to complete its execution.

class DeviceThread(Thread):
    """
    Manages the lifecycle and operations of a single Device, including
    fetching neighbors, handling script assignments, and synchronizing
    with other devices using a thread pool.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        # Initialize a ThreadPool for parallel execution of scripts.
        self.thread_pool = ThreadPool(8, self.device)

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously fetches neighboring devices, waits for new scripts
        to be assigned, dispatches these scripts to the thread pool, and
        then synchronizes with other devices via a barrier before
        resetting for the next timepoint.
        """
        while True:
            # Pre-condition: Fetch the current set of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # If no neighbors are returned (e.g., supervisor signals shutdown), break the loop.
                break

            # Block Logic: Wait until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()
            # Reset the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()

            # Block Logic: Remove the current device itself from the list of neighbors to avoid self-interaction.
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    neighbours.remove(device)
                    break

            # Block Logic: Submit each assigned script to the thread pool for asynchronous execution.
            for (script, location) in self.device.scripts:
                self.thread_pool.submit_task(neighbours, script, location)

            # Block Logic: Wait at the barrier until all devices have completed their script processing for this timepoint.
            self.device.barrier.wait()

        # Post-condition: After the main loop terminates, join all threads in the thread pool to ensure their completion.
        self.thread_pool.join_threads()


from threading import Thread
from Queue import Queue

class ThreadPool(object):
    """
    Manages a pool of worker threads to execute tasks concurrently.

    Tasks are submitted to a queue, and worker threads pick them up for processing.
    This helps in managing and reusing a fixed number of threads for script execution.
    """

    def __init__(self, threads_count, device):
        """
        Initializes the ThreadPool.

        Args:
            threads_count (int): The number of worker threads to create in the pool.
            device (Device): The Device instance associated with this thread pool.
        """
        self.threads = [] # List to hold the Thread objects.
        self.device = device # The associated Device instance.
        self.threads_count = threads_count # Number of threads in the pool.
        self.queue = Queue(threads_count) # A queue to hold tasks for the threads.

        self.create_threads(threads_count) # Immediately create and start the worker threads.

    def create_threads(self, threads_count):
        """
        Creates and starts the specified number of worker threads.

        Each thread will continuously execute tasks from the queue.

        Args:
            threads_count (int): The number of threads to create.
        """
        i = 0
        while i < threads_count:
            # Create a new thread targeting the `execute_task` method.
            thread = Thread(target=self.execute_task)
            self.threads.append(thread)
            i += 1

        # Start all created worker threads.
        for thread in self.threads:
            thread.start()

    def submit_task(self, neighbours, script, location):
        """
        Submits a new task to the thread pool's queue.

        Args:
            neighbours (list): List of neighboring devices.
            script (object): The script to be executed.
            location (int): The data location relevant to the script.
        """
        # Place the task (tuple of arguments) into the queue.
        self.queue.put((neighbours, script, location))

    def execute_task(self):
        """
        The main loop for each worker thread.

        Threads continuously retrieve tasks from the queue and execute them.
        The loop terminates upon receiving a special shutdown signal.
        """
        while True:
            # Retrieve a task from the queue; blocks if the queue is empty.
            elem = self.queue.get()
            neighbours = elem[0]
            script = elem[1]
            location = elem[2]

            # Block Logic: Check for a shutdown signal (None, None, None tuple).
            if neighbours is None and script is None and location is None:
                return # Terminate the thread.

            self.run_script(neighbours, script, location) # Execute the script.

    def run_script(self, neighbours, script, location):
        """
        Executes a given script, collecting data from neighbors and the local device,
        and then updates data based on the script's result.

        Args:
            neighbours (list): A list of neighboring Device instances.
            script (object): The script object to execute.
            location (int): The data location relevant to the script.
        """
        script_data = [] # Initialize a list to hold collected data for the script.

        # Block Logic: Collect data from neighboring devices.
        for device in neighbours:
            data = device.get_data(location) # Retrieve data from a neighbor.
            if data is not None:
                script_data.append(data) # Add valid data to the script's input.

        # Block Logic: Collect data from the current device.
        data = self.device.get_data(location) # Retrieve data from the current device.
        if data is not None:
            script_data.append(data) # Add valid data to the script's input.

        # Pre-condition: Check if there is any data to process.
        if script_data != []:
            # Action: Execute the script with the aggregated data.
            result = script.run(script_data)

            # Block Logic: Update data in neighboring devices.
            for device in neighbours:
                device.set_data(location, result) # Update neighbor's data with the script's result.

            # Block Logic: Update data in the current device.
            self.device.set_data(location, result) # Update the current device's data with the script's result.

    def join_threads(self):
        """
        Sends shutdown signals to all worker threads and waits for their completion.
        """
        i = 0
        while i < self.threads_count:
            # Send a shutdown signal (None, None, None) to each thread via the queue.
            self.submit_task(None, None, None)
            i += 1

        # Wait for all worker threads to finish executing and terminate.
        for thread in self.threads:
            thread.join()
