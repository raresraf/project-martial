"""
This module implements a device simulation framework that utilizes a thread pool
for script execution and a reusable barrier for synchronization. It defines:
- Device: Represents a simulated device managing sensor data and script assignments.
- DeviceThread: The main thread for a Device, coordinating script execution and data sharing.
- Worker: A worker thread responsible for processing assigned tasks from a ThreadPool.
- ThreadPool: Manages a pool of Worker threads to execute tasks concurrently.

The system uses Events, Locks, and Queues for inter-thread communication and data consistency.
"""


from threading import Event, Thread, Lock , Condition
from queue import Worker, ThreadPool
from reusable_barrier_semaphore import ReusableBarrier

class Device(object):
    """
    Represents a simulated device in a distributed system. Each device manages
    its sensor data, assigns and executes scripts using a thread pool,
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
        self.script_received = Event() # Event to signal when new scripts are assigned.
        self.wait_neighbours = Event() # Event to signal when neighbour information is available.
        self.scripts = [] # List to store assigned scripts.
        self.neighbours = [] # List to store neighboring devices.
        self.allDevices = [] # List of all devices in the simulation.
        self.locks = [] # List to store location-specific locks.
        self.pool = ThreadPool(8) # ThreadPool for executing scripts concurrently.
        self.lock = Lock() # Lock for protecting access to shared device attributes.
        self.thread = DeviceThread(self) # The main thread for this device.
        self.thread.start() # Start the main device thread.

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
        This includes setting up the global barrier and initializing location-specific locks.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        self.allDevices = devices # Store the list of all devices.
        self.barrier = ReusableBarrier(len(devices)) # Initialize a global barrier for all devices.

        # Block Logic: Initialize a fixed number of location-specific locks.
        for i in range(0, 50): # Assuming there are 50 locations (hardcoded).
            self.locks.append(Lock())

        pass # Placeholder for additional setup logic.

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific location.
        If a script is provided, it's added to the device's script list and
        then added as a task to the thread pool for execution.
        If no script is provided (None), it signals that scripts have been received for the timepoint.

        Args:
            script (Script or None): The script object to assign, or None to signal script reception.
            location (str): The location associated with the script.
        """
        
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
            # Block Logic: Add the script execution as a task to the thread pool.
            self.pool.add_task(self.executeScript,script,location)
        else:
            self.script_received.set() # Signal that all scripts for this timepoint have been assigned.

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            Any or None: The sensor data if available for the location, otherwise None.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

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

    def executeScript(self,script,location):
        """
        Executes a script for a given location, collecting data from neighbors and itself,
        then propagating the results. This method is designed to be run by the `ThreadPool`.

        Args:
            script (Script): The script object to be executed.
            location (str): The location associated with the script.
        """

        self.wait_neighbours.wait() # Block Logic: Wait until neighbour information is available.
        script_data = [] # List to collect data for the current script.

        # Block Logic: Collect data from neighboring devices, if available.
        if not self.neighbours is None:
            for device in self.neighbours:
                device.locks[location].acquire() # Acquire lock for neighbour device's location.
                data = device.get_data(location)
                device.locks[location].release() # Release lock.

                if data is not None:
                    script_data.append(data)

        self.locks[location].acquire() # Acquire lock for current device's location.
        data = self.get_data(location)
        self.locks[location].release() # Release lock.

        if data is not None:
            script_data.append(data)

        # Block Logic: If data was collected, execute the script and update devices.
        if script_data != []:
            result = script.run(script_data) # Inline: Execute the assigned script with collected data.

            # Block Logic: Propagate the script's result to neighboring devices, if available.
            if not self.neighbours is None:
                for device in self.neighbours:

                    device.locks[location].acquire() # Acquire lock for neighbour device's location.
                    device.set_data(location, result)
                    device.locks[location].release() # Release lock.

            self.locks[location].acquire() # Acquire lock for current device's location.
            self.set_data(location, result)
            self.locks[location].release() # Release lock.


class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for orchestrating the overall
    simulation workflow for that device. It fetches neighbor information,
    assigns scripts to the thread pool, and synchronizes with other devices.
    """
    

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent Device instance this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Executes the main logic of the DeviceThread.
        - Continuously fetches neighbor information from the supervisor.
        - If no neighbors are returned (signal for shutdown), it terminates the thread pool and breaks.
        - Signals that neighbor information is available.
        - Adds assigned scripts as tasks to the thread pool.
        - Waits for all tasks in the thread pool to complete.
        - Synchronizes with the global barrier after all scripts are processed.
        """

        # Block Logic: Main loop for continuous processing of timepoints.
        while True:
            self.device.script_received.clear() # Clear the script_received event for the next cycle.
            self.device.wait_neighbours.clear() # Clear the wait_neighbours event for the next cycle.

            self.device.neighbours = [] # Clear previous neighbours.
            # Block Logic: Fetch updated neighbor information from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.wait_neighbours.set() # Signal that neighbors are now available.

            # Block Logic: If no neighbors are returned, it's a shutdown signal.
            if self.device.neighbours is None:
                self.device.pool.wait_completion() # Wait for any remaining tasks to complete.
                self.device.pool.terminateWorkers() # Terminate all worker threads in the pool.
                self.device.pool.threadJoin() # Wait for worker threads to join.
                return # Exit the main loop.

            # Block Logic: Add all assigned scripts as tasks to the thread pool.
            for (script, location) in self.device.scripts:
                self.device.pool.add_task(self.device.executeScript,script,location)

            self.device.script_received.wait() # Block Logic: Wait for the Device to assign all scripts.
            self.device.pool.wait_completion() # Block Logic: Wait for all tasks in the thread pool to complete.

            # Block Logic: Synchronize all devices at the global barrier after all scripts are processed.
            for dev in self.device.allDevices:
                dev.barrier.wait()



from Queue import Queue
from threading import Thread

class Worker(Thread):
    """
    A worker thread that continuously fetches and executes tasks from a shared queue.
    It can be configured to run as a daemon and has a termination flag.
    """
    def __init__(self, tasks):
        """
        Initializes a Worker thread.

        Args:
            tasks (Queue): The shared queue from which the worker fetches tasks.
        """
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True # Set as daemon so it exits when the main program exits.
        self.terminate_worker = False # Flag to signal the worker to terminate.
        self.start() # Start the worker thread automatically.

    def run(self):
        """
        The main execution loop for the worker thread.
        It continuously fetches tasks from the `tasks` queue, executes them,
        and marks them as done. The worker terminates upon receiving a `None` function as a task.
        """
        # Block Logic: Main loop for continuous task processing.
        while True:
            func, args, kargs = self.tasks.get() # Block Logic: Fetch a task (function, args, kwargs) from the queue.
            # Block Logic: Check if a termination signal (None function) has been received.
            if func == None:
                self.tasks.task_done() # Mark the termination task as done.
                break # Exit the worker loop.
            try: func(*args, **kargs) # Block Logic: Execute the task function with its arguments.
            except Exception, e: print e # Inline: Print any exception that occurs during task execution.
            self.tasks.task_done() # Mark the current task as done in the queue.


class ThreadPool:
    """
    Manages a pool of worker threads. It initializes a specified number of `Worker`
    threads, provides a method to add tasks to their shared queue, and handles
    starting and stopping these workers.
    """
    def __init__(self, num_threads):
        """
        Initializes the ThreadPool.

        Args:
            num_threads (int): The number of worker threads to create in the pool.
        """
        self.tasks = Queue(99999) # Bounded queue to hold tasks.
        self.workers = [] # List to hold Worker thread instances.
        # Block Logic: Create and append `num_threads` Worker instances to the pool.
        for _ in range(num_threads):
            self.workers.append(Worker(self.tasks))

    def add_task(self, func, *args, **kargs):
        """
        Adds a new task to the queue for workers to process.

        Args:
            func (callable): The function to execute as a task.
            *args: Positional arguments for the function.
            **kargs: Keyword arguments for the function.
        """
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """
        Blocks until all tasks in the queue have been processed.
        """
        self.tasks.join()

    def terminateWorkers(self):
        """
        Sends termination signals to all worker threads.
        """
        # Block Logic: Add `None` tasks to the queue as stop signals for each worker.
        for worker in self.workers:
            worker.tasks.put([None,None,None]) # A task with None func acts as a shutdown signal.
            worker.terminate_worker = True # Set worker's internal termination flag.

    def threadJoin(self):
        """
        Waits for all worker threads in the pool to terminate.
        """
        for worker in self.workers:
            worker.join()

