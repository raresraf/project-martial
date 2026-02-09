"""
@file device.py
@brief This module defines the Device and DeviceThread classes for simulating a distributed sensor network.

The Device class represents an individual sensor device in the network, managing its sensor data,
communication with a supervisor, and execution of scripts. The DeviceThread class handles the
concurrent execution of data processing scripts for each device using a thread pool.

Algorithm: Distributed simulation with synchronized data processing using barriers and locks.
Data Flow: Sensor data is processed by scripts, potentially involving data exchange with neighboring devices.
Synchronization: Reusable barriers ensure all devices complete a timepoint before proceeding. Locks
manage concurrent access to shared sensor data locations.
"""

from threading import Lock, Thread, Event
from Queue import Queue
from barrier import ReusableBarrier


class Device(object):
    """
    @brief Represents a single sensor device in a distributed network.

    Each device manages its own sensor data, processes incoming scripts, and communicates
    with a central supervisor. It also coordinates with other devices through a shared barrier
    and uses locks for data consistency.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: Unique identifier for the device.
        @param sensor_data: Dictionary containing initial sensor readings for various locations.
        @param supervisor: Reference to the central supervisor managing the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when a new script has been assigned.
        self.scripts = list() # List to store incoming scripts and their associated locations.
        self.timepoint_done = Event() # Event to signal when the device has completed processing for a timepoint.
        self.thread = DeviceThread(self) # Dedicated thread for this device's operations.
        self.thread.start() # Starts the device's operational thread.
        self.barrier = None # Shared barrier for synchronizing timepoints across devices.
        self.lock_dict = None # Dictionary of locks for managing concurrent access to sensor data locations.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Initializes shared synchronization primitives for all devices.

        This method is designed to be called once by the device with device_id 0 to
        set up a common barrier and lock dictionary for all devices in the network.
        Pre-condition: This method should only be called by a designated master device (device_id == 0).
        """
        if self.device_id is 0: # Block Logic: Ensures that only the master device initializes shared resources.
            barrier = ReusableBarrier(len(devices)) # Creates a reusable barrier for N devices.
            lock_dict = dict() # Initializes a dictionary to hold locks for various data locations.

            # Block Logic: Assigns the shared barrier and lock dictionary to all devices.
            for device in devices:
                device.barrier = barrier
                device.lock_dict = lock_dict

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific sensor data location.

        If a script is provided, it's added to the device's script queue and a signal is set.
        If no script is provided (None), it signals that processing for the current timepoint is complete.

        @param script: The script object to be executed.
        @param location: The sensor data location associated with the script.
        """
        if script is not None: # Block Logic: Handles the assignment of a new script.
            self.scripts.append((script, location)) # Adds the script and its location to a list.
            self.script_received.set() # Signals that a new script has been received.
        else: # Block Logic: Handles the case where no script is provided, indicating timepoint completion.
            self.timepoint_done.set() # Signals that the timepoint processing for this device is finished.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key for the sensor data.
        @return: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a given location.
        @param location: The key for the sensor data.
        @param data: The new data value to set.
        """
        if location in self.sensor_data: # Block Logic: Ensures the location exists before updating.
            self.sensor_data[location] = data # Updates the sensor data at the specified location.

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread.
        """
        self.thread.join() # Waits for the device's thread to complete its execution.


class DeviceThread(Thread):
    """
    @brief Manages the concurrent execution of scripts for a single device.

    This thread maintains a queue for tasks (scripts) and a pool of worker threads
    to execute these tasks. It also coordinates with other devices using shared
    synchronization primitives.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id) # Initializes the base Thread class.
        self.device = device # Reference to the associated Device object.
        self.queue = Queue() # Queue for holding tasks (scripts to be executed).
        self.thread_pool = list() # List to store references to worker threads.

        self.thread_num = 8 # Configurable number of worker threads in the pool.
        
        # Block Logic: Populates the thread pool with worker threads.
        for _ in range(0, self.thread_num):
            my_thread = Thread(target=self.executor_service) # Creates a new worker thread targeting executor_service.
            my_thread.start() # Starts the worker thread.
            self.thread_pool.append(my_thread) # Adds the worker thread to the pool.

    def run(self):
        """
        @brief The main execution loop for the device's thread.

        This loop continuously fetches scripts from the supervisor, assigns them to the
        thread pool, and synchronizes with other devices at each timepoint.
        """
        while "Not finished": # Loop Invariant: Continues until a shutdown signal is received (neighbours is None).
            neighbours = self.device.supervisor.get_neighbours() # Retrieves information about neighboring devices from the supervisor.

            # Block Logic: Handles the shutdown condition for the device.
            if neighbours is None:
                # Block Logic: Signals all worker threads to shut down by enqueuing None.
                for _ in range(0, self.thread_num):
                    self.queue.put(None)
                self.shutdown() # Waits for worker threads to finish.
                self.thread_pool = list() # Clears the thread pool.
                break # Exits the main loop.

            # Block Logic: Waits until the device has received all scripts for the current timepoint.
            self.device.timepoint_done.wait()

            # Block Logic: Enqueues scripts for processing by worker threads.
            for (script, location) in self.device.scripts:
                queue_info = [script, location, neighbours] # Gathers necessary information for script execution.
                self.queue.put(queue_info) # Adds the task to the queue for processing.

            # Block Logic: Waits for all enqueued tasks (scripts) to be processed.
            self.queue.join()

            # Block Logic: Synchronizes with other devices at the end of the timepoint.
            self.device.barrier.wait()

            # Block Logic: Resets the timepoint_done event for the next iteration.
            self.device.timepoint_done.clear()


    def executor_service(self):
        """
        @brief Worker thread function that continuously processes tasks from the queue.
        """
        while "Not finished": # Loop Invariant: Continues until a None task is received (shutdown signal).
            tasks = self.queue.get() # Retrieves a task from the queue.

            # Block Logic: Handles the shutdown signal for the worker thread.
            if tasks is None:
                self.queue.task_done() # Signals that this task (None) is done.
                break # Exits the worker thread loop.
            else: # Block Logic: Processes a valid script task.
                script_t = tasks[0] # Extracts the script object.
                location_t = tasks[1] # Extracts the sensor data location.
                neighbours_t = tasks[2] # Extracts the list of neighbors.

            # Block Logic: Ensures a lock exists for the current data location, creating it if necessary.
            if self.device.lock_dict.get(location_t) is None:
                self.device.lock_dict[location_t] = Lock() # Inline: Creates a new lock for the specific location.

            # Block Logic: Acquires the lock to ensure exclusive access to the data location.
            self.device.lock_dict[location_t].acquire()

            # Block Logic: Processes the data using the assigned script.
            self.data_processing(self.device, script_t, location_t, neighbours_t)

            # Block Logic: Releases the lock, allowing other threads to access the data location.
            self.device.lock_dict[location_t].release()

            # Block Logic: Signals that the current task has been completed.
            self.queue.task_done()

    @classmethod
    def data_processing(cls, device, script, location, neighbours):
        """
        @brief Processes sensor data using a given script, potentially involving neighbor data.

        @param device: The current Device instance.
        @param script: The script to run for data processing.
        @param location: The sensor data location being processed.
        @param neighbours: A list of neighboring Device instances.
        """
        script_info = list() # Variable: Collects relevant sensor data for the script.
        # Block Logic: Gathers sensor data from neighboring devices.
        for i in range(0, len(neighbours)):
            data = neighbours[i].get_data(location) # Retrieves data from a neighbor at the specified location.
            if data: # Conditional: Checks if data was successfully retrieved.
                script_info.append(data) # Adds retrieved data to the script's input.

        data = device.get_data(location) # Retrieves the current device's own sensor data.
        # Conditional: Appends the device's own data if it exists.
        if data != None:
            script_info.append(data)

        if script_info: # Conditional: Proceeds only if there is data to process.
            result = script.run(script_info) # Action: Executes the script with collected data.
            send_info = [location, result] # Variable: Stores the updated data and its location.

            # Block Logic: Updates the sensor data on neighboring devices.
            for i in range(0, len(neighbours)):
                neighbours[i].set_data(send_info[0], send_info[1]) # Sets the data for a neighbor.

            # Block Logic: Updates the sensor data on the current device.
            device.set_data(send_info[0], send_info[1])


    def shutdown(self):
        """
        @brief Shuts down all worker threads in the thread pool.
        """
        # Block Logic: Joins each worker thread, ensuring they complete their current tasks before exiting.
        for i in range(0, self.thread_num):
            self.thread_pool[i].join() # Waits for a worker thread to terminate.