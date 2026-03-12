


"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `Device`: Represents a single device, managing its sensor data and orchestrating operations.
- `DeviceThread`: The main orchestrating thread for a `Device`, leveraging a `ThreadPool`
  to execute scripts concurrently.
- `ThreadPool`: A custom thread pool manager that uses `Queue` and `Thread` instances
  to distribute tasks.
- `Barrier`: A synchronization primitive (imported from `barrier` module) used for
  inter-device synchronization.

The system utilizes `threading.Event` for signaling and `threading.Lock` for protecting
location-specific data.
"""


from threading import Event, Thread, Lock
from barrier import Barrier
from threadpool import ThreadPool

class Device(object):

    """

    Represents a single device within a simulated distributed environment.

    Each device manages its own sensor data, communicates with a supervisor,

    and orchestrates multi-threaded script execution through its `DeviceThread`

    and associated `ThreadPool`. It uses per-location locks for data consistency.

    """



    def __init__(self, device_id, sensor_data, supervisor):

        """

        Initializes a Device instance.



        Args:

            device_id (int): A unique identifier for the device.

            sensor_data (dict): A dictionary containing initial sensor data for the device.

            supervisor (object): A reference to a supervisor object for inter-device communication.

        """

        

        self.device_id = device_id

        self.sensor_data = sensor_data

        self.supervisor = supervisor



        self.script_received = Event() # Event to signal when new scripts are assigned.

        self.scripts = [] # List to hold assigned scripts (tuples of (script, location)).

        self.timepoint_done = Event() # Event to signal that the current timepoint's processing is complete.

        self.thread = DeviceThread(self) # The main orchestrating thread for this device.

        self.thread.start() # Start the DeviceThread.



        self.barrier = None # Placeholder for the global Barrier, set in setup_devices.

        # Dictionary of `Lock` objects, one for each data location in `sensor_data`.

        self.locks = {location : Lock() for location in sensor_data}



        def __str__(self):



            """



            Returns a string representation of the device.



    



            Returns:



                str: A string in the format "Device <device_id>".



            """



            



            return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes a global barrier and distributes it to all devices,
        then starts their respective `DeviceThread`s. This method is designed to be
        called only by the device with `device_id == 0`.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        # Block Logic: Only the device with device_id 0 performs this setup.
        if self.device_id == 0:
            self.barrier = Barrier(len(devices)) # Inline: Creates a global Barrier for synchronization among all DeviceThreads.

            # Block Logic: Distributes the created global barrier to all other devices.
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier
                # Inline: Creates and starts the main orchestrating thread for each device.
                device.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals timepoint completion.
        If a script is provided, it's appended to the device's internal script list
        and `script_received` event is set. If `script` is None, `timepoint_done`
        is set, and `script_received` is also set (likely to release `DeviceThread`'s wait).

        Args:
            script (object): The script object to be executed, or None to signal end of assignments.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
            self.script_received.set() # Signal that a new script has been received.
        else:
            self.timepoint_done.set() # If script is None, signal that script assignments for the timepoint are done.
            self.script_received.set() # Also signal script_received to ensure DeviceThread is released if waiting.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's `sensor_data` dictionary.

        Args:
            location (int): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location in this device's `sensor_data` dictionary.
        The data is updated only if the location exists in the `sensor_data`.

        Args:
            location (int): The location identifier for which to set data.
            data (any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device by waiting for its main `DeviceThread` to complete.
        This method assumes `self.devices` is populated only for `device_id == 0`,
        and that device handles joining all threads.
        """
        self.thread.join() # Wait for the DeviceThread to finish its execution.


class DeviceThread(Thread):
    """
    The main orchestrating thread for a `Device`.
    This thread manages a `ThreadPool` to execute scripts concurrently,
    fetches neighbor information, submits scripts to the pool, and
    handles timepoint synchronization using a global barrier.
    """

    def __init__(self, device):
        """
        Initializes a `DeviceThread` instance.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(7, device) # Initializes a custom thread pool with 7 worker threads.

    def run(self):
        """
        The main execution loop for the `DeviceThread`.
        It continuously fetches neighbor data, waits for script assignments,
        submits scripts to its `ThreadPool` for concurrent execution,
        and synchronizes across devices using a global barrier.
        """
        while True:
            # Block Logic: Fetch neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Inline: If `neighbours` is None, it signals termination for the device.
            if neighbours is None:
                break # Exit the main loop, initiating the shutdown sequence.

            # Block Logic: Wait for `script_received` (scripts assigned) and `timepoint_done` (timepoint complete).
            # This ensures all necessary input for the current timepoint is ready.
            self.device.script_received.wait()
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Clear the event for the next timepoint.

            # Block Logic: Submit all assigned scripts to the `ThreadPool` for concurrent execution.
            for (script, location) in self.device.scripts:
                # `submit` wraps the script, location, and neighbors as a task for a worker thread.
                self.thread_pool.submit(neighbours, script, location)

            # Block Logic: Synchronize with other devices at the global barrier.
            # This ensures all devices have completed their script processing for the timepoint.
            self.device.barrier.wait()

        # Block Logic: When the main loop breaks (device shutdown), gracefully shutdown the thread pool.
        self.thread_pool.shutdown()


class ThreadPool(object):
    """
    A custom thread pool implementation for executing tasks concurrently.
    It manages a fixed number of worker `Thread` instances, distributing tasks
    (script execution requests) via a shared `Queue`. It provides mechanisms to
    submit tasks, wait for their completion, and gracefully shut down the worker threads.
    """
    
    def __init__(self, threads_count, device):
        """
        Initializes a `ThreadPool` with a specified number of worker threads.

        Args:
            threads_count (int): The number of worker `Thread` instances to create in the pool.
            device (Device): The parent `Device` object that owns this thread pool.
        """
        self.queue = Queue(threads_count) # A queue to hold tasks, bounded by the number of threads.
        self.threads = [] # List to hold references to the spawned worker threads.
        self.device = device # Reference to the parent `Device` (provides access to its locks, etc.).

        # Block Logic: Create and start `threads_count` worker threads.
        for _ in xrange(threads_count):
            new_thread = Thread(target=self.execute) # Each worker thread will run the `execute` method.
            self.threads.append(new_thread)
            new_thread.start()

    def execute(self):
        """
        The main loop for each worker thread in the pool.
        It continuously fetches tasks from the queue, executes them,
        and signals task completion. It terminates upon receiving a `None` task.
        """
        while True:
            now = self.queue.get() # Inline: Retrieve a task from the shared queue.
            if now is None: # Inline: `None` is a sentinel value indicating thread termination.
                self.queue.task_done() # Signal that this task (the None sentinel) is done.
                return # Terminate the worker thread.

            self.run_script(now) # Execute the script associated with the task.
            self.queue.task_done() # Signal that the current task has been completed.

    def run_script(self, script_env_data):
        """
        Executes a script using collected data from neighbors and the local device.
        This is the actual work performed by a worker thread for each submitted task.

        Args:
            script_env_data (tuple): A tuple containing (neighbours, script, location).
                                     - neighbours (list): List of neighboring devices.
                                     - script (object): The script object to execute.
                                     - location (int): The data location identifier.
        """
        neighbours, script, location = script_env_data # Unpack parameters.
        script_data = [] # List to collect input data for the script.

        # Block Logic: Collect data from all neighboring devices (excluding self) at the specified location.
        for device in neighbours:
            if device.device_id != self.device.device_id: # Avoid collecting from self here.
                # Inline: Acquire the neighbor's location-specific lock before accessing its data.
                # Only if the location is managed by that device.
                if location in device.sensor_data:
                    device.locks[location].acquire()
                data = device.get_data(location) # Get data from the neighbor.
                if data is not None:
                    script_data.append(data)
        
        # Block Logic: Collect data from this device's own sensor data.
        if location in self.device.sensor_data: # Acquire this device's location-specific lock.
            self.device.locks[location].acquire()
        data = self.device.get_data(location) # Get data from this device.
        if data is not None:
            script_data.append(data)

        # Block Logic: If input data is available, execute the script and update device data.
        if script_data != []:
            # Inline: Execute the script's `run` method with the collected data.
            result = script.run(script_data)

            # Block Logic: Update sensor data for all involved devices (neighbors and self) with the result.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result) # Update neighbor's data.
                    # Inline: Release neighbor's location-specific lock after updating its data.
                    if location in device.sensor_data:
                        device.locks[location].release()

            self.device.set_data(location, result) # Update this device's own data.
            if location in self.device.sensor_data: # Release this device's location-specific lock.
                self.device.locks[location].release()

    def submit(self, neighbours, script, location):
        """
        Submits a task (script execution request) to the thread pool's queue.

        Args:
            neighbours (list): List of neighboring devices.
            script (object): The script object to execute.
            location (int): The data location identifier.
        """
        self.queue.put((neighbours, script, location)) # Add the task as a tuple to the queue.

    def shutdown(self):
        """
        Waits for all tasks currently in the queue to be completed and then
        gracefully terminates all worker threads in the pool.
        """
        self.queue.join() # Inline: Blocks until all tasks in the queue are marked as done.

        # Block Logic: Add a `None` sentinel to the queue for each worker to signal termination.
        for _ in self.threads:
            self.queue.put(None)

        # Block Logic: Wait for all worker threads to fully terminate.
        for thread in self.threads:
            thread.join()
