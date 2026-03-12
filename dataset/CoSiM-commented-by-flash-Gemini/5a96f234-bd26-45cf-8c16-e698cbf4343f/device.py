




"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `Device`: Represents a single device, managing sensor data and orchestrating operations.
- `WorkPool`: A thread-safe pool for managing and distributing scripts/tasks to worker threads.
- `Worker`: A thread that fetches scripts from the `WorkPool`, executes them, and updates data.
- `DeviceThread`: The main thread for a `Device`, interacting with the `WorkPool` and
  managing overall device-level synchronization and neighbor information.

The system features concurrent execution of scripts within each device via `Worker` threads
managed by `WorkPool`, and inter-device synchronization using `ReusableBarrierCond`
from the `barrier` module. It also uses `threading.Event` and `threading.Lock` for various
signaling and data consistency purposes.
"""


from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond


class Device(object):


    """


    Represents a single device within a simulated distributed environment.


    Each device manages its own sensor data, communicates with a supervisor,


    and orchestrates multi-threaded script execution through its `DeviceThread`


    and `WorkPool`. It uses per-location locks for data consistency.


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


        self.thread = None # Placeholder for the DeviceThread, initialized in `setup_devices`.


        self.work_pool = None # Placeholder for the WorkPool, initialized in `setup_devices`.





        def __str__(self):





            """





            Returns a string representation of the device.





    





            Returns:





                str: A string in the format "Device <device_id>".





            """





            





            return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes a global barrier and shared location-specific locks,
        then distributes them to all devices and starts their `DeviceThread`s.
        This method is designed to be called only by the device with `device_id == 0`.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        if self.device_id == 0:
            # Inline: Creates a global ReusableBarrierCond for synchronization among all DeviceThreads.
            barrier = ReusableBarrierCond(len(devices))
            lock_locations = [] # List to hold shared locks for each data location.

            # Block Logic: Initializes a unique `Lock` for each distinct data location present across all devices.
            for device in devices:
                for _ in xrange(len(device.sensor_data)): # Assumes sensor_data keys are consecutive integers or mapped.
                                                           # This loop creates `num_locations` locks, not `num_locations` *per device*.
                    lock_locations.append(Lock())

            # Block Logic: Distributes the created global barrier and `lock_locations` to all devices
            # and starts their respective `DeviceThread`s.
            for device in devices:
                tasks_finish = Event() # Event to signal when all tasks for a timepoint are finished in the WorkPool.
                # Inline: Initializes the WorkPool for each device, providing it with the `tasks_finish` event
                # and the global `lock_locations`.
                device.work_pool = WorkPool(tasks_finish, lock_locations)

                # Inline: Creates and starts the main orchestrating thread for each device.
                device.thread = DeviceThread(device, barrier, tasks_finish)
                device.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals that script assignments are complete.
        If a script is provided, it's appended to the device's internal script list.
        If `script` is None, it signals that script assignments for the current
        timepoint are complete by setting `script_received`.

        Args:
            script (object): The script object to be executed, or None to signal end of assignments.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
        else:
            self.script_received.set() # If script is None, signal that script assignments are done for the timepoint.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's `sensor_data` dictionary.

        Args:
            location (int): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data \
                else None

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
        This also triggers the shutdown of the associated `WorkPool` and its worker threads.
        """
        self.thread.join() # Wait for the DeviceThread to finish its execution.

class WorkPool(object):
    """
    A thread-safe work pool designed to manage and distribute tasks (scripts)
    to a fixed number of `Worker` threads. It provides mechanisms for adding
    tasks, retrieving them, and signaling when all tasks for a timepoint are finished.
    """

    def __init__(self, tasks_finish, lock_locations):
        """
        Initializes the WorkPool.

        Args:
            tasks_finish (Event): An Event to signal when all tasks in the current
                                  batch have been processed by workers.
            lock_locations (list): A shared list of `Lock` objects, where each lock
                                   corresponds to a specific data location.
        """
        
        self.workers = [] # List to hold the spawned `Worker` threads.
        self.tasks = [] # List to hold the current batch of tasks (scripts).
        self.current_task_index = 0 # Index to track the next task to be given to a worker.
        self.lock_get_task = Lock() # Lock to ensure thread-safe access to `tasks` and `current_task_index`.
        self.work_to_do = Event() # Event to signal workers when there are tasks in the pool.
        self.tasks_finish = tasks_finish # Reference to the `tasks_finish` Event from the Device.
        self.lock_locations = lock_locations # Reference to the shared list of location locks.
        self.max_num_workers = 8 # Maximum number of worker threads in this pool.

        # Inline: Create and start `max_num_workers` Worker threads.
        for i in xrange(self.max_num_workers):
            self.workers.append(Worker(self, self.lock_get_task, \
                    self.work_to_do, self.lock_locations))
            self.workers[i].start()

    def set_tasks(self, tasks):
        """
        Sets a new batch of tasks for the work pool and signals workers that there is work to do.

        Args:
            tasks (list): A list of task objects (tuples of (script, location, neighbours, self_device)).
        """
        self.tasks = tasks # Assign the new list of tasks.
        self.current_task_index = 0 # Reset the task index to the beginning of the new batch.

        self.work_to_do.set() # Signal all waiting workers that there is work available.

    def get_task(self):
        """
        Thread-safely retrieves a single task from the work pool.

        Returns:
            tuple or None: A task tuple (script, location, neighbours, self_device) if available,
                           or None if all tasks have been distributed.
        """
        # Block Logic: Check if there are tasks remaining in the current batch.
        if self.current_task_index < len(self.tasks):
            task = self.tasks[self.current_task_index] # Get the current task.

            self.current_task_index = self.current_task_index + 1 # Advance to the next task.

            # Inline: If all tasks have now been retrieved, clear `work_to_do` and signal `tasks_finish`.
            if self.current_task_index == len(self.tasks):
                self.work_to_do.clear() # Clear the event as no more tasks are immediately available.
                self.tasks_finish.set() # Signal that all tasks have been taken from the pool.

            return task
        else: # All tasks have been distributed.
            return None

    def close(self):
        """
        Signals to all workers that the work pool is permanently closed and no more tasks will be added.
        It does this by making `get_task` return None, which workers interpret as a termination signal.
        Then, it waits for all worker threads to join.
        """
        self.tasks = [] # Clear any remaining tasks.
        self.current_task_index = len(self.tasks) # Set index to indicate no more tasks.

        # Inline: Signal workers to wake up and check for termination.
        self.work_to_do.set()

        # Inline: Wait for all worker threads to terminate.
        for worker in self.workers:
            worker.join()

class Worker(Thread):
    """
    A worker thread that fetches tasks from a shared `WorkPool`.
    For each task, it collects data from the device and its neighbors,
    executes the script, and updates the devices' sensor data,
    using location-specific locks managed by the `WorkPool`.
    """

    def __init__(self, work_pool, lock_get_task, work_to_do, lock_locations):
        """
        Initializes a `Worker` thread.

        Args:
            work_pool (WorkPool): Reference to the parent `WorkPool` instance.
            lock_get_task (Lock): The lock used to protect `WorkPool.get_task()` calls.
            work_to_do (Event): Event to signal when there is work available in the `WorkPool`.
            lock_locations (list): A shared list of `Lock` objects for data locations.
        """
        Thread.__init__(self)
        self.work_pool = work_pool
        self.lock_get_task = lock_get_task
        self.work_to_do = work_to_do
        self.lock_locations = lock_locations

    def run(self):
        """
        The main execution method for the `Worker` thread.
        It continuously fetches tasks from the `WorkPool`. Upon receiving a task,
        it collects data from the device and its neighbors, executes the script,
        updates the devices' sensor data, and handles thread-safe access to tasks
        and location data. If a `None` task is received, the thread terminates.
        """
        while True:
            # Block Logic: Acquire lock to safely get a task from the workpool.
            self.lock_get_task.acquire()

            # Block Logic: Wait for `work_to_do` event to be set, indicating tasks are available.
            self.work_to_do.wait()

            task = self.work_pool.get_task() # Retrieve a task from the workpool.

            # Inline: Release the lock after getting the task.
            self.lock_get_task.release()

            # Inline: If `get_task()` returns None, it signals the permanent end of work.
            if task is None:
                break # Terminate the worker thread.

            # Block Logic: Unpack the task components.
            script = task[0] # The script object to execute.
            location = task[1] # The data location.
            neighbours = task[2] # List of neighboring devices.
            self_device = task[3] # Reference to the worker's parent device.

            # Block Logic: Acquire the location-specific lock to ensure exclusive access
            # to data at this `location` across all devices during script execution and data update.
            self.lock_locations[location].acquire()

            script_data = [] # List to collect input data for the script.

            # Block Logic: Collect data from all neighboring devices at the specified location.
            for device in neighbours:
                data = device.get_data(location) # Get data from the neighbor.
                if data is not None:
                    script_data.append(data) # Add to script input if available.
            
            # Block Logic: Collect data from this worker's own parent device at the specified location.
            data = self_device.get_data(location)
            if data is not None:
                script_data.append(data) # Add to script input if available.

            # Block Logic: If input data is available, execute the script and update device data.
            if script_data != []:
                # Inline: Execute the script's `run` method with the collected data.
                result = script.run(script_data)

                # Block Logic: Update sensor data for all involved devices (neighbors and self) with the result.
                for device in neighbours:
                    device.set_data(location, result) # Update neighbor's data.
                
                self_device.set_data(location, result) # Update this device's own data.

            # Inline: Release the location-specific lock.
            self.lock_locations[location].release()





class DeviceThread(Thread):
    """
    The main orchestrating thread for a `Device`.
    This thread manages the device's `WorkPool`, distributing scripts for concurrent
    execution by `Worker` threads. It also fetches neighbor information from the supervisor
    and handles global timepoint synchronization using a `ReusableBarrierCond`.
    """

    def __init__(self, device, barrier, tasks_finish):
        """
        Initializes a `DeviceThread` instance.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
            barrier (ReusableBarrierCond): The global barrier for inter-device synchronization.
            tasks_finish (Event): An Event to signal when all tasks in the WorkPool
                                  for a timepoint have been completed.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.tasks_finish = tasks_finish

    def run(self):
        """
        The main execution loop for the `DeviceThread`.
        It continuously orchestrates the processing of scripts in timepoints,
        fetching neighbor data, distributing scripts to the `WorkPool`,
        and synchronizing across devices using a global barrier.
        """
        while True:
            # Block Logic: Synchronize with other devices at the global barrier.
            # This marks the beginning of a new timepoint for all devices.
            self.barrier.wait()

            # Block Logic: Fetch neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Inline: If `neighbours` is None, it signals termination for the device.
            if neighbours is None:
                self.device.work_pool.close() # Signal the workpool to shut down its workers.
                break # Exit the main loop, terminating the DeviceThread.

            # Block Logic: Wait for new scripts to be assigned to this device.
            self.device.script_received.wait()

            tasks = [] # List to hold tasks to be submitted to the workpool.
            # Block Logic: Prepare tasks for the workpool from the device's assigned scripts.
            # Each task includes the script, location, neighbor list, and a reference to this device.
            for (script, location) in self.device.scripts:
                tasks.append((script, location, neighbours, self.device))

            # Block Logic: If there are tasks for the current timepoint, submit them to the workpool.
            if tasks != []:
                self.device.work_pool.set_tasks(tasks) # Load tasks into the workpool and signal workers.

                # Block Logic: Wait for all tasks in the current batch to be completed by workers.
                self.tasks_finish.wait()

                self.tasks_finish.clear() # Clear the event for the next timepoint.

            # Inline: Clear the `script_received` event for the next timepoint.
            self.device.script_received.clear()
