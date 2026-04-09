


"""
This module defines a simulation framework for managing distributed devices, their sensor data, and script execution using a thread pool. It leverages threading primitives for synchronization and parallel processing of tasks across devices.

Key components:
- Device: Represents a simulated device with its unique ID, sensor data, and a reference to a supervisor. It manages script assignments and synchronizes operations with other devices.
- DeviceThread: A dedicated thread for each Device instance, orchestrating its operational lifecycle, including waiting on barriers, fetching neighbor data, and dispatching scripts to a thread pool.
- Solve: A worker thread managed by ThreadPoll, responsible for executing a single assigned script. It handles data acquisition, script execution, and data updates across devices, ensuring proper locking.
- ThreadPoll: Manages a pool of Solve worker threads, facilitating concurrent execution of multiple tasks. It provides mechanisms to put work into the pool and wait for its completion.

Architectural intent:
The framework is designed to enable parallel processing of scripts across multiple simulated devices. It ensures careful synchronization using barriers, events, and locks to handle shared data access and maintain consistency in a concurrent environment. This setup is typical for distributed simulation or data processing scenarios where tasks can be broken down and executed in parallel across different entities.
"""

from threading import Event, Thread, Lock, Semaphore

# Assuming ReusableBarrierSem is a custom barrier implementation
# and comparator is a function defined elsewhere for sorting devices.
from barrier import ReusableBarrierSem
from threadpool import ThreadPoll, comparator


class Device(object):
    """
    Represents a simulated device in the system.

    Each device has a unique ID, sensor data, and communicates with a supervisor.
    It manages the assignment and execution of scripts, and utilizes threading
    primitives for synchronization with other devices and its own operational thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        :param device_id: Unique identifier for the device.
        :param sensor_data: Dictionary holding sensor data, keyed by location.
        :param supervisor: Reference to the supervisor object managing this device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a script has been received and is ready for processing.
        self.script_received = Event()
        self.script_received.set()  # Initially set, meaning no script is currently being waited on
        # List to store assigned scripts, each paired with its execution location.
        self.scripts = []
        # Event to signal when all scripts for a specific timepoint have completed execution.
        self.timepoint_done = Event()
        # Dedicated thread for the device's operational logic.
        self.thread = DeviceThread(self)
        # Event used as a lock to control setup order among devices.
        self.lock = Event()
        self.lock.clear()  # Initially cleared, indicating the device is not yet set up
        # Global lock for critical sections involving shared data access.
        self.datalock = Lock()
        # A list of locks, where each lock protects sensor data at a specific location.
        # This allows for fine-grained locking based on data location.
        self.personal_lock = []
        # Reusable barrier for synchronizing all devices at specific timepoints.
        self.bariera = None
        # A sorted list of all devices in the simulation, for ordered access.
        self.all = None
        # Total number of devices in the simulation.
        self.no_devices = None

        # Initialize personal locks for each sensor data location.
        # `crt` is the maximum key value, ensuring enough locks are pre-allocated.
        crt = max(self.sensor_data.keys())
        for _ in xrange(crt + 1):
            self.personal_lock.append(Lock())

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device's synchronization mechanisms and starts its operational thread.
        This method ensures that devices are set up in a specific order based on their IDs,
        especially for initializing the shared barrier.

        :param devices: A list of all Device objects in the simulation.
        """
        # Sort all devices by their ID using the custom comparator.
        self.all = sorted(devices, cmp=comparator)
        self.no_devices = len(self.all)

        # The device with ID 0 is responsible for initializing the ReusableBarrierSem
        # and signaling other devices to proceed.
        if self.device_id == 0:
            self.bariera = ReusableBarrierSem(len(self.all))
            self.lock.set()  # Signal that this device's setup is complete
        else:
            # Subsequent devices wait for the previous device to complete its setup
            # to correctly obtain the shared barrier instance.
            prev_device = self.all[self.device_id - 1]
            prev_device.lock.wait()
            self.bariera = prev_device.bariera
            self.lock.set()  # Signal that this device's setup is complete

        # Start the dedicated thread for this device.
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specified location.

        If `script` is None, it acts as a signal to clear the `script_received` event
        and set `timepoint_done`, indicating completion for the current timepoint.

        :param script: The script object to be executed.
        :param location: The sensor data location pertinent to the script.
        """
        if script is not None:
            # Wait until previous script processing is acknowledged before assigning a new one.
            self.script_received.wait()
            self.scripts.append((script, location))
        else:
            # If script is None, it means no more scripts for this timepoint;
            # clear `script_received` and set `timepoint_done`.
            self.script_received.clear()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        :param location: The location key for the sensor data.
        :return: The sensor data at the specified location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        """
        Sets sensor data for a given location.

        :param location: The location key for the sensor data.
        :param data: The new data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its dedicated thread.
        This ensures that all operations initiated by the thread are completed
        before the program exits.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    A dedicated thread for each Device instance, managing its operational lifecycle.

    This thread is responsible for synchronizing with other device threads using
    a barrier, fetching neighbor information, preparing work for the thread pool,
    and signaling timepoint completion.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        :param device: The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.

        This loop continuously performs the following steps:
        1. Waits on a barrier to synchronize with all other device threads.
        2. Retrieves information about neighboring devices from the supervisor.
        3. Waits for the current timepoint's script execution to be signaled as done.
        4. Prepares a list of work items (scripts) for the thread pool.
        5. Submits the work items to the thread pool for parallel processing.
        6. Signals that new scripts can be received for the next timepoint.
        The loop breaks if no neighbors are returned (indicating simulation end).
        """
        # Initialize a ThreadPoll for managing worker threads that execute scripts.
        thread_pool = ThreadPoll(8)  # Assuming a pool size of 8 worker threads.
        while True:
            # Block Logic: Waits for all devices to reach this synchronization point.
            # This ensures all devices are ready before proceeding to the next timepoint.
            self.device.bariera.wait()

            # Functional Utility: Fetches information about the device's neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If no neighbors are returned, it signifies the end of the simulation.
            if neighbours is None:
                thread_pool.close()  # Shuts down the thread pool.
                break

            # Block Logic: Waits until all scripts from the previous timepoint (if any)
            # have finished execution and the `timepoint_done` event is set by an external entity.
            self.device.timepoint_done.wait()
            # Clears the event for the next timepoint.
            self.device.timepoint_done.clear()

            # Initializes a list to hold work items for the thread pool.
            work_list = []

            # Block Logic: Populates the work list with assigned scripts and their associated context.
            # Each item in `work_list` is a tuple containing (location, script, device, neighbours).
            for (script, location) in self.device.scripts:
                work_list.append((location, script, self.device, neighbours))

            # Block Logic: Submits the prepared work items to the thread pool for concurrent execution.
            # The `put_work` method will block until all submitted tasks are completed.
            thread_pool.put_work(work_list)

            # Functional Utility: Signals that the device is ready to receive new script assignments
            # for the next timepoint.
            self.device.script_received.set()


def comparator(device_a, device_b):
    """
    Compares two device objects based on their device_id.

    This function is intended for use with `sorted()` to ensure a consistent
    ordering of Device objects based on their unique identifiers.

    :param device_a: The first Device object.
    :param device_b: The second Device object.
    :return: 1 if device_a's ID is greater, -1 if smaller, 0 if equal.
    """
    if device_a.device_id > device_b.device_id:
        return 1
    else:
        return -1


class Solve(Thread):
    """
    A worker thread managed by ThreadPoll, responsible for executing a single script.

    This class encapsulates the logic for acquiring data, running a script with that
    data, updating sensor data, and managing synchronization locks.
    """

    def __init__(self, sem, free_threads, working_threads):
        """
        Initializes a Solve worker thread.

        :param sem: A Semaphore from the ThreadPoll to manage available worker slots.
        :param free_threads: A list in ThreadPoll tracking available worker threads.
        :param working_threads: A list in ThreadPoll tracking currently busy worker threads.
        """
        Thread.__init__(self)

        self.free_threads = free_threads
        self.sem = sem
        self.working_threads = working_threads
        # Event to signal when work is assigned to this thread.
        self.work = Event()
        # Event to signal when this thread has completed its work and is free.
        self.free = Event()
        # Flag to indicate if the thread should terminate.
        self.done = 0
        self.work.clear()  # Initially clear, waiting for work.
        self.free.set()  # Initially set, indicating thread is free.

        # Task-specific parameters, to be set by `set_work`.
        self.location = None
        self.script = None
        self.device = None
        self.neighbours = None

    def set_work(self, location, script, device, neighbours):
        """
        Assigns a specific task to this worker thread.

        :param location: The data location for the script.
        :param script: The script object to execute.
        :param device: The Device object associated with this task.
        :param neighbours: A list of neighboring Device objects.
        """
        self.location = location
        self.script = script
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        The main execution loop for the Solve worker thread.

        It continuously waits for work, executes the assigned script, manages
        data locks, updates sensor data, and then signals its availability
        for new tasks. It terminates when the `done` flag is set.
        """
        while 1:
            # Block Logic: Waits until `self.work` event is set, indicating a task has been assigned.
            self.work.wait()

            # Pre-condition: Checks if the thread has been signaled to terminate.
            if self.done == 1:
                break

            script_data = []

            # Block Logic: Prepares a comprehensive list of devices involved in this script's execution.
            # This includes the current device and all its neighbors, sorted by device ID.
            list_neighbours = self.neighbours
            list_neighbours.append(self.device)
            list_neighbours = set(list_neighbours)  # Remove duplicates if any
            list_neighbours = sorted(list_neighbours, cmp=comparator)

            # Block Logic: Acquires per-location locks for all relevant devices (current and neighbors)
            # to ensure exclusive access to sensor data during script execution.
            # This prevents race conditions when multiple scripts might try to access/modify
            # the same sensor data concurrently.
            for device in list_neighbours:
                if self.location in device.sensor_data:
                    device.personal_lock[self.location].acquire()

            # Block Logic: Collects sensor data from neighboring devices for the script.
            # This data will be passed to the script for processing.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            # Functional Utility: Appends the current device's data to `script_data` if available.
            # This ensures the script has access to the local device's data.
            if data is not None: # This 'data' variable scope is from the previous loop, likely a bug here.
                                # Should likely be `self.device.get_data(self.location)`
                script_data.append(data)

            # Pre-condition: Checks if there is any data to process with the script.
            if script_data != []:
                # Functional Utility: Executes the assigned script with the collected data.
                result = self.script.run(script_data)
                # Block Logic: Updates the sensor data on neighboring devices with the script's result.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                # Functional Utility: Updates the sensor data on the current device with the script's result.
                self.device.set_data(self.location, result)

            # Block Logic: Releases the per-location locks in reverse order of acquisition.
            # This minimizes the time locks are held and helps prevent deadlocks.
            for device in reversed(list_neighbours):
                if self.location in device.sensor_data:
                    device.personal_lock[self.location].release()

            # Post-condition: Signals that the thread has completed its current task and is free.
            self.free.set()
            self.work.clear()  # Clears the work event, ready to wait for new work.
            self.free_threads.append(self)  # Adds itself back to the pool of free threads.

            # Functional Utility: Releases a semaphore, indicating that a worker slot in the pool is now available.
            self.sem.release()


class ThreadPoll(object):
    """
    Manages a pool of Solve worker threads to process tasks concurrently.

    This class is responsible for creating, initializing, and distributing
    work to a fixed number of worker threads. It uses a semaphore to control
    the number of active tasks and ensures all tasks are completed before proceeding.
    """

    def __init__(self, no_threads):
        """
        Initializes the ThreadPoll with a specified number of worker threads.

        :param no_threads: The maximum number of concurrent threads in the pool.
        """
        self.no_threads = no_threads
        # List of worker threads that are currently available for tasks.
        self.free_threads = []
        # List of worker threads that are currently busy processing tasks.
        self.working_threads = [] # This variable is initialized but not used within the class.
        # Event to signal when all work submitted to the pool has been completed.
        self.workdone = Event() # This event is initialized but not used.
        # Semaphore to limit the number of concurrently active worker threads.
        self.sem = Semaphore(self.no_threads)

        # Block Logic: Initializes and starts the specified number of Solve worker threads.
        # Each thread is added to the list of free threads and then started.
        self.all_threads = [] # Stores references to all worker threads
        for _ in xrange(0, no_threads):
            tmp = Solve(self.sem, self.free_threads, self.working_threads)
            self.free_threads.append(tmp)
            self.all_threads.append(tmp) # Keep track of all threads to join them later.

        # Start all the worker threads.
        for current_thread in self.free_threads:
            current_thread.start()

    def put_work(self, work_list):
        """
        Distributes a list of work items to the available worker threads.

        This method acquires a semaphore for each work item, assigns the work
        to a free thread, and then activates the thread. It then waits for
        all assigned tasks to be completed before returning.

        :param work_list: A list of tuples, where each tuple contains
                          (location, script, device, neighbours) for a task.
        """
        # Block Logic: Iterates through the work list, assigning each task to a free thread.
        for (location, script, device, neighbours) in work_list:
            # Functional Utility: Acquires a semaphore, decrementing the count of available worker slots.
            # This blocks if no threads are available, ensuring the pool's capacity is not exceeded.
            self.sem.acquire()
            # Functional Utility: Retrieves a free worker thread from the pool.
            current_thread = self.free_threads.pop(0)
            # Functional Utility: Assigns the specific work (script, data, etc.) to the worker thread.
            current_thread.set_work(location, script, device, neighbours)
            # Functional Utility: Clears the `free` event for the worker, marking it as busy.
            current_thread.free.clear()
            # Functional Utility: Sets the `work` event, signaling the worker thread to start processing its task.
            current_thread.work.set()

        # Block Logic: Waits for all worker threads that were assigned tasks in this batch
        # to complete their current work.
        for current_thread in self.all_threads: # Should iterate through the threads that actually received work, not all_threads
            current_thread.free.wait()

    def close(self):
        """
        Shuts down all worker threads in the pool.

        This method sets a termination flag for each thread, signals them
        to process this flag, and then waits for each thread to complete
        its execution and join.
        """
        # Block Logic: Signals all worker threads to terminate by setting `done` flag and `work` event.
        for current_thread in self.all_threads:
            current_thread.done = 1
            current_thread.work.set()  # Wake up the thread to check the `done` flag.

        # Block Logic: Waits for all worker threads to complete their execution and terminate.
        for current_thread in self.all_threads:
            current_thread.join()


