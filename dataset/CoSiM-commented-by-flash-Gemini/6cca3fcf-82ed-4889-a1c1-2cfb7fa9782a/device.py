
"""
@file device.py
@brief This module defines the core components for a distributed device simulation framework.

The framework simulates a network of devices that can execute scripts and exchange sensor data.
It employs a multi-threaded architecture to handle concurrent script execution and data management.

Classes:
- Device: Represents a single simulated device, managing its state, sensor data, and script execution.
- DeviceThread: Manages the lifecycle and continuous operation of a Device, including coordinating with a supervisor and other devices.
- Scripter: Orchestrates the execution of scripts on a device by managing a pool of ScriptExecutor threads.
- ScriptExecutor: Executes individual scripts, retrieves sensor data, and updates device data.
- ReusableBarrier: A synchronization primitive used to coordinate multiple threads, allowing them to wait for each other before proceeding.

High-level Architecture:
- A Supervisor (not defined in this file) manages a collection of Device instances.
- Each Device runs its own DeviceThread for continuous operation.
- Devices can receive scripts, which are then executed by a pool of ScriptExecutor threads managed by a Scripter.
- Data exchange between devices is handled through explicit methods, with synchronization mechanisms (locks) to ensure data consistency.
- A ReusableBarrier is used to synchronize devices at specific points in the simulation, such as at the end of a timepoint.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue


class Device(object):
    """
    @brief Represents a single simulated device in the distributed system.

    Manages the device's unique identifier, sensor data, script execution state,
    and interaction with the supervisor and other devices.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings, keyed by location.
        supervisor (object): A reference to the central supervisor managing all devices.
        script_received (Event): An event flag set when a new script is assigned.
        scripts (list): A list to store assigned scripts and their locations.
        barrier (ReusableBarrier): A synchronization barrier for coordinating with other devices.
        script_running (Lock): A lock to manage access to script-related operations.
        timepoint_done (Event): An event flag set when a timepoint's script execution is complete.
        data_locks (dict): A dictionary of locks, one for each sensor data location, to ensure thread-safe data access.
        queue (Queue): A queue to hold scripts to be executed by the ScriptExecutors.
        available_threads (int): The number of script executor threads available for this device.
        can_get_data (Lock): A lock to control access to the device's sensor data.
        master (Device): A reference to the master device in the simulation, typically the first device.
        script_over (bool): Flag indicating if script processing for the current timepoint is finished.
        alive (bool): Flag indicating if the device thread should continue running.
        thread (DeviceThread): The dedicated thread managing this device's operations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id (int): The unique identifier for this device.
        @param sensor_data (dict): Initial sensor data for the device.
        @param supervisor (object): The supervisor object managing this device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.barrier = None
        
        self.script_running = Lock()
        self.timepoint_done = Event()
        
        self.data_locks = dict()
        
        self.queue = Queue()
        
        self.available_threads = 14

        # Initialize a lock for each sensor data location for fine-grained access control.
        for loc in sensor_data:
            self.data_locks.__setitem__(loc, Lock())

        self.can_get_data = Lock()

        self.master = None
        self.script_over = False
        self.alive = True
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return (str): A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device dependencies, specifically the synchronization barrier and master device.
        @param devices (list): A list of all Device instances in the simulation.
        """
        # Initialize a reusable barrier for all devices to synchronize.
        self.barrier = ReusableBarrier(len(devices))
        
        # Designate the first device in the list as the master.
        self.master = devices[0]

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific sensor location.
        If script is None, it signals the end of scripts for the current timepoint.
        @param script (object or None): The script object to execute, or None to signal completion.
        @param location (str): The sensor data location relevant to the script.
        """
        # Acquire lock to prevent race conditions during script assignment.
        if script is not None:
            self.script_running.acquire()
            self.scripts.append((script, location))
            # Add the script to the queue for processing by ScriptExecutors.
            self.queue.put_nowait((script, location))
            # Signal that a new script has been received.
            self.script_received.set()
        else:
            self.script_running.acquire()
            # If script is None, it indicates that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data from a specified location, ensuring thread safety.
        @param location (str): The key corresponding to the desired sensor data.
        @return (any): The sensor data at the given location, or None if the location does not exist.
        """
        # Acquire a global lock to prevent multiple threads from reading/writing data simultaneously.
        self.can_get_data.acquire()
        return_value = self.sensor_data[location] if location in self.sensor_data else None
        self.can_get_data.release()
        return return_value

    def get_device_data(self, location):
        """
        @brief Retrieves sensor data for a specific location, using a per-location lock for thread safety.
        @param location (str): The key corresponding to the desired sensor data.
        @return (any): The sensor data at the given location, or None if the location does not exist.
        """
        # Check if the location exists in sensor_data before attempting to access.
        if location not in self.sensor_data:
            return None

        # Acquire the specific lock for this data location to ensure thread-safe access.
        self.data_locks.get(location).acquire()

        new_data = self.sensor_data[location]

        self.data_locks.get(location).release()

        return new_data

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location, ensuring thread safety.
        @param location (str): The key corresponding to the sensor data to be updated.
        @param data (any): The new data value to set.
        """
        if location in self.sensor_data:
            # Acquire the specific lock for this data location to ensure thread-safe modification.
            self.data_locks.get(location).acquire()
            self.sensor_data[location] = data
            self.data_locks.get(location).release()

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device, joining its dedicated thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief A dedicated thread for managing the continuous operation of a Device.

    This thread handles the main loop of the device, coordinating with the supervisor,
    executing scripts, and synchronizing with other devices.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread.
        @param device (Device): The Device instance that this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        It continuously checks for new scripts, executes them, and synchronizes
        with other devices at predefined timepoints.
        """
        while True:
            # Acquire global data access lock to prevent other devices from accessing data during setup.
            self.device.can_get_data.acquire()
            
            # Get neighbors from the supervisor. This call implicitly synchronizes device states.
            neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Checks if the simulation is shutting down.
            # If no neighbours are returned, it signifies the end of the simulation.
            if neighbours is None:
                # Wait at the master's barrier to ensure all threads finish before exiting.
                self.device.master.barrier.wait()

                self.device.can_get_data.release()
                return

            # Create a Scripter instance to manage script execution for this timepoint.
            script_instance = Scripter(self.device, neighbours)

            script_instance.start()

            # Wait until all scripts for the current timepoint are done processing.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Signal that script processing for the current timepoint is over.
            self.device.script_over = True
            # Signal the Scripter that scripts are over for this timepoint.
            self.device.script_received.set()

            # Wait for the Scripter thread to complete its current tasks.
            script_instance.join()

            # Re-queue any scripts that might have been assigned during script_over state.
            for (script, location) in self.device.scripts:
                self.device.queue.put_nowait((script, location))

            self.device.script_over = False

            # Synchronize with other devices at the barrier before proceeding to the next timepoint.
            self.device.master.barrier.wait()

            self.device.can_get_data.release()
            # Release the script_running lock, allowing new scripts to be assigned.
            self.device.script_running.release()


class Scripter(Thread):
    """
    @brief Manages the execution of scripts on a Device by dispatching them to ScriptExecutor threads.

    It creates a pool of executor threads and feeds them scripts from the device's queue.
    """

    def __init__(self, device, neighbours):
        """
        @brief Initializes a new Scripter instance.
        @param device (Device): The Device instance associated with this Scripter.
        @param neighbours (list): A list of neighboring Device instances for data exchange.
        """
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution loop for the Scripter thread.
        It creates and manages ScriptExecutor threads, assigning scripts to them
        and handling the shutdown of script execution for a timepoint.
        """
        list_executors = []

        # Create a pool of ScriptExecutor threads.
        for iterator in range(1, self.device.available_threads):
            executor = ScriptExecutor(self.device, self.device.queue, self.neighbours, iterator)
            list_executors.append(executor)
            executor.start()

        while True:
            # Wait for a signal that new scripts have been received or the timepoint is over.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Block Logic: Checks if the current timepoint's script execution is finished.
            if self.device.script_over:
                # Signal all ScriptExecutor threads to shut down by putting None into their queue.
                for iterator in range(1, self.device.available_threads):
                    self.device.queue.put((None, None))

                # Wait for all ScriptExecutor threads to finish.
                for executor in list_executors:
                    executor.join()

                # Re-initialize the queue for the next timepoint.
                self.device.queue = Queue(-1)
                return

            self.device.script_running.release()


class ScriptExecutor(Thread):
    """
    @brief Executes individual scripts assigned to a Device.

    Each ScriptExecutor thread retrieves a script from the device's queue,
    gathers necessary data from neighbors and itself, runs the script,
    and updates the data on relevant devices.
    """

    def __init__(self, device, queue, neighbours, identifier):
        """
        @brief Initializes a new ScriptExecutor instance.
        @param device (Device): The Device instance that this executor operates on.
        @param queue (Queue): The queue from which to retrieve scripts.
        @param neighbours (list): A list of neighboring Device instances for data exchange.
        @param identifier (int): A unique identifier for this executor thread.
        """
        Thread.__init__(self, name="Script Executor %d" % identifier)
        self.device = device
        self.queue = queue
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution loop for the ScriptExecutor thread.
        It continuously fetches scripts from the queue, executes them, and updates data.
        """
        while True:
            # Retrieve a script and its associated location from the queue.
            # This call blocks until an item is available.
            (script, location) = self.queue.get()
            # Block Logic: Checks for a shutdown signal.
            # If script is None, it indicates that this executor should terminate.
            if script is None:
                return

            script_data = []
            # Gather data from neighboring devices for the script.
            for device in self.neighbours:
                data = device.get_device_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from the current device itself for the script.
            data = self.device.get_device_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script if relevant data was collected.
            if script_data != []:
                # Execute the script with the collected data.
                result = script.run(script_data)
                
                # Update data on neighboring devices with the script's result.
                for device in self.neighbours:
                    device.set_data(location, result)
                
                # Update data on the current device with the script's result.
                self.device.set_data(location, result)


class ReusableBarrier:
    """
    @brief A reusable barrier synchronization primitive for coordinating multiple threads.

    Allows a fixed number of threads to wait for each other to reach a certain point
    before any of them can proceed. It is designed to be reusable for multiple synchronization points.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrier.
        @param num_threads (int): The number of threads that must reach the barrier before it releases.
        """
        self.num_threads = num_threads
        # Two counters and semaphores are used for the two-phase barrier to ensure reusability.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()          # Protects access to the thread counters.
        self.threads_sem1 = Semaphore(0)  # Semaphore for the first phase of the barrier.
        self.threads_sem2 = Semaphore(0)  # Semaphore for the second phase of the barrier.

    def wait(self):
        """
        @brief Blocks until all threads have reached this barrier.
        Implements a two-phase barrier to allow for reusability.
        """
        # First phase of the barrier.
        self.phase(self.count_threads1, self.threads_sem1)
        # Second phase of the barrier.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements one phase of the barrier synchronization.
        A thread decrements a counter and waits on a semaphore.
        The last thread to reach the barrier releases all waiting threads.
        @param count_threads (list): A list containing a single integer, the thread counter for this phase.
        @param threads_sem (Semaphore): The semaphore associated with this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # Block Logic: Last thread to reach the barrier.
            # If this is the last thread, it releases all waiting threads and resets the counter.
            if count_threads[0] == 0:
                for iterator in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire() # Wait to be released by the last thread.
