"""
@file device.py
@brief Implements components for a distributed system, likely a simulation or sensor network,
focusing on concurrent data processing, synchronization, and task management.
This module defines Device objects that manage sensor data and execute scripts
using multiple worker threads, employing various synchronization primitives
like events, locks, and a reusable barrier.
"""

from threading import Event, Thread, Lock, Condition # Condition is used by ReusableBarrier.

class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for thread synchronization using a threading.Condition object.
    This barrier allows a fixed number of threads to wait for each other before
    proceeding, and can be reused across multiple synchronization points.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.

        @param num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Counter for threads yet to reach the barrier.
        self.cond = Condition() # Condition variable for signaling and waiting.

    def wait(self):
        """
        @brief Blocks the calling thread until all 'num_threads' have reached the barrier.
        When the last thread arrives, all waiting threads are notified and the barrier resets.
        """
        self.cond.acquire() # Acquires the lock associated with the condition variable.
        self.count_threads -= 1 # Decrements the count of threads yet to reach.
        # Conditional Logic: If this is the last thread to reach the barrier.
        if self.count_threads == 0:
            self.cond.notify_all() # Notifies all waiting threads to proceed.
            self.count_threads = self.num_threads # Resets the counter for future use.
        else:
            self.cond.wait() # Waits (releases lock and blocks) until notified.
        self.cond.release() # Releases the lock.


class Device(object):
    """
    @brief Represents a single device in the distributed system.
    Each device has a unique ID, manages its sensor data, and interacts with a supervisor.
    It processes assigned scripts using a dedicated thread and a pool of worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id (int): A unique identifier for the device.
        @param sensor_data (dict): A dictionary holding sensor readings for different locations.
        @param supervisor (Supervisor): A reference to the central supervisor managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when scripts have been assigned.
        self.scripts = [] # List to hold (script, location) tuples assigned to this device.
        self.timepoint_done = Event() # Event to signal that script assignment for a timepoint is complete.

        self.locks_location = [] # List of Locks, one for each data location, managed centrally.
        self.barrier_timepoint = None # Reference to the global ReusableBarrier.

        self.thread = DeviceThread(self) # The main thread responsible for this device's lifecycle.
        self.thread.start() # Starts the main DeviceThread.

    def __str__(self):
        """
        @brief Provides a string representation of the Device.

        @return str: A formatted string indicating the device ID.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs initial setup for the device, including initializing global locks and barriers.
        Device 0 coordinates this setup and propagates instances to other devices.

        @param devices (list): A list of all Device instances in the system.
        """
        # Conditional Logic: Only Device 0 performs the global setup.
        if self.device_id == 0:
            self.barrier_timepoint = ReusableBarrier(len(devices)) # Initializes the global barrier.
            
            # Block Logic: Initializes 100 Locks for `locks_location` list.
            # This suggests a fixed maximum of 100 data locations.
            iteration = 0
            while iteration < 100:
                iteration += 1
                lock = Lock()
                self.locks_location.append(lock)

            # Block Logic: Propagates the initialized locks and barrier to all other devices.
            for device in devices:
                device.locks_location = self.locks_location # Assigns the globally shared list of locks.
                device.barrier_timepoint = self.barrier_timepoint # Assigns the globally shared barrier.


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location or signals timepoint completion.

        @param script (callable): The script (function or object with a run method) to execute.
                                  If None, it signals that script assignment for the timepoint is done.
        @param location (int): The identifier of the data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location)) # Appends the script and its location.
            self.script_received.set() # Signals that scripts have been received.
        else:
            self.timepoint_done.set() # Signals that script assignment for the timepoint is complete.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location (int): The identifier of the data location.
        @return any: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location (int): The identifier of the data location to update.
        @param data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's main operational thread.
        """
        self.thread.join()


class Worker(Thread):
    """
    @brief A worker thread responsible for executing a portion of scripts for a Device.
    It collects data from its device and its neighbors, runs the assigned script,
    and then propagates the updated data back.
    """
    
    def __init__(self, thread_id, device, neighbors, nr_scripts, scripts):
        """
        @brief Initializes a new Worker instance.

        @param thread_id (int): A unique identifier for this worker thread within the DeviceThread.
        @param device (Device): The Device object this worker thread belongs to.
        @param neighbors (list): A list of neighboring Device objects.
        @param nr_scripts (int): The total number of scripts assigned to the Device for the current timepoint.
        @param scripts (list): The list of (script, location) tuples assigned to the Device.
        """
        Thread.__init__(self)
        self.neighbors = neighbors
        self.device = device
        self.scripts = scripts
        self.nr_scripts = nr_scripts
        self.thread_id = thread_id

    def run(self):
        """
        @brief The main execution logic for the Worker.
        It processes a subset of scripts assigned to its Device, collects data,
        runs the script, and updates data on devices, ensuring proper synchronization.
        """
        # Block Logic: Iterates through scripts, processing every 8th script starting from its thread_id.
        # This is a common pattern for distributing tasks among a fixed number of workers (here 8).
        for index in range(self.thread_id, self.nr_scripts, 8):

            (script, location) = self.scripts[index] # Retrieves the script and its location.

            with self.device.locks_location[location]: # Acquires the location-specific lock.
                script_data = [] # List to collect data for the script.
                
                # Block Logic: Collects data from neighboring devices.
                for device in self.neighbors:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Collects data from its own device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Conditional Logic: If any data was collected, executes the script and propagates results.
                if script_data != []:
                    result = script.run(script_data) # Executes the script.

                    # Block Logic: Propagates the new data to all neighboring devices.
                    for device in self.neighbors:
                        device.set_data(location, result)

                    self.device.set_data(location, result) # Updates data on its own device.


class DeviceThread(Thread):
    """
    @brief The main thread of execution for a Device.
    It is responsible for fetching neighbor information, spawning `Worker` threads
    for script processing, and managing synchronization across devices.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device (Device): The Device object this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.list_workers = [] # List to hold WorkerThread instances.

    def run(self):
        """
        @brief The main execution loop of the DeviceThread.
        It continuously fetches neighbor information, waits for scripts to be assigned,
        spawns worker threads to execute them, waits for worker completion,
        clears events, and then synchronizes with other devices via a global barrier.
        """
        while True:
            # Block Logic: Fetches neighbor devices from the supervisor.
            neighbors = self.device.supervisor.get_neighbours()
            # Conditional Logic: If no neighbors are returned (supervisor signals shutdown), terminates.
            if neighbors is None:
                break

            self.device.timepoint_done.wait() # Waits until scripts for the timepoint are assigned.

            self.device.neighbours = neighbors # Updates the device's neighbor list.

            # Block Logic: Spawns 8 Worker threads to process scripts concurrently.
            # Each worker is assigned a portion of the scripts to execute.
            nr_scripts = len(self.device.scripts)
            for thread_id in range(0, 8):
                worker = Worker(thread_id, self.device, neighbors, nr_scripts, self.device.scripts)
                self.list_workers.append(worker)
                worker.start()

            # Block Logic: Joins all worker threads, waiting for their completion.
            for worker in self.list_workers:
                worker.join()

            self.device.timepoint_done.clear() # Clears the event for the next timepoint.
            # Synchronizes all devices at the global barrier, ensuring all devices complete current timepoint before next.
            self.device.barrier_timepoint.wait()

