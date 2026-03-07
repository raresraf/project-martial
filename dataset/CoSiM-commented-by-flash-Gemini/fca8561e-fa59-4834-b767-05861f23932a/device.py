"""
@file device.py
@brief Implements components for a distributed system, likely a simulation or sensor network,
focusing on concurrent data processing, synchronization, and task management.
This module defines Device objects that manage sensor data and execute scripts
using multiple worker threads and a centralized locking mechanism for data locations.
"""

from threading import Event, Thread, Lock
from Queue import Queue # In Python 3, this is `from queue import Queue`.
from barrier import ReusableBarrier # Assumed to be an external module defining ReusableBarrier.

class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for thread synchronization.
    This barrier ensures that a specified number of threads all reach a certain point
    before any of them are allowed to proceed. It uses semaphores and a lock to
    manage the waiting and releasing of threads across two phases for reusability.
    """

    def __init__(self, numOfTh):
        """
        @brief Initializes the reusable barrier.

        @param numOfTh (int): The total number of threads that must reach the barrier.
        """
        self.numOfTh = numOfTh
        # Stores counts and semaphores for two phases, allowing the barrier to be reused.
        self.threads = [{}, {}]
        self.threads[0]['count'] = numOfTh  # Counter for threads in phase 0.
        self.threads[1]['count'] = numOfTh  # Counter for threads in phase 1.
        self.threads[0]['sem'] = Semaphore(0)  # Semaphore for threads waiting in phase 0.
        self.threads[1]['sem'] = Semaphore(0)  # Semaphore for threads waiting in phase 1.
        self.lock = Lock()  # Lock to protect access to the counters.

    def wait(self):
        """
        @brief Blocks the calling thread until all 'numOfTh' threads have reached the barrier
        and then allows them to proceed. This method executes both phases of the barrier.
        """
        # Block Logic: Iterates through the two phases of the barrier.
        for i in range(0, 2):
            with self.lock:  # Ensures exclusive access to the counter.
                self.threads[i]['count'] -= 1  # Decrements the thread count for the current phase.
                # Conditional Logic: If this is the last thread to reach the barrier in this phase.
                if self.threads[i]['count'] == 0:
                    # Releases all waiting threads from the semaphore.
                    for _ in range(self.numOfTh):
                        self.threads[i]['sem'].release()
                    # Resets the counter for the next use of this phase.
                    self.threads[i]['count'] = self.numOfTh
            self.threads[i]['sem'].acquire()  # Threads wait here until released by the last thread.


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
        self.thread_number = 8  # Fixed number of worker threads per device.
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []  # List to hold (script, location) tuples assigned to this device.

        self.timepoint_done = Event()  # Event to signal that all scripts for a timepoint have been assigned.
        self.location_locks = {}  # Dictionary to store locks for specific data locations.
                                 # Potentially shared across devices (managed by Device 0).
        self.devices_barrier = None  # Global barrier for device synchronization.
        self.setup_devices_done = Event() # Signals that the device's setup has been completed.
        self.neighbours = []  # List of neighboring devices.
        
        self.thread = DeviceThread(self)  # The main thread responsible for this device's lifecycle.
        self.thread.start()  # Starts the main DeviceThread.

    def __str__(self):
        """
        @brief Provides a string representation of the Device.

        @return str: A formatted string indicating the device ID.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs initial setup for the device, including initializing global locks and barriers.
        Device 0 coordinates this setup.

        @param devices (list): A list of all Device instances in the system.
        """
        # Conditional Logic: Only Device 0 performs the global setup.
        if self.device_id is 0:
            self.devices_barrier = ReusableBarrier(len(devices))  # Initializes the global barrier.
            # Creates a "master_lock" within the location_locks dictionary.
            # This master_lock is likely used to protect the `location_locks` dictionary itself.
            self.location_locks["master_lock"] = Lock()
            # Block Logic: Propagates the initialized barrier and location_locks to all other devices.
            for dev in devices:
                if dev.device_id != self.device_id: # Avoids assigning to itself if it's device 0.
                    dev.devices_barrier = self.devices_barrier
                    dev.location_locks = self.location_locks
                dev.setup_devices_done.set() # Signals that setup for this device is done.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location or signals timepoint completion.

        @param script (callable): The script (function or object with a run method) to execute.
                                  If None, it signals that script assignment for the timepoint is done.
        @param location (int): The identifier of the data location the script operates on.
        """
        # Conditional Logic: If a script is provided, it's added to the scripts list.
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set() # Signals that script assignment for the timepoint is complete.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location (int): The identifier of the data location.
        @return any: The sensor data at the specified location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

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


class DeviceThread(Thread):
    """
    @brief The main thread of execution for a Device.
    It is responsible for fetching neighbor information, coordinating the processing
    of scripts using a queue-based worker pool, and managing synchronization across devices.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device (Device): The Device object this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = [] # List of neighboring devices (fetched in run method).
        self.script_queue = Queue() # Queue for distributing scripts to worker threads.
        self.worker_pool = [] # List to hold WorkerThread instances.
        
        # Block Logic: Creates and initializes `thread_number` (8) WorkerThread instances.
        for _ in range(self.device.thread_number):
            self.worker_pool.append(WorkerThread(self))

    def run(self):
        """
        @brief The main execution loop of the DeviceThread.
        It continuously fetches neighbor information, waits for scripts to be assigned,
        submits them to its worker pool, waits for worker pool completion,
        and then synchronizes with other devices via a global barrier.
        """
        # Block Logic: Starts all worker threads in the pool.
        for worker in self.worker_pool:
            worker.start()

        while True:
            # Block Logic: Fetches neighbor devices from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()
            # Conditional Logic: If no neighbors are returned (supervisor signals shutdown), terminates.
            if self.neighbours is None:
                break # Terminates the thread.

            self.device.setup_devices_done.wait() # Waits until global setup is complete.
            self.device.timepoint_done.wait() # Waits until scripts for the timepoint are assigned.

            # Block Logic: Puts all assigned scripts into the script queue for worker threads to pick up.
            for (script, location) in self.device.scripts:
                self.script_queue.put((script, location))

            self.script_queue.join() # Blocks until all items in the queue have been processed.
            self.device.devices_barrier.wait() # Synchronizes all devices at the global barrier.
            self.device.timepoint_done.clear() # Clears the event for the next timepoint.

        # Block Logic: Signals all worker threads to terminate and waits for their completion.
        for _ in range(len(self.worker_pool)):
            self.script_queue.put(None) # Puts None as a termination signal for each worker.

        for worker in self.worker_pool:
            worker.join() # Waits for all worker threads to terminate.

class WorkerThread(Thread):
    """
    @brief An auxiliary worker thread responsible for executing scripts from the `script_queue`.
    It collects data from its device and neighbors, runs the assigned script,
    and then propagates the updated data back.
    """

    def __init__(self, device_thread):
        """
        @brief Initializes a new WorkerThread instance.

        @param device_thread (DeviceThread): The DeviceThread managing this worker.
        """
        Thread.__init__(self)
        self.device_thread = device_thread

    def run(self):
        """
        @brief The main execution loop for a WorkerThread.
        It continuously retrieves tasks (scripts) from the queue, processes them,
        and updates data on devices, ensuring proper synchronization using location-specific locks.
        """
        while True:
            script_pair = self.device_thread.script_queue.get() # Retrieves a script task from the queue.

            # Conditional Logic: If a None signal is received, it indicates termination.
            if script_pair is None:
                break # Terminates the worker thread.

            script, location = script_pair # Unpacks the script and its location.

            # Synchronization: Acquires the "master_lock" to safely access or create location-specific locks.
            with self.device_thread.device.location_locks["master_lock"]:
                # Conditional Logic: Creates a new lock for the location if one doesn't already exist.
                if location not in self.device_thread.device.location_locks:
                    self.device_thread.device.location_locks[location] = Lock()

            # Synchronization: Acquires the location-specific lock for this data location.
            self.device_thread.device.location_locks[location].acquire()

            script_data = [] # List to collect data for the script.
            
            # Block Logic: Collects data from neighboring devices.
            for device in self.device_thread.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Collects data from its own device.
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)
            
            # Conditional Logic: If any data was collected, executes the script and propagates results.
            if script_data != []:
                result = script.run(script_data) # Executes the script.

                # Block Logic: Propagates the new data to all neighboring devices.
                for device in self.device_thread.neighbours:
                    device.set_data(location, result)
                
                self.device_thread.device.set_data(location, result) # Updates data on its own device.

            self.device_thread.script_queue.task_done() # Signals that the current task is complete.
            self.device_thread.device.location_locks[location].release() # Releases the location-specific lock.
