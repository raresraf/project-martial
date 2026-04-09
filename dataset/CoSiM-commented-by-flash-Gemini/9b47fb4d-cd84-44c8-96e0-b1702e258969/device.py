"""
@file device.py
@brief Implements a simulated distributed device system for processing sensor data and managing
       inter-device communication using a master-worker pattern within a global synchronization framework.
       This module defines Device nodes, a custom ReusableBarrier for global synchronization,
       and worker threads (WorkerThread) for concurrent script execution.

Architectural Intent:
- Simulate a network of interconnected devices (e.g., sensor nodes).
- Employ a master-worker model for concurrent script execution within each device, with workers sharing a task queue.
- Coordinate execution across all devices (`Device` instances) using a reusable barrier synchronization mechanism.
- Manage local sensor data and facilitate data exchange with neighboring devices, ensuring thread safety
  through explicit locking on shared data locations.

Domain: Distributed Systems, Concurrency, Multi-threading, Simulation, Sensor Networks.
"""

from threading import Event, Thread, Condition, Lock
from Queue import Queue, Empty

class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier synchronization mechanism.
           Threads wait at the barrier until all `num_threads` have arrived, allowing subsequent reuse.
           This implementation uses a `Condition` object for managing waiting and notification.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier.
        @param num_threads The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        # Functional Utility: Counter for threads currently waiting at the barrier.
        self.count_threads = self.num_threads
        # Functional Utility: Condition variable for managing thread waiting and notification.
        self.cond = Condition()

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have called `wait()`.
               Once all threads arrive, they are all released simultaneously.
               The barrier then resets for future use.
        """
        # Block Logic: Ensures exclusive access to the counter and condition variable.
        self.cond.acquire()
        self.count_threads -= 1
        # Block Logic: Checks if this is the last thread to arrive at the barrier.
        # Invariant: `self.count_threads` is zero, meaning all `num_threads` have arrived.
        if self.count_threads == 0:
            self.cond.notify_all() # Functional Utility: Notifies all waiting threads to proceed.
            self.count_threads = self.num_threads # Functional Utility: Resets the counter for barrier reuse.
        else:
            self.cond.wait() # Functional Utility: Releases the lock and waits until notified by the last thread.
        self.cond.release() # Functional Utility: Releases the lock after proceeding.

class Device(object):
    """
    @brief Represents a single device (node) in the simulated distributed system.
           Each device manages its own sensor data, communicates with a supervisor,
           and orchestrates concurrent script execution across its worker threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id Unique identifier for this device.
        @param sensor_data Dictionary containing local sensor readings keyed by location.
        @param supervisor Reference to the supervisor object for global coordination and neighbor information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Functional Utility: Stores a reference to the current list of neighboring devices received from the supervisor.
        self.neighbours = None
        # Functional Utility: Defines the number of worker threads that this device will utilize.
        self.num_threads = 24
        # Functional Utility: Event signaling that new scripts have been assigned for the current timepoint.
        self.script_received = Event()
        # Functional Utility: List to store all scripts assigned to this device over its lifetime.
        self.scripts = []
        # Functional Utility: List to hold references to the `WorkerThread` instances managed by the first device.
        self.workers = []
        # Functional Utility: Reference to the shared task `Queue` for worker threads.
        self.queue = None
        # Functional Utility: Event signaling that all work for a timepoint (or simulation end) is done.
        self.work_done = None
        # Functional Utility: Reference to the shared `ReusableBarrier` for global device synchronization.
        self.barrier = None
        # Functional Utility: Reference to the shared global `Lock` for protecting global setup actions.
        self.lock = None
        # Functional Utility: List of `Lock` objects, where `location_lock[idx]` protects data at a specific location index.
        self.location_lock = []
        # Functional Utility: The main thread for this device, handling orchestration.
        self.thread = DeviceThread(self)

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the device with references to globally shared resources (queue, barrier, locks).
               The device with the smallest `device_id` (likely 0) performs the initial setup
               and creation of these shared objects and worker threads.
        @param devices A list of all Device instances in the system.
        """
        # Block Logic: The first device (based on device_id) initializes global shared resources.
        # Invariant: `self.queue`, `self.work_done`, `self.barrier`, `self.lock`, and `self.location_lock`
        #            are initialized once and shared across all devices. `WorkerThread`s are also started here.
        if self.device_id == devices[0].device_id:
            self.queue = Queue() # Functional Utility: Initializes the shared task queue for WorkerThreads.
            self.work_done = Event() # Functional Utility: Initializes the event to signal completion of work.
            self.barrier = ReusableBarrier(len(devices)) # Functional Utility: Initializes the global barrier.
            self.lock = Lock() # Functional Utility: Initializes a global lock for setup.

            # Block Logic: Initializes a list of locks for up to 100 data locations, ensuring fine-grained access control.
            for loc in range(100):
                self.location_lock.append(Lock())

            # Block Logic: Creates and starts `num_threads` worker threads.
            # Invariant: `self.workers` contains running `WorkerThread` instances.
            for thread_id in range(self.num_threads):
                self.workers.append(WorkerThread(self.queue, self.work_done))
                self.workers[thread_id].start()
        else:
            # Block Logic: Other devices simply reference the shared resources initialized by the first device.
            self.queue = devices[0].queue
            self.work_done = devices[0].work_done
            self.barrier = devices[0].barrier
            self.lock = devices[0].lock
            self.location_lock = devices[0].location_lock

        # Functional Utility: Starts the main `DeviceThread` for this device.
        self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by this device's worker threads at a specific data location.
               If `script` is None, it signals that all scripts for the current timepoint have been assigned.
        @param script The script object to be executed. If None, it signals completion of script assignment.
        @param location The data location (e.g., sensor ID) the script will operate on.
        """
        # Block Logic: If a valid script is provided, it's added to the device's list of scripts.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Functional Utility: Signals that all scripts for the current timepoint have been assigned.
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location from this device's local sensor data.
               Note: This method does not acquire/release locks; callers are responsible for managing access.
        @param location The location (key) for which to retrieve data.
        @return The sensor data value if found, otherwise None.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location on this device.
               Note: This method does not acquire/release locks; callers are responsible for managing access.
        @param location The location (key) for which to set data.
        @param data The data to be stored.
        """
        # Block Logic: Updates `sensor_data` only if the location key already exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its main DeviceThread.
               If this device manages worker threads, it also joins them.
        """
        self.thread.join()
        # Block Logic: If this device (the first one) manages worker threads, it joins them to ensure proper shutdown.
        if self.workers != []:
            # Functional Utility: Puts sentinel values into the queue to signal worker threads to terminate.
            for thread_id in range(self.num_threads):
                self.queue.put((None, None, None))
            # Block Logic: Waits for all worker threads to complete their execution and terminate.
            for worker in self.workers:
                worker.join()


class DeviceThread(Thread):
    """
    @brief The main orchestration thread for a Device. Responsible for interacting with the supervisor,
           fetching neighbor information, distributing assigned scripts to the shared task queue,
           and coordinating global synchronization.
    """
    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
               Handles fetching neighbors, queuing scripts for workers, and participating
               in global device synchronization at each timepoint.
        """
        # Block Logic: Main loop for processing timepoints and supervisor interactions.
        while True:
            # Functional Utility: Acquires the global lock to safely access supervisor and neighbor information.
            self.device.lock.acquire()
            # Functional Utility: Fetches information about neighboring devices from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.lock.release() # Functional Utility: Releases the global lock.

            # Block Logic: Checks if the supervisor has signaled the end of the simulation.
            # Invariant: `self.device.neighbours` is None when the simulation should terminate.
            if self.device.neighbours is None:
                self.device.work_done.set() # Functional Utility: Signals that all work is done, allowing worker threads to terminate.
                break # Break from the main `while True` loop, ending the `DeviceThread`.

            # Functional Utility: Waits for the `assign_script` method to signal that all scripts for the current timepoint have been assigned.
            self.device.script_received.wait()
            self.device.script_received.clear() # Functional Utility: Clears the event for the next timepoint.

            # Block Logic: Puts all assigned scripts for the current timepoint into the shared task queue for worker threads.
            # Invariant: `self.device.scripts_queue` contains all (script, location) pairs for the current timepoint.
            for (script, location) in self.device.scripts:
                self.device.queue.put((self.device, script, location))

            # Functional Utility: Blocks until all tasks (scripts) currently in the queue have been processed by worker threads.
            self.device.queue.join()
            
            # Functional Utility: Global synchronization point: waits for all DeviceThreads to complete their timepoint processing.
            self.device.barrier.wait()

class WorkerThread(Thread):
    """
    @brief A worker thread that processes tasks (scripts) from a shared queue.
           It gathers data from its parent device and neighbors, executes the script,
           and updates data, ensuring thread safety with explicit location-based locks.
    """
    def __init__(self, queue, job_done):
        """
        @brief Initializes a WorkerThread instance.
        @param queue The shared `Queue` from which to retrieve tasks.
        @param job_done An `Event` to signal when the overall simulation or current timepoint is complete.
        """
        Thread.__init__(self, name="Worker Thread")
        self.tasks = queue
        self.job_done = job_done
        self.daemon = True # Functional Utility: Sets the thread as a daemon, allowing the program to exit if main threads finish.

    def run(self):
        """
        @brief The main execution loop for the WorkerThread.
               Continuously retrieves tasks, processes them, and updates data,
               terminating when the simulation ends.
        """
        # Block Logic: Continuously processes tasks from the queue until a termination signal is received.
        while True:
            try:
                # Functional Utility: Attempts to retrieve a task from the queue without blocking indefinitely.
                (device, script, location) = self.tasks.get(False)
            except Empty:
                # Block Logic: If the queue is empty and the `job_done` event is set, it's a signal to terminate.
                if self.job_done.is_set():
                    break # Break from the `while True` loop, terminating the `WorkerThread`.
                else:
                    continue # Functional Utility: Continues looping if no task is available yet.

            # Functional Utility: List to aggregate data relevant to the current script's execution.
            script_data = []
            
            # Functional Utility: Acquires the specific lock for the assigned data location, ensuring exclusive access
            # during data gathering and updating for this location across all devices.
            device.location_lock[location].acquire()

            # Block Logic: Gathers data for the specified location from all neighboring devices.
            # Invariant: `script_data` contains available data from neighbors for the given `location`.
            for neighbour in device.neighbours:
                data = neighbour.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Functional Utility: Gathers data for the specified location from the local device.
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script only if relevant data was collected.
            if script_data != []:
                # Functional Utility: Executes the assigned script with the collected data.
                result = script.run(script_data)

                # Functional Utility: Updates the data at 'location' on the local device with the script's result.
                device.set_data(location, result)
                # Block Logic: Updates the data at 'location' on all neighboring devices with the script's result.
                # Invariant: Neighboring devices' data is updated with consistency.
                for neighbour in device.neighbours:
                    neighbour.set_data(location, result)

            # Functional Utility: Releases the specific lock for the assigned data location.
            device.location_lock[location].release()
            
            # Functional Utility: Marks the current task as done in the queue, allowing `queue.join()` to proceed.
            self.tasks.task_done()
