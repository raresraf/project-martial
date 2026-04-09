"""
@file device.py
@brief Implements a simulated distributed device system where devices process sensor data
       and communicate with neighbors. This version introduces a factory pattern for worker
       threads to execute scripts concurrently on each device.

Architectural Intent:
- Simulate a network of interconnected devices (e.g., sensor nodes).
- Support concurrent data processing via worker threads on each device.
- Coordinate execution across devices using a global barrier.
- Manage local sensor data and exchange data with neighboring devices, with fine-grained locking.

Domain: Distributed Systems, Concurrency, Factory Pattern, Simulation.
"""

from threading import Event, Thread, Lock
from barrier import Barrier # Assuming barrier.py contains a Barrier class for synchronization.
from workerfactory import WorkerFactory # Assuming workerfactory.py contains a WorkerFactory class.

class Device(object):
    """
    @brief Represents a single device (node) in the simulated distributed system.
           Each device manages its own sensor data, communicates with a supervisor,
           and uses a main DeviceThread along with auxiliary Worker threads for concurrent operations.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id Unique identifier for this device.
        @param sensor_data Dictionary containing local sensor readings.
        @param supervisor Reference to the supervisor object for global coordination.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Functional Utility: Event to signal when new scripts have been assigned.
        self.script_received = Event()
        # Functional Utility: List to hold tuples of (script, location) assigned to this device.
        self.scripts = []
        # Functional Utility: Event to signal completion of processing for a specific timepoint.
        self.timepoint_done = Event()
        # Functional Utility: The main thread for this device, handling orchestration.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Functional Utility: List of (location, Lock) pairs for protecting specific data locations.
        self.locks = []
        # Functional Utility: Reference to the global barrier for synchronizing all Device instances.
        self.barrier = None

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the device with references to all other devices in the system.
               Initializes the global barrier if this is device 0, and sets up per-location locks.
        @param devices A list of all Device instances in the system.
        """
        # Functional Utility: Calculates the total number of devices.
        num_devices = len(devices)
        # Block Logic: Initializes the global barrier instance for all devices. This happens once on device 0.
        # Invariant: The global barrier (`self.barrier`) is initialized and shared across all devices.
        if self.barrier is None and self.device_id == 0:
            self.barrier = Barrier(num_devices)
            # Block Logic: Ensures all devices reference the same barrier instance.
            for dev in devices:
                if dev.barrier is None:
                    dev.barrier = self.barrier
        # Block Logic: Creates a Lock for each data location in the sensor_data, allowing fine-grained concurrency control.
        # Invariant: Each data location has an associated lock to prevent race conditions during read/write.
        for loc in self.sensor_data:
            self.locks.append((loc, Lock()));

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by this device's worker threads at a specific data location.
        @param script The script object to be executed.
        @param location The data location (e.g., sensor ID) the script will operate on.
        """
        # Block Logic: If a script is provided, adds it to the list of scripts to be processed and signals its arrival.
        # Invariant: `self.scripts` contains pending (script, location) tuples.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Functional Utility: Signals that the processing for the current timepoint is done for this device.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location, acquiring a lock for thread safety.
        @param location The location (key) for which to retrieve data.
        @return The sensor data or None if the location is not found.
        """
        # Block Logic: Checks if the location exists in the sensor data.
        if location in self.sensor_data:
            # Block Logic: Acquires the specific lock associated with this data location.
            # Pre-condition: `self.locks` contains a lock for `location`.
            for iter in self.locks:
                if iter[0] == location:
                    iter[1].acquire();
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location and releases its corresponding lock.
        @param location The location (key) for which to set data.
        @param data The data to be stored.
        """
        # Block Logic: Checks if the location exists in the sensor data before updating.
        if location in self.sensor_data:
            self.sensor_data[location] = data
            # Block Logic: Releases the specific lock associated with this data location.
            # Pre-condition: `self.locks` contains a lock for `location` and it was previously acquired.
            for iter in self.locks:
                if iter[0] == location:
                    iter[1].release();

    def shutdown(self):
        """
        @brief Shuts down the device by joining its main DeviceThread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main orchestration thread for a Device. Responsible for interacting with the supervisor,
           managing timepoints, and delegating script execution to worker threads via WorkerFactory.
    """
    # Functional Utility: Defines the number of worker cores (threads) available per device.
    num_cores = 8
    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Functional Utility: WorkerFactory instance to manage and utilize a pool of worker threads.
        self.worker_factory = WorkerFactory(DeviceThread.num_cores, device)

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
               Handles fetching neighbors, processing assigned scripts, managing timepoints,
               and synchronizing globally with other devices.
        """
        # Block Logic: Main loop for processing timepoints and supervisor interactions.
        while True:
            # Functional Utility: Fetches information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: If no neighbors are returned (signaling simulation end), break the loop.
            # Invariant: `neighbours` is None when the simulation is to terminate.
            if neighbours is None:
                break
            
            # Block Logic: Inner loop to manage script assignments and timepoint completion.
            while True:
                # Functional Utility: Waits for the supervisor to signal timepoint completion for this device.
                if self.device.timepoint_done.wait():
                    # Block Logic: Checks if new scripts have been received for processing.
                    if self.device.script_received.isSet():
                        self.device.script_received.clear() # Functional Utility: Clears the event after processing.
                        # Block Logic: Adds all received scripts as tasks to the WorkerFactory.
                        for (script, location) in self.device.scripts:
                            self.worker_factory.add_tasks((neighbours, script, location))
                    else:
                        # Functional Utility: Clears the timepoint_done event for the next cycle.
                        self.device.timepoint_done.clear()
                        # Functional Utility: Signals that scripts are now ready to be processed.
                        self.device.script_received.set()
                        break # Break from inner while loop
            
            # Functional Utility: Waits for all worker threads to finish processing their current tasks.
            self.worker_factory.wait_for_finish()

            # Functional Utility: Global synchronization point: waits for all devices to complete their timepoint processing.
            self.device.barrier.wait()

        # Functional Utility: Shuts down the WorkerFactory and its associated worker threads.
        self.worker_factory.shutdown()


from Queue import Queue
from threading import Thread

class WorkerFactory(object):
    """
    @brief Manages a pool of worker threads for a specific Device.
           Responsible for creating, distributing tasks to, and coordinating
           the lifecycle of these worker threads.
    """
    def __init__(self, num_workers, parent_device):
        """
        @brief Initializes the WorkerFactory.
        @param num_workers The number of worker threads to manage.
        @param parent_device The Device instance to which these workers belong.
        """
        self.num_workers = num_workers
        # Functional Utility: A thread-safe queue to hold tasks for worker threads.
        self.task_queue = Queue(num_workers)
        # Functional Utility: List to store references to the worker threads.
        self.worker_threads = []
        self.current_device = parent_device
        self.start_workers()

    def start_workers(self):
        """
        @brief Creates and starts the specified number of Worker threads.
        """
        # Block Logic: Populates the `worker_threads` list with new Worker instances and starts them.
        for _ in range(0, self.num_workers):
            worker_thread = Worker(self.task_queue, self.current_device)
            self.worker_threads.append(worker_thread)
        for worker in self.worker_threads:
            worker.start()

    def add_tasks(self, necessary_data):
        """
        @brief Adds a new task to the task queue for processing by a worker thread.
        @param necessary_data A tuple containing (neighbors, script, location) required for task execution.
        """
        self.task_queue.put(necessary_data)

    def wait_for_finish(self):
        """
        @brief Blocks until all tasks in the queue have been processed by the worker threads.
        """
        self.task_queue.join()

    def shutdown(self):
        """
        @brief Shuts down the WorkerFactory and all its managed worker threads.
        """
        self.task_queue.join()
        # Block Logic: Adds sentinel tasks to the queue to signal worker threads to terminate.
        for _ in xrange(self.num_workers):
            self.add_tasks((None, None, None)) # Functional Utility: Sentinel value to terminate workers.

        # Block Logic: Waits for all worker threads to complete their execution and terminate.
        for worker in self.worker_threads:
            worker.join()

class Worker(Thread):
    """
    @brief A worker thread that processes tasks (scripts) from a shared queue,
           gathers data from its parent device and neighbors, executes the script,
           and updates data.
    """
    def __init__(self, task_queue, parent_device):
        """
        @brief Initializes a Worker thread.
        @param task_queue The shared Queue from which to retrieve tasks.
        @param parent_device The Device instance to which this worker belongs.
        """
        Thread.__init__(self)
        self.my_queue = task_queue
        self.current_device = parent_device

    def run(self):
        """
        @brief The main execution loop for the Worker thread.
               Continuously retrieves tasks, processes them, and updates data.
        """
        # Block Logic: Continuously processes tasks from the queue until a termination signal is received.
        while True:
            # Functional Utility: Retrieves a task from the queue. Blocks if the queue is empty.
            neigh, script, location = self.my_queue.get()
            # Block Logic: Checks for a sentinel value to signal the worker thread to terminate.
            # Invariant: (None, None, None) tuple indicates shutdown.
            if neigh is None or script is None or location is None:
                self.my_queue.task_done() # Functional Utility: Marks the sentinel task as done.
                break

            # Functional Utility: List to aggregate data for the current script's execution.
            script_data = []
            
            # Block Logic: Gathers data for the specified location from all neighboring devices.
            # Invariant: `script_data` contains available data from neighbors (excluding self) for the given `location`.
            for device in neigh:
                if self.current_device.device_id != device.device_id: # Block Logic: Excludes data from itself if present in neighbors.
                    data = device.get_data(location) # Functional Utility: Retrieves data with locking.
                    if data is not None:
                        script_data.append(data)

            # Functional Utility: Gathers data for the specified location from the local device.
            data = self.current_device.get_data(location) # Functional Utility: Retrieves data with locking.
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script only if relevant data was collected.
            # Pre-condition: `script_data` is not empty.
            if script_data != []:
                # Functional Utility: Executes the assigned script with the collected data.
                result = script.run(script_data)

                # Block Logic: Updates the data at 'location' on all neighboring devices with the script's result.
                # Invariant: Neighboring devices' data is updated with consistency.
                for device in neigh:
                    if self.current_device.device_id != device.device_id: # Block Logic: Excludes updating itself if present in neighbors.
                        device.set_data(location, result) # Functional Utility: Sets data with locking.
                # Functional Utility: Updates the data at 'location' on the local device with the script's result.
                self.current_device.set_data(location, result) # Functional Utility: Sets data with locking.
            self.my_queue.task_done() # Functional Utility: Marks the current task as done in the queue.
