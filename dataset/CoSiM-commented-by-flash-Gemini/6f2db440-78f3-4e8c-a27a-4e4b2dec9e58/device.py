"""
This module implements a multi-threaded device simulation system.

It features devices that interact with a supervisor, exchange data with neighbors,
and process assigned scripts using a worker thread pool. The system employs
various threading primitives like Events, Locks, Conditions, and Semaphores
for fine-grained synchronization and coordination across devices and threads.

The architecture includes:
- `Device`: Represents a single simulated entity with its own data and threads.
- `DeviceThread`: The main control thread for each `Device`, coordinating supervisor
  interactions and timepoint progression.
- `WorkerThreadPool`: Manages a pool of worker threads within each `Device` for
  concurrent script execution.
- `SimpleWorker`: Individual threads from the pool that execute scripts.
"""

from threading import Event, Thread, Lock, Condition, Semaphore


class Device(object):
    """
    Represents a single simulated device in the system.

    Each device manages its own sensor data, interacts with a supervisor,
    and coordinates script execution through its main DeviceThread and
    an internal WorkerThreadPool. Devices synchronize globally via a "lead"
    device (device_id 0) and locally using per-location locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary mapping data locations (int) to their values.
            supervisor (object): A reference to the supervisor object that manages devices.
        """
        self.devices = None          # List of all Device objects in the simulation.
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal when new scripts have been assigned.
        self.scripts = []            # List of (script, location) tuples assigned to this device.
        self.timepoint_done = Event()  # Event to signal that all scripts for a timepoint have been assigned.
        
        # The main thread responsible for this device's control flow.
        self.thread = DeviceThread(self)
        self.thread.start()

        self.lead_device_index = -1  # Index of the lead device (device_id 0) in the `devices` list.

        # This list will be populated with global locks for each data location.
        # It's shared across all devices for concurrent access protection.
        self.location_locks = []

        # Special initialization for the lead device (device_id 0) for global coordination.
        if device_id == 0:
            self.threads_that_finished_no = 0  # Counter for devices that finished current timepoint.
            self.next_time_point_cond = Condition() # Condition for global timepoint synchronization.
            self.can_start = Event() # Event to signal that global setup is complete and workers can start.

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global resources and device interconnections.
        This method is called by the supervisor once all devices are created.
        The lead device (ID 0) initializes global location locks and signals
        other devices to proceed.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.devices = devices

        # Find the index of the lead device (device_id 0) in the global list.
        for i in xrange(len(self.devices)):
            if devices[i].device_id == 0:
                self.lead_device_index = i
                break

        if self.device_id == 0:
            self.can_start.clear() # Clear event until setup is complete.

            # Determine the maximum data location ID to correctly size the location_locks list.
            max_lock = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > max_lock:
                        max_lock = location

            # Initialize a global list of Locks, one for each possible data location.
            for _ in range(0, max_lock + 1):
                self.location_locks.append(Lock())

            # Distribute the globally initialized location locks to all devices.
            for device in devices:
                device.location_locks = self.location_locks

            self.can_start.set() # Signal that global setup is complete.
        else:
            # Non-lead devices wait for the lead device to complete global setup.
            devices[self.lead_device_index].can_start.wait()

    def assign_script(self, script, location):
        """
        Assigns a processing script and its associated data location to this device.
        If `script` is None, it signals that script assignment for the current
        timepoint is complete.

        Args:
            script (object): The script object (must have a `run` method) to be executed,
                             or None to signal timepoint completion.
            location (int): The data location this script pertains to.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a new script has been received.
        else:
            self.timepoint_done.set() # Signal that no more scripts are coming for this timepoint.

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location within this device.

        Args:
            location (int): The location ID for which to retrieve data.

        Returns:
            Any: The data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location within this device.

        Args:
            location (int): The location ID for which to set data.
            data (Any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the graceful shutdown sequence for the device's main thread.
        """
        self.thread.join()

    def notify_finish(self):
        """
        Notifies the lead device that this device has completed its processing
        for the current timepoint. This method is used for global synchronization
        across all devices.
        """
        # Acquire the condition lock of the lead device to safely update the shared counter.
        self.devices[self.lead_device_index].next_time_point_cond.acquire()
        self.devices[self.lead_device_index].threads_that_finished_no += 1

        # If all devices have finished, reset the counter and notify all waiting devices.
        if self.devices[self.lead_device_index].threads_that_finished_no == len(self.devices):
            self.devices[self.lead_device_index].threads_that_finished_no = 0
            self.devices[self.lead_device_index].next_time_point_cond.notifyAll()
        else:
            # If not all devices have finished, this device waits.
            self.devices[self.lead_device_index].next_time_point_cond.wait()

        # Release the condition lock.
        self.devices[self.lead_device_index].next_time_point_cond.release()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    It is responsible for interacting with the supervisor to get neighbor
    information, managing the timepoint progression, and delegating script
    execution to a `WorkerThreadPool`. It also participates in global
    synchronization with other DeviceThreads.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = WorkerThreadPool(device) # Worker pool for executing scripts.

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously processes timepoints: retrieves neighbor information,
        submits scripts to its worker pool, waits for script completion,
        and synchronizes with other devices for the next timepoint.
        """
        while True:
            # Retrieves the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            # If no neighbors are returned (e.g., shutdown signal from supervisor),
            # gracefully shut down the worker pool and exit the loop.
            if neighbours is None:
                self.thread_pool.shutdown()
                break

            # Waits until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Clear the event for the next timepoint.

            # Submits all assigned scripts to the worker thread pool for execution.
            for (script, location) in self.device.scripts:
                self.thread_pool.do_work(script, location, neighbours)

            # Waits until all work submitted to the thread pool for this timepoint is finished.
            self.thread_pool.wait_to_finish_work()

            # Notifies the lead device that this device has finished its timepoint processing.
            # This call blocks until all devices have notified, ensuring global synchronization.
            self.device.notify_finish()


class WorkerThreadPool(object):
    """
    Manages a pool of `SimpleWorker` threads for a `Device`.

    It receives script execution requests, assigns them to available workers,
    and tracks the completion of all submitted work.
    """
    
    def __init__(self, device):
        """
        Initializes the WorkerThreadPool.

        Args:
            device (Device): The Device instance this thread pool belongs to.
        """
        self.device = device
        self.work_finished_event = Event() # Event to signal when all workers are idle and all work is done.
        self.work_finished_event.set()     # Initially set as no work is pending.
        self.worker_pool = []              # List to hold all SimpleWorker threads.

        self.ready_for_work_queue = []     # Queue of SimpleWorker threads ready for new tasks.
        self.read_to_work_thread_sem = Semaphore(8) # Semaphore to limit the number of concurrently active workers.
                                                  # Initialized to the pool size (8).
        self.queue_lock = Lock()           # Lock to protect `ready_for_work_queue`.
        
        # Create and start the SimpleWorker threads.
        for _ in xrange(8):
            thread = SimpleWorker(self, self.device)
            self.worker_pool.append(thread)
            self.ready_for_work_queue.append(thread) # Initially all workers are ready.
            thread.start()

    def do_work(self, script, location, neighbours):
        """
        Submits a script execution task to an available worker in the pool.

        Args:
            script (object): The script object to execute.
            location (int): The data location for the script.
            neighbours (list): List of neighboring devices.
        """
        # If work was finished, clear the event as new work is being submitted.
        if self.work_finished_event.isSet():
            self.work_finished_event.clear()
        
        # Acquire a semaphore to get an available worker (blocks if all workers are busy).
        self.read_to_work_thread_sem.acquire()
        
        self.queue_lock.acquire()
        worker = self.ready_for_work_queue.pop(0) # Get an idle worker from the queue.
        self.queue_lock.release()
        
        # Assign the work to the worker.
        worker.do_work(script, location, neighbours)

    def shutdown(self):
        """
        Initiates the graceful shutdown of all worker threads in the pool.
        """
        for worker in self.worker_pool:
            worker.should_i_stop = True # Signal the worker to stop.
            worker.data_for_work_ready.release() # Release semaphore to wake up waiting worker.
        
        # Join all worker threads to ensure they have finished before exiting.
        for worker in self.worker_pool:
            worker.join()

    def worker_finished(self, worker):
        """
        Callback method invoked by a `SimpleWorker` when it completes a task.
        Makes the worker available for new tasks and potentially signals
        that all work is finished.

        Args:
            worker (SimpleWorker): The worker thread that just finished its task.
        """
        self.queue_lock.acquire()
        self.ready_for_work_queue.append(worker) # Add the worker back to the ready queue.

        # If all workers are back in the queue and the work_finished_event is not set,
        # it means all submitted tasks are now complete.
        if len(self.ready_for_work_queue) == 8 and not self.work_finished_event.isSet():
            self.work_finished_event.set() # Signal that all work is finished.

        self.queue_lock.release()
        self.read_to_work_thread_sem.release() # Release semaphore to indicate a worker is now idle.

    def wait_to_finish_work(self):
        """
        Blocks until all tasks submitted to the thread pool have been completed.
        """
        self.work_finished_event.wait()


class SimpleWorker(Thread):
    """
    A simple worker thread responsible for executing a single assigned script
    task. It runs continuously, waiting for new tasks from the `WorkerThreadPool`.
    """
    
    def __init__(self, worker_pool, device):
        """
        Initializes a SimpleWorker thread.

        Args:
            worker_pool (WorkerThreadPool): The thread pool this worker belongs to.
            device (Device): The Device instance this worker operates for.
        """
        Thread.__init__(self)
        self.worker_pool = worker_pool
        self.should_i_stop = False # Flag to signal the worker to terminate.
        self.data_for_work_ready = Semaphore(0) # Semaphore for worker to wait for new work.
        self.device = device
        self.script = None    # The script to be executed.
        self.location = None  # The data location for the script.
        self.neighbours = None # List of neighboring devices.

    def do_work(self, script, location, neighbours):
        """
        Assigns a new task to this worker.

        Args:
            script (object): The script object to execute.
            location (int): The data location for the script.
            neighbours (list): List of neighboring devices.
        """
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.data_for_work_ready.release() # Release semaphore to wake up the worker.


    def run(self):
        """
        The main execution loop for the SimpleWorker.

        It continuously waits for tasks, executes them, and then notifies
        its pool upon completion.
        """
        while True:
            self.data_for_work_ready.acquire() # Blocks here until a new task is assigned.

            if self.should_i_stop is True:
                break # Exit loop if shutdown is signaled.
            
            # Acquire a global lock for the specific data location to prevent race conditions.
            self.device.location_locks[self.location].acquire()
            script_data = [] # Collects data from neighbors and self for the script.
            
            # Gathers data from all neighboring devices for the current location.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Gathers data from its own device for the current location.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # If any data was collected, run the script and update devices.
            if script_data != []:
                
                result = self.script.run(script_data) # Executes the script.

                # Updates the data in neighboring devices with the script's result.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                # Updates its own device's data with the script's result.
                self.device.set_data(self.location, result)
            
            # Releases the global lock for the data location.
            self.device.location_locks[self.location].release()
            
            # Notifies the thread pool that this worker has finished its task.
            self.worker_pool.worker_finished(self)
