"""
@05314305-b286-4c5b-a80e-5c46defa6a97/arch/arm/crypto/Makefile device.py
@brief Implements a distributed processing system for managing device operations, job execution, and inter-device synchronization.
This module defines core components such as `Barrier` for thread synchronization, `Job` for encapsulating processing tasks,
`DeviceSync` for managing device-level events and locks, `Worker` for executing individual jobs, `WorkerPool` for managing
worker threads, and `Device` and `DeviceThread` for orchestrating device-specific logic and communication within a network
of devices.
"""


from threading import Event, Thread, Condition, Lock
import Queue

WORKERS_PER_DEVICE = 8
NUM_LOCATION_LOCKS = 25

class Barrier():
    """
    @brief Implements a reusable barrier synchronization mechanism for coordinating multiple threads.
    All participating threads must reach the barrier before any can proceed, ensuring synchronized execution
    at specific points in a concurrent program.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes a new Barrier instance.
        @param num_threads The total number of threads that must reach the barrier.
        """
        self.condition = Condition()
        self.count_threads = 0
        self.num_threads = num_threads

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` threads have reached this barrier.
        Upon reaching the barrier, threads are released, and the barrier resets for future use.
        @pre `num_threads` threads are expected to call this method.
        @post All `num_threads` threads that called `wait()` have been released.
        """
        self.condition.acquire()
        self.count_threads = self.count_threads + 1

        # Block Logic: Checks if the last thread has arrived.
        # If all threads have arrived, release them and reset the barrier.
        if self.count_threads == self.num_threads:
            self.condition.notify_all()
            self.count_threads = 0
        # Otherwise, wait for other threads to arrive.
        else:
            self.condition.wait()

        self.condition.release()

class Job():
    """
    @brief Represents a single processing job within the distributed system.
    Each job encapsulates the necessary context for a worker to perform a computation,
    including its physical location, data from neighboring devices, and the script to execute.
    """
    
    def __init__(self, location, neighbours, script):
        """
        @brief Initializes a new Job instance.
        @param location The specific data location this job pertains to.
        @param neighbours A list of neighboring devices from which to fetch data.
        @param script The processing script (callable) to execute for this job.
        """
        self.location = location
        self.neighbours = neighbours
        self.script = script

    def get_location(self):
        """
        @brief Retrieves the location associated with this job.
        @return The job's location identifier.
        """
        return self.location

    def get_neighbours(self):
        """
        @brief Retrieves the list of neighboring devices for this job.
        @return A list of device objects representing neighbors.
        """
        return self.neighbours

    def get_script(self):
        """
        @brief Retrieves the processing script for this job.
        @return The callable script to be executed.
        """
        return self.script
class DeviceSync(object):
    """
    @brief Manages synchronization primitives and events for a device in a distributed system.
    This class orchestrates inter-thread and inter-device synchronization, including a barrier
    for coordinated thread execution, locks for managing access to specific data locations,
    and events for signaling script reception and setup completion.
    """
    
    def __init__(self):
        """
        @brief Initializes a new DeviceSync instance with default synchronization states.
        The barrier and location locks are initialized to None or empty, to be set up later.
        """
        self.barrier = None
        self.location_locks = []
        self.receive_scripts =  Event()
        self.setup = Event()

    def init_barrier(self, num_threads):
        """
        @brief Initializes the synchronization barrier with a specified number of threads.
        @param num_threads The total number of threads that will participate in this barrier.
        """
        self.barrier = Barrier(num_threads)

    def wait_threads(self):
        """
        @brief Causes the calling thread to wait at the barrier until all participating threads have arrived.
        This ensures coordinated execution across multiple threads/devices.
        """
        self.barrier.wait()

    def init_location_locks(self, num_locations):
        """
        @brief Initializes a list of locks, one for each data location.
        These locks are used to ensure exclusive access to data at specific locations during updates.
        @param num_locations The total number of data locations requiring individual locks.
        """
        # Block Logic: Iteratively creates and appends a new Lock object for each location.
        for location in range(0, num_locations):
            self.location_locks.append(Lock())

    def get_location_lock(self, location):
        """
        @brief Retrieves the lock associated with a specific data location.
        @param location The identifier of the data location.
        @return The `threading.Lock` object for the specified location.
        """
        return self.location_locks[location]


    def set_receive_scripts(self):
        """
        @brief Sets the 'receive_scripts' event, signaling that new scripts are available for processing.
        """
        self.receive_scripts.set()

    def wait_receive_scripts(self):
        """
        @brief Blocks until the 'receive_scripts' event is set, indicating scripts are ready.
        """
        self.receive_scripts.wait()

    def clear_scripts(self):
        """
        @brief Clears the 'receive_scripts' event, resetting its state to 'not set'.
        This prepares the event for the next cycle of script reception.
        """
        self.receive_scripts.clear()

    def set_setup_event(self):
        """
        @brief Sets the 'setup' event, signaling that the device's initial setup is complete.
        """
        self.setup.set()

    def wait_setup_event(self):
        """
        @brief Blocks until the 'setup' event is set, indicating initial setup is complete.
        """
        self.setup.wait()
class Worker(Thread):
    """
    @brief A worker thread responsible for executing individual jobs.
    Each worker continuously fetches jobs from a shared queue, processes them by gathering data
    from its own device and neighbors, executing a script, and updating relevant data.
    """
    
    def __init__(self, device, scripts):
        """
        @brief Initializes a Worker instance.
        @param device The associated `Device` instance this worker belongs to.
        @param scripts The shared `Queue` from which to retrieve `Job` objects.
        """
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts

    def get_neighbours_data(self, job):
        """
        @brief Gathers data from the worker's own device and all neighboring devices specified in the job.
        @param job The `Job` object containing information about the location and neighbors.
        @return A list of data collected from the current device and its neighbors. Empty values (None) are filtered out.
        """
        # Invariant: all_data list is populated with data from neighbors and own device.
        all_data = []

        # Block Logic: Iterate through neighboring devices and retrieve data.
        for device in job.neighbours:
            data = device.get_data(job.get_location())
            all_data.append(data)

        # Retrieve data from the worker's own device.
        data = self.device.get_data(job.get_location())
        all_data.append(data)

        # Filter out any None values (locations with no data) from the collected data.
        all_data = list(filter(None, all_data))

        return all_data

    def update_neighbours_data(self, job, new_data):
        """
        @brief Updates data on the worker's own device and all neighboring devices specified in the job.
        @param job The `Job` object containing information about the location and neighbors.
        @param new_data The data to update at the specified location on all relevant devices.
        """
        # Block Logic: Update data on neighboring devices.
        for device in job.neighbours:
            device.set_data(job.get_location(), new_data)
        # Update data on the worker's own device.
        self.device.set_data(job.get_location(), new_data)

    def run(self):
        """
        @brief The main execution loop for the worker thread.
        Continuously fetches jobs, processes them, and updates data,
        until a termination job (script is None) is received.
        """
        while True:
            # Block Logic: Retrieve a job from the scripts queue.
            job = self.scripts.get()

            # Block Logic: Check for a termination signal (script is None).
            # If a termination job is received, mark the task as done and exit the loop.
            if job.script is None:
                self.scripts.task_done()
                break

            # Block Logic: Acquire a lock for the job's location to ensure data consistency during processing.
            with self.device.syncronization.get_location_lock(job.get_location()):
                data = self.get_neighbours_data(job)
                # Invariant: data might be empty if no data was found in neighbors or own device.
                if data != []:
                    # Execute the job's script with the collected data.
                    new_data = job.script.run(data)
                    # Update the data on the current device and its neighbors with the new results.
                    self.update_neighbours_data(job, new_data)

            # Mark the current job as done in the queue.
            self.scripts.task_done()

class WorkerPool(object):
    """
    @brief Manages a pool of worker threads (`Worker` instances) for a device.
    This class is responsible for creating, starting, and stopping a fixed number of workers,
    and distributing jobs to them via a shared queue.
    """


    def __init__(self, device, num_workers):
        """
        @brief Initializes a WorkerPool instance.
        @param device The `Device` instance that owns this worker pool.
        @param num_workers The number of worker threads to create in the pool.
        """
        self.device = device
        self.num_workers = num_workers

        self.scripts = Queue.Queue() # The queue where jobs (scripts) are placed for workers to pick up.
        self.workers_scripts = [] # A list to hold references to the worker threads.

        self.start_workers()

    def start_workers(self):
        """
        @brief Starts all worker threads in the pool.
        Each worker is initialized with a reference to the device and the shared jobs queue.
        """
        # Block Logic: Iterates to create and start the specified number of worker threads.
        for worker_id in range(0, self.num_workers):
            self.workers_scripts.append(Worker(self.device, self.scripts))
            self.workers_scripts[worker_id].start()

    def join_workers(self):
        """
        @brief Waits for all worker threads to complete their current jobs and terminate.
        This method ensures graceful shutdown by waiting for the jobs queue to empty
        and then for each worker thread to finish.
        """

        # Block Logic: Iterates through workers (excluding the last one, potential off-by-one or specific design choice)
        # to ensure their completion.
        for worker_id in range(0, self.num_workers - 1):
            # Wait until all items in the queue have been gotten and processed.
            self.scripts.join()
            # Block until the thread terminates.
            self.workers_scripts[worker_id].join()

    def stop_workers(self):
        """
        @brief Signals all worker threads to stop by adding termination jobs to the queue.
        After adding termination jobs, it waits for all workers to join, ensuring a clean shutdown.
        """
        # Block Logic: Add a termination Job for each worker.
        # A Job with script=None acts as a sentinel to signal a worker to terminate.
        for worker_id in range(0, WORKERS_PER_DEVICE): # Uses a constant for number of workers here
            self.add_job(Job(None, None, None))

        # Wait for all worker threads to process their termination jobs and stop.
        self.join_workers()

    def delete_workers(self):
        """
        @brief Removes worker instances from the internal list.
        Note: This does not stop the worker threads; `stop_workers` should be called first.
        """
        # Block Logic: Iteratively removes worker instances from the end of the list.
        for worker_id in range(0, self.num_workers - 1):
            del self.workers_scripts[-1]

    def add_job(self, job):
        """
        @brief Adds a job to the shared queue for processing by a worker thread.
        @param job The `Job` object to be added to the queue.
        """
        self.scripts.put(job)

class Device(object):
    """
    @brief Represents a single device in the distributed processing system.
    Each device manages its own sensor data, a pool of worker threads to process jobs,
    and synchronization mechanisms for coordinating with other devices and a supervisor.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id A unique identifier for this device.
        @param sensor_data A dictionary or similar structure holding sensor data for various locations.
        @param supervisor The supervisor object responsible for managing the network of devices.
        """
        self.device_id = device_id
        self.scripts = [] # Stores scripts assigned to this device, awaiting processing.
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.syncronization = DeviceSync() # Manages synchronization primitives for this device.
        self.worker_pool = WorkerPool(self, WORKERS_PER_DEVICE) # Manages worker threads for job execution.
        self.thread = DeviceThread(self) # The main thread for this device's operations.
        self.thread.start() # Starts the device's main operational thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device [device_id]".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs initial setup for all devices in the network.
        This method is designed to be called by only one designated device (e.g., the last device in the list)
        to initialize shared synchronization resources across all devices.
        @param devices A list of all `Device` objects in the network.
        """
        # Pre-condition: This block ensures that device setup happens only once, typically initiated by a designated device.
        # This prevents redundant initialization of shared synchronization resources.
        if self.device_id == len(devices) - 1: # Logic: Only the last device (based on its ID) performs the setup.
            self.syncronization.init_barrier(len(devices)) # Initialize a barrier for all devices.
            self.syncronization.init_location_locks(NUM_LOCATION_LOCKS) # Initialize locks for data locations.

            # Block Logic: Propagate the initialized synchronization objects and signal setup completion to all devices.
            for device in devices:
                device.syncronization.barrier = self.syncronization.barrier # Assign the shared barrier.
                device.syncronization.location_locks = self.syncronization.location_locks # Assign the shared location locks.
                device.syncronization.set_setup_event() # Signal that setup is complete for this device.

    def assign_script(self, script, location):
        """
        @brief Assigns a processing script to a specific location on this device.
        If the script is None, it signals that script reception is complete for this device.
        @param script The script (callable) to assign, or None to signal completion.
        @param location The data location associated with the script.
        """
        # Block Logic: Assign the script if it's not a termination signal.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If script is None, it's a signal that no more scripts will be assigned in this phase.
            self.syncronization.set_receive_scripts()

    def add_job(self, job):
        """
        @brief Adds a job to the device's worker pool for asynchronous processing.
        @param job The `Job` object to be processed by a worker.
        """
        self.worker_pool.add_job(job)

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        @param location The identifier of the data location.
        @return The sensor data for the given location, or None if the location is not found.
        """
        # Inline: Safely access sensor_data dictionary to avoid KeyError.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.
        @param location The identifier of the data location.
        @param data The new data to set for the specified location.
        """
        # Block Logic: Only update data if the location already exists in sensor_data.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device, waiting for its main thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main thread of execution for a single device.
    This thread manages the device's lifecycle, including waiting for setup completion,
    coordinating with the supervisor to get neighbor information, synchronizing with
    other device threads, and dispatching assigned scripts to the worker pool.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a DeviceThread instance.
        @param device The `Device` instance that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the device thread.
        This loop orchestrates the device's operational cycle, including synchronization
        with other devices and dispatching jobs to its worker pool.
        """

        # Block Logic: Wait for the device's initial setup to be completed.
        self.device.syncronization.wait_setup_event()

        # Invariant: The device is fully set up, and ready to enter its operational cycle.
        while True:
            # Block Logic: Obtain information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Pre-condition: Check if a termination signal (neighbours is None) has been received.
            if neighbours is None:
                # If termination signal, stop the worker pool and break out of the loop.
                self.device.worker_pool.stop_workers()
                break

            # Block Logic: Synchronize with other device threads using a barrier, ensuring all devices
            # are at the same operational phase before proceeding.
            self.device.syncronization.wait_threads()
            # Block Logic: Wait for scripts to be received and assigned to this device.
            self.device.syncronization.wait_receive_scripts()

            # Block Logic: Iterate through assigned scripts and add them as jobs to the worker pool.
            for (script, location) in self.device.scripts:
                self.device.add_job(Job(location, neighbours, script))

            # Block Logic: Synchronize again, ensuring all devices have dispatched their jobs.
            self.device.syncronization.wait_threads()
            # Block Logic: Clear the script reception signal, preparing for the next cycle.
            self.device.syncronization.clear_scripts()
