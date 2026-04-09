

"""
This module implements a multi-threaded simulation framework for distributed devices.
Each `Device` object represents a simulated entity that manages its sensor data and executes
assigned scripts. The framework leverages Python's `threading` module to enable
concurrent operations, with synchronization handled by a `ReusableBarrier` and various `Lock` objects.

Key Components:
- `ReusableBarrier`: A custom implementation of a barrier that allows multiple threads
  to synchronize at a specific point in their execution, and can be used repeatedly.
- `Device`: Represents a single node in the distributed system. It holds device-specific
  information, sensor data, and manages its own thread of execution (`DeviceThread`).
- `DeviceThread`: The main operational thread for each `Device`. It coordinates the
  fetching of neighbor data, distributes script execution to `DeviceWorker` threads,
  and manages synchronization across devices for time-step progression.
- `DeviceWorker`: A worker thread responsible for executing a subset of scripts for a
  `Device`. It collects relevant data, runs the script, and updates sensor data.

Architecture:
The simulation proceeds in time steps. At each step, all `DeviceThread` instances
synchronize using a shared barrier. Each `DeviceThread` then fetches data from
its neighbors (if any) via a `supervisor` (not defined in this file but assumed).
Scripts assigned to a device are then distributed among a pool of `DeviceWorker`
threads for parallel execution. After all `DeviceWorker` threads complete their tasks,
the `DeviceThread` again synchronizes with other devices via the barrier before
proceeding to the next time step. This architecture ensures data consistency and
efficient parallel processing in a distributed simulation environment.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue


class ReusableBarrier(object):
    """
    Implements a reusable barrier for synchronizing multiple threads.

    Threads wait at the barrier until all participating threads have arrived,
    after which all are released. This barrier is designed to be used
    multiple times, facilitating cyclic synchronization in simulations or
    iterative algorithms.
    """

    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        :param num_threads: The total number of threads that must reach the barrier
                            before all waiting threads are released.
        """
        self.num_threads = num_threads
        # Two sets of counters and semaphores are used to make the barrier reusable.
        # Threads alternate between `count_threads1`/`threads_sem1` and `count_threads2`/`threads_sem2`
        # in successive barrier synchronizations.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # A lock to protect access to the thread counters, preventing race conditions.
        self.count_lock = Lock()
        # Semaphores for threads to wait on in the first and second phases.
        # Initially, no permits are available, so threads will block until released.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        participating threads have also called `wait()`.

        This method orchestrates the two phases of the reusable barrier
        to ensure proper synchronization and reset for subsequent uses.
        """
        # Execute the first phase of the barrier.
        self.phase(self.count_threads1, self.threads_sem1)
        # Execute the second phase of the barrier.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Manages a single phase of the reusable barrier.

        This internal method is called twice by `wait()` for each barrier synchronization.
        It decrements a shared counter and either releases all waiting threads
        (if the counter reaches zero) or causes the current thread to wait.

        :param count_threads: A list containing the shared counter for this phase.
                              (Using a list to allow modification within `with` statement).
        :param threads_sem: The semaphore associated with this phase for threads to wait on.
        """
        # Ensure exclusive access to the shared counter.
        with self.count_lock:
            count_threads[0] -= 1  # Decrement the count of threads yet to reach the barrier.

            # If this is the last thread to reach the barrier, release all waiting threads.
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()  # Release all threads waiting on this semaphore.
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # Acquire the semaphore; this thread will either pass immediately (if released by the last thread)
        # or block until it's released by the last thread to enter this phase.
        threads_sem.acquire()


class Device(object):
    """
    Represents a single simulated device in a distributed system.

    Each device has a unique identifier, stores its own sensor data,
    interacts with a `supervisor` to get neighbor information, and
    executes assigned scripts. It manages its own dedicated thread
    (`DeviceThread`) and utilizes various synchronization primitives
    for coordinated operations within the simulation framework.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        :param device_id: A unique integer identifier for this device.
        :param sensor_data: A dictionary containing the sensor data for this device,
                            where keys represent locations/data points.
        :param supervisor: An object representing the supervisor responsible for
                           managing devices and providing information like neighbors.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # A queue to potentially store results or messages for this device.
        self.result_queue = Queue.Queue()
        # A lock to protect modifications to the device's sensor data, ensuring thread safety.
        self.set_lock = Lock()
        # Synchronization primitives shared across all devices, initialized in `setup_devices`.
        self.neighbours_lock = None
        self.neighbours_barrier = None

        # Event to signal that new scripts have been received and are ready for processing.
        self.script_received = Event()
        # List to hold tuples of (script, location) assigned to this device for execution.
        self.scripts = []
        # Event to signal that all processing for the current timepoint is complete.
        self.timepoint_done = Event() # This event is initialized but not explicitly used to `wait()` in this class.

        # The dedicated thread responsible for managing this device's operations.
        self.thread = DeviceThread(self)

    def __str__(self):
        """
        Returns a human-readable string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared synchronization mechanisms (lock and barrier) among all devices
        and starts this device's dedicated thread.

        The device with the smallest `device_id` (assumed to be `devices[0]`) initializes
        these shared objects, and other devices then reference them.

        :param devices: A list of all `Device` objects participating in the simulation.
        """
        # The first device in the sorted list (based on on `device_id` if there was a comparator)
        # creates the shared lock and barrier. This assumes `devices[0]` is indeed the "first" device.
        if self.device_id == devices[0].device_id:
            self.neighbours_lock = Lock()  # Lock to protect access to neighbor information from the supervisor.
            self.neighbours_barrier = ReusableBarrier(len(devices))  # Barrier for global device synchronization.
        # Subsequent devices reference the shared lock and barrier created by the first device.
        else:
            self.neighbours_lock = devices[0].neighbours_lock
            self.neighbours_barrier = devices[0].neighbours_barrier

        # Start the device's dedicated operational thread.
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a particular sensor data location.

        If `script` is `None`, it acts as a signal that no more scripts are coming
        for the current timepoint, and signals `script_received` and `timepoint_done`.

        :param script: The script object (presumably with a `run` method) to execute.
        :param location: The key in `sensor_data` that the script pertains to.
        """
        if script is not None:
            self.scripts.append((script, location))  # Add the script and its location to the device's queue.
            self.script_received.set()  # Signal that scripts are available for processing.
        else:
            # Signal that script reception is done for this timepoint and processing can begin.
            self.script_received.set()
            # Signal that this device considers its timepoint processing complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves the sensor data for a specific location from this device.

        :param location: The key for the desired sensor data.
        :return: The sensor data at `location` if it exists, otherwise `None`.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specific location on this device.

        Access to `sensor_data` is protected by `self.set_lock` to ensure thread safety
        during data modification.

        :param location: The key for the sensor data to be updated.
        :param data: The new value for the sensor data.
        """
        # Acquire the lock to ensure exclusive access for writing to `sensor_data`.
        self.set_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.set_lock.release()  # Release the lock after modification.

    def shutdown(self):
        """
        Initiates a graceful shutdown of the device by waiting for its
        dedicated `DeviceThread` to complete its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a `Device`.

    It orchestrates the device's participation in the simulation by:
    - Synchronizing with other devices using a shared barrier.
    - Retrieving neighbor information from the `supervisor`.
    - Distributing assigned scripts among a pool of `DeviceWorker` threads for parallel execution.
    - Waiting for the completion of script execution by its workers.
    - Synchronizing again with other devices before advancing to the next time step.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        :param device: The `Device` object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # A list to hold `DeviceWorker` instances for parallel script execution.
        self.workers = []

    def run(self):
        """
        The main loop of the DeviceThread.

        This loop represents the progression of time steps in the simulation.
        It continuously performs synchronization, data acquisition, script dispatch,
        and result processing until the simulation is signaled to end.
        """
        while True:
            # Block Logic: Acquire the `neighbours_lock` to ensure exclusive access
            # while retrieving neighbor information from the supervisor. This prevents
            # race conditions if multiple devices try to access the supervisor concurrently.
            self.device.neighbours_lock.acquire()

            # Functional Utility: Get the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Release the lock after accessing neighbor information.
            self.device.neighbours_lock.release()

            # Pre-condition: If `get_neighbours()` returns `None`, it signifies the end of the simulation.
            if neighbours is None:
                break  # Exit the main simulation loop.

            # Block Logic: Wait until scripts for the current timepoint have been assigned.
            # This ensures that `self.device.scripts` is populated before proceeding.
            self.device.script_received.wait()

            # Functional Utility: Clear the list of workers from the previous time step.
            self.workers = []
            # Block Logic: Create a fixed pool of `DeviceWorker` threads.
            # Assuming a pool size of 8 as an example.
            for i in range(8):
                self.workers.append(DeviceWorker(self.device, i, neighbours))

            # Block Logic: Distribute the scripts assigned to this device among the worker threads.
            # The goal is to optimize by assigning scripts pertaining to the same location
            # to the same worker, or balancing load if no such affinity exists.
            for (script, location) in self.device.scripts:
                # Functional Utility: Flag to track if the script was added to an existing worker.
                added = False
                # Attempt to assign the script to a worker that already handles its location.
                for worker in self.workers:
                    if location in worker.locations:
                        worker.add_script(script, location)
                        added = True
                        break  # Script assigned, move to the next script.

                # If no existing worker handles this location, assign it to the worker with the fewest locations.
                if added == False:
                    minimum = len(self.workers[0].locations)
                    chosen_worker = self.workers[0]
                    for worker in self.workers:
                        if minimum > len(worker.locations):
                            minimum = len(worker.locations)
                            chosen_worker = worker

                    chosen_worker.add_script(script, location)

            # Block Logic: Start all `DeviceWorker` threads to begin parallel script execution.
            for worker in self.workers:
                worker.start()

            # Block Logic: Wait for all `DeviceWorker` threads to complete their assigned tasks.
            # This ensures all scripts for the current timepoint are executed before advancing.
            for worker in self.workers:
                worker.join()

            # Block Logic: Synchronize all `DeviceThread` instances using the shared barrier.
            # All devices must reach this point before any can proceed to the next time step.
            self.device.neighbours_barrier.wait()
            # Functional Utility: Clear the `script_received` event, indicating readiness for new script assignments.
            self.device.script_received.clear()


class DeviceWorker(Thread):
    """
    A worker thread dedicated to executing a subset of scripts for a `Device`.

    Each worker processes scripts related to specific data locations,
    collects data from the local device and its neighbors, runs the script,
    and updates the relevant sensor data.
    """

    def __init__(self, device, worker_id, neighbours):
        """
        Initializes a DeviceWorker thread.

        :param device: The `Device` object that owns this worker.
        :param worker_id: A unique integer identifier for this worker thread within its device.
        :param neighbours: A list of `Device` objects considered neighbors to the owning device.
        """
        Thread.__init__(self)
        self.device = device
        self.worker_id = worker_id
        # Lists to store the scripts and their associated locations assigned to this worker.
        self.scripts = []
        self.locations = []
        self.neighbours = neighbours

    def add_script(self, script, location):
        """
        Assigns a script and its corresponding location to this worker for execution.

        :param script: The script object to be executed.
        :param location: The sensor data location relevant to this script.
        """
        self.scripts.append(script)
        self.locations.append(location)

    def run_scripts(self):
        """
        Executes all scripts assigned to this worker thread.

        For each script, it gathers data from the owning device and its neighbors
        for the specified location, runs the script with this data, and then
        updates the sensor data on both the owning device and its neighbors
        with the script's result.
        """
        for (script, location) in zip(self.scripts, self.locations):
            script_data = []

            # Block Logic: Collects data from neighboring devices for the current script's location.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Functional Utility: Collects data from the owning device for the current script's location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Pre-condition: Checks if any data was collected to proceed with script execution.
            if script_data != []:
                # Functional Utility: Executes the script with the aggregated sensor data.
                res = script.run(script_data)

                # Block Logic: Updates the sensor data on neighboring devices with the result of the script.
                for device in self.neighbours:
                    device.set_data(location, res)
                # Functional Utility: Updates the sensor data on the owning device with the result.
                self.device.set_data(location, res)

    def run(self):
        """
        The entry point for the DeviceWorker thread. It simply calls `run_scripts`
        to execute all assigned scripts.
        """
        self.run_scripts()
