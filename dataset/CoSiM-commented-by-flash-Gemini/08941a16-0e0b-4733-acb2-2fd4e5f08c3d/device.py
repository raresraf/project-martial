"""
This module implements a distributed device simulation framework, utilizing a master-worker
threading model and a reusable barrier for synchronization.

Algorithm:
- ReusableBarrier: A synchronization primitive ensuring all participating threads reach
  a specific point before any can proceed.
- Device: Represents a simulated physical device with sensor data and script processing capabilities.
- DeviceThreadMaster: Acts as the orchestrator for each device, fetching data, queueing tasks,
  and managing overall timepoint synchronization.
- DeviceThreadWorker: Worker threads that consume tasks (scripts) from a queue, execute them
  on collected data, and update device states in a thread-safe manner using location-specific locks.
- Producer-Consumer Pattern: `DeviceThreadMaster` produces tasks into a `Queue`, and `DeviceThreadWorker`s
  consume them.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue, Empty


class ReusableBarrier(object):
    """
    Implements a reusable barrier synchronization mechanism using a Condition variable.

    This barrier allows a specified number of threads to wait until all have
    reached a certain point, after which all are released simultaneously. It
    is designed for repeated use without reinitialization.

    Attributes:
        num_threads (int): The total number of threads expected to participate in the barrier.
        count_threads (int): The current count of threads waiting at the barrier.
        cond (threading.Condition): The condition variable used for signaling and waiting.
    """

    def __init__(self, num_threads):
        """
        Initializes a new instance of the ReusableBarrier.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can pass.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        # Initialize a Condition variable, which internally uses a Lock.
        self.cond = Condition()

    def wait(self):
        """
        Causes the calling thread to wait until all `num_threads` threads
        have reached this barrier.

        Pre-condition: The thread holds the `cond` lock implicitly before waiting.
        Invariant: The `count_threads` is atomically decremented. When it reaches zero,
                   all waiting threads are notified, and the `count_threads` is reset.
        """
        self.cond.acquire()
        try:
            self.count_threads -= 1
            # If this thread is the last to reach the barrier:
            if self.count_threads == 0:
                self.cond.notify_all() # Release all waiting threads.
                self.count_threads = self.num_threads # Reset for next use.
            else:
                self.cond.wait() # Wait until signaled by the last thread.
        finally:
            self.cond.release()


class Device(object):
    """
    Represents a simulated device in a distributed environment.

    Each device has a unique ID, manages its sensor data, and can receive
    scripts for processing. It employs a master-worker threading model
    and uses shared synchronization primitives (a reusable barrier and locks)
    to coordinate operations with other devices.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings,
                            where keys represent locations.
        supervisor (Supervisor): A reference to the central supervisor managing devices.
        scripts (list): A list to temporarily store assigned scripts (script, location) tuples.
        neighbours (list): A list of neighboring Device objects (populated by the master thread).
        barrier (ReusableBarrier): A shared barrier for synchronizing all devices at timepoint end.
        locks (list): A list of `threading.Lock` objects, one for each data location,
                      to ensure exclusive access during script execution.
        timepoint_done (threading.Event): Event to signal when script assignments for a timepoint are complete.
        tasks_ready (threading.Event): Event to signal worker threads that tasks are available in the queue.
        tasks (Queue): A thread-safe queue for scripts to be processed by worker threads.
        simulation_ended (bool): A flag to signal all threads when the simulation has ended.
        master (DeviceThreadMaster): The dedicated master orchestrating thread for this device.
        workers (list): A list of `DeviceThreadWorker` instances forming the worker pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): Initial sensor data for the device.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # List to hold (script, location) tuples assigned to this device.
        self.scripts = []
        self.neighbours = [] # Populated by DeviceThreadMaster.

        self.barrier = None # Will be set by setup_devices.
        # Locks for data locations, initialized by setup_devices.
        self.locks = []
        # Event for signaling that script assignments for a timepoint are complete.
        self.timepoint_done = Event()
        # Event for signaling worker threads that tasks are ready in the queue.
        self.tasks_ready = Event()
        # Thread-safe queue for scripts to be processed by worker threads.
        self.tasks = Queue()
        self.simulation_ended = False

        # Create and start the master orchestrating thread for this device.
        self.master = DeviceThreadMaster(self)
        self.master.start()

        # Create and start a pool of worker threads.
        self.workers = []
        for i in xrange(8): # xrange for Python 2 compatibility.
            worker = DeviceThreadWorker(self, i)
            self.workers.append(worker)
            worker.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device [<device_id>]".
        """
        return "Device [%d]" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization primitives (a reusable barrier and location-specific locks)
        across all devices. This method ensures these resources are only initialized once
        by the device with `device_id == 0`.

        Pre-condition: This method should be called by a central entity (e.g., supervisor)
                       after all devices are instantiated.
        Invariant: If the current device is the designated initializer, a new `ReusableBarrier`
                   and a list of `Lock` objects (24 for 24 locations) are created. These shared
                   resources are then propagated to all `Device` instances.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Only the device with ID 0 initializes shared resources.
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            # Initialize 24 location-specific locks.
            locks = [Lock() for _ in xrange(24)] # xrange for Python 2 compatibility.
            # Propagate the created barrier and locks to all devices.
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed on data at a specific location for this device.
        If `script` is None, it signals that script assignments for the current timepoint are complete.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (int): The data location where the script should be applied.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that script assignment for the current timepoint is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (int): The location from which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location
                 does not exist in the device's sensor_data.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        Args:
            location (int): The location at which to set data.
            data (any): The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device by joining its master
        and worker threads.
        """
        self.master.join()
        for worker in self.workers:
            worker.join()


class DeviceThreadMaster(Thread):
    """
    The dedicated master orchestrating thread for a `Device` object.

    This thread is responsible for:
    1. Interacting with the supervisor to get neighbor information.
    2. Acting as a producer by pushing assigned scripts into a shared queue.
    3. Managing the simulation flow for its associated device.
    4. Synchronizing with other `DeviceThreadMaster` instances using a shared barrier.

    Attributes:
        device (Device): The Device object associated with this master thread.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThreadMaster instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        # Initialize the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device [%d] Thread Master" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThreadMaster.

        Invariant: The loop continues until the supervisor signals termination.
                   Within each iteration, it retrieves neighbor information,
                   waits for script assignments, puts these scripts into the
                   shared queue, signals worker threads that tasks are ready,
                   waits for all tasks to be processed, and then synchronizes
                   with other master threads via a barrier.
        """
        while True:
            # Retrieve information about neighboring devices from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: If no neighbors are returned (e.g., simulation termination signal),
            # set the simulation_ended flag, signal worker threads, and break the loop.
            if self.device.neighbours is None:
                self.device.simulation_ended = True
                self.device.tasks_ready.set() # Wake up workers to see simulation_ended flag.
                break

            # Wait for the `timepoint_done` event, which signals that script assignments
            # for the current timepoint are complete.
            self.device.timepoint_done.wait()

            # Block Logic: Populate the shared queue with all assigned scripts.
            # This acts as the producer part of the producer-consumer pattern.
            for task in self.device.scripts:
                self.device.tasks.put(task)

            # Signal worker threads that tasks are ready in the queue.
            self.device.tasks_ready.set()

            # Wait until all tasks in the queue have been processed by worker threads.
            self.device.tasks.join()

            # Clear the `tasks_ready` event and `timepoint_done` event, resetting them for the next timepoint.
            self.device.tasks_ready.clear()
            self.device.timepoint_done.clear()
            # Clear the list of scripts for the next timepoint.
            self.device.scripts = [] # Functional utility: Clear the processed scripts list.

            # Synchronize with other DeviceThreadMaster instances using the shared barrier,
            # ensuring all devices complete their timepoint processing before proceeding.
            self.device.barrier.wait()


class DeviceThreadWorker(Thread):
    """
    A worker thread responsible for executing scripts from a shared task queue.

    These threads act as consumers in the master-worker pattern, processing scripts
    assigned to their associated `Device`. They use location-specific locks to
    ensure thread-safe updates to sensor data.

    Attributes:
        device (Device): The Device object associated with this worker thread.
        thread_id (int): A unique identifier for this worker thread within its device.
    """

    def __init__(self, device, thread_id):
        """
        Initializes a new DeviceThreadWorker instance.

        Args:
            device (Device): The Device associated with this worker.
            thread_id (int): A unique ID for this worker thread.
        """
        # Initialize the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device [%d] Thread Worker [%d]" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        The main execution method for the DeviceThreadWorker.

        Invariant: The loop continuously tries to get tasks from the queue until
                   `device.simulation_ended` flag is set and the queue is empty.
                   Each task involves waiting for tasks to be ready, acquiring
                   a location-specific lock, collecting data, executing the script,
                   updating data, releasing the lock, and marking the task as done.
        """
        while not self.device.simulation_ended:
            # Wait for the `tasks_ready` event, which signals that tasks are available in the queue.
            self.device.tasks_ready.wait()

            try:
                # Attempt to get a script task from the queue without blocking indefinitely.
                script, location = self.device.tasks.get(block=False)

                # Acquire a location-specific lock to ensure exclusive access for data at this location.
                self.device.locks[location].acquire()

                script_data = []

                # Block Logic: Collect data from neighboring devices for the current location.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Collect data from the current device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: If there is data collected, execute the script and update results.
                if len(script_data) > 0:
                    # Execute the script with the collected data.
                    result = script.run(script_data)

                    # Update the sensor data in neighboring devices.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    # Update the sensor data in the current device.
                    self.device.set_data(location, result)

                # Release the location-specific lock.
                self.device.locks[location].release()

                # Mark the task as done in the queue.
                self.device.tasks.task_done()
            except Empty:
                # If the queue is empty, and simulation is not ended, just pass and wait for next signal.
                pass
