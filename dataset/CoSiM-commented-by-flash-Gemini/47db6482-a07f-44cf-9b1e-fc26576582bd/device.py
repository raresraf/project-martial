

"""
This module implements a device simulation framework utilizing a master-worker
threading pattern and a reusable barrier for synchronization. It defines
`Device` objects, master and worker threads for managing device operations
and script execution, and a custom `ReusableBarrier` for inter-thread coordination.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue, Empty


class ReusableBarrier(object):
    """
    A synchronization primitive that allows multiple threads to wait for each
    other to reach a common point before any of them can proceed. This barrier
    can be reset and reused for subsequent synchronization points.
    """

    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrier instance.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can proceed.
        """
        self.num_threads = num_threads

        # Functional Utility: `count_threads` tracks how many threads have reached the barrier.
        self.count_threads = self.num_threads
        # Functional Utility: `cond` is the condition variable used for blocking and notifying threads.
        self.cond = Condition()

    def wait(self):
        """
        Causes the calling thread to wait until all other threads have also
        called `wait()` on this barrier.
        """
        self.cond.acquire() # Acquire the lock associated with the condition variable.
        self.count_threads -= 1 # Decrement the count of threads yet to reach the barrier.
        # Pre-condition: Checks if this is the last thread to arrive at the barrier.
        if self.count_threads == 0:
            self.cond.notify_all() # Functional Utility: Wakes up all threads waiting on this condition.
            self.count_threads = self.num_threads # Functional Utility: Resets the counter for reuse of the barrier.
        else:
            self.cond.wait() # Block Logic: Waits until notified by the last thread to arrive.
        self.cond.release() # Release the lock.


class Device(object):
    """
    Represents a simulated device in a distributed system. Each device manages
    its own sensor data, processes assigned scripts, and coordinates its
    operations through a dedicated master thread, worker threads, and
    shared synchronization primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor data, keyed by location.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List of (script, location) tuples assigned to this device.
        self.neighbours = [] # List of neighboring devices, updated by the master thread.

        self.barrier = None # Barrier for synchronizing device threads at timepoints.
        self.locks = [] # List of locks, one per location, for fine-grained access control.
        self.timepoint_done = Event() # Signals completion of script assignments for a timepoint.
        self.tasks_ready = Event() # Signals that tasks are ready in the queue for workers.
        self.tasks = Queue() # Queue for distributing scripts to worker threads.
        self.simulation_ended = False # Flag to signal termination of the simulation.

        # Functional Utility: `master` is the Device's main thread, responsible for
        # fetching neighbor data and distributing tasks.
        self.master = DeviceThreadMaster(self)
        self.master.start()

        # Block Logic: Initializes and starts a pool of worker threads.
        # Functional Utility: `workers` are DeviceThreadWorker instances responsible
        # for executing scripts from the task queue.
        self.workers = []
        for i in xrange(8): # Creates 8 worker threads.
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
        Configures the device's synchronization mechanisms.
        Device 0 initializes the shared barrier and per-location locks,
        which are then distributed to all other devices.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        # Block Logic: Device 0 acts as the coordinator to initialize the shared barrier and locks.
        if self.device_id == 0:
            # Functional Utility: Creates a `ReusableBarrier` instance for global synchronization
            # across all device threads.
            barrier = ReusableBarrier(len(devices))
            # Functional Utility: Creates a fixed number of `Lock` objects to protect access
            # to shared resources across different locations.
            locks = [Lock() for _ in xrange(24)] # Assumes a maximum of 24 locations.
            # Block Logic: Propagates the initialized barrier and locks to all other devices.
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location for this device.
        If no script is provided (None), it signals that the timepoint is done.

        Args:
            script (Script or None): The script object to assign, or None if the timepoint is complete.
            location (int): The numerical identifier for the location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Functional Utility: Signals that all script assignments for the current timepoint are complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location for which to retrieve data.

        Returns:
            any: The sensor data for the specified location, or None if not found.
        """
        # Pre-condition: Checks if the location exists in the device's sensor data.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location. Access to `sensor_data` is protected
        implicitly by the location-specific locks managed by the `DeviceThreadWorker`.

        Args:
            location (int): The location for which to set data.
            data (any): The new data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device's master and worker threads, waiting for their completion.
        """
        self.master.join() # Waits for the master thread to finish.
        # Block Logic: Iterates through all worker threads and waits for each to finish.
        for worker in self.workers:
            worker.join()


class DeviceThreadMaster(Thread):
    """
    The master thread for a Device. It is responsible for fetching neighbor
    information from the supervisor, distributing assigned scripts to the
    worker queue, and coordinating synchronization at each timepoint.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThreadMaster instance.

        Args:
            device (Device): The Device instance this master thread is managing.
        """
        Thread.__init__(self, name="Device [%d] Thread Master" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the master thread. It continuously
        fetches neighbor data, processes script assignments, pushes them
        to the task queue for workers, and synchronizes with other devices.
        """
        while True:
            # Block Logic: Fetches the current set of neighboring devices from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # Pre-condition: Checks if a shutdown signal has been received from the supervisor.
            if self.device.neighbours is None:
                # Functional Utility: Sets the `simulation_ended` flag to true, signaling
                # all worker threads to terminate.
                self.device.simulation_ended = True
                # Functional Utility: Notifies worker threads that tasks are ready (or that
                # the simulation has ended and they should check the flag).
                self.device.tasks_ready.set()
                
                break

            # Block Logic: Halts execution until the current timepoint's scripts
            # have been assigned and marked as ready by the device.
            self.device.timepoint_done.wait()

            # Block Logic: Puts each assigned script into the shared task queue
            # for worker threads to pick up.
            for task in self.device.scripts:
                self.device.tasks.put(task)

            # Functional Utility: Notifies worker threads that new tasks are available in the queue.
            self.device.tasks_ready.set()

            # Block Logic: Waits for all tasks submitted for the current timepoint
            # to be completed by the worker threads.
            self.device.tasks.join()

            # Functional Utility: Clears the `tasks_ready` event, indicating no new tasks
            # are immediately available and workers should wait.
            self.device.tasks_ready.clear()

            # Functional Utility: Clears the `timepoint_done` event, preparing for the
            # next cycle of script assignment.
            self.device.timepoint_done.clear()

            # Functional Utility: Synchronizes with all other DeviceThreadMaster instances
            # across devices using a shared barrier, ensuring all devices complete
            # their current timepoint processing before proceeding.
            self.device.barrier.wait()


class DeviceThreadWorker(Thread):
    """
    A worker thread for a Device. It continuously fetches scripts from a
    shared queue, executes them, interacts with local and neighboring
    sensor data, and ensures thread-safe access using per-location locks.
    """

    def __init__(self, device, thread_id):
        """
        Initializes a new DeviceThreadWorker instance.

        Args:
            device (Device): The Device instance this worker thread is associated with.
            thread_id (int): A unique identifier for this worker thread.
        """
        Thread.__init__(self, name="Device [%d] Thread Worker [%d]" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        The main execution loop for the worker thread. It repeatedly waits for
        tasks, retrieves them from the queue, executes the associated script,
        and updates sensor data, all while respecting shared locks.
        """
        # Invariant: The loop continues as long as the simulation is not ended.
        while not self.device.simulation_ended:
            # Block Logic: Waits until the master thread signals that new tasks are ready.
            self.device.tasks_ready.wait()

            try:
                # Functional Utility: Attempts to retrieve a script task from the queue.
                # `block=False` ensures it doesn't block indefinitely if the queue is empty.
                script, location = self.device.tasks.get(block=False)

                # Functional Utility: Acquires a lock for the specific location to ensure
                # exclusive access to the sensor data during script execution.
                self.device.locks[location].acquire()

                script_data = [] # Accumulator for all data relevant to the script.

                # Block Logic: Gathers sensor data from neighboring devices at the specified location.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Block Logic: Retrieves the local device's sensor data for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: Checks if there is any data collected for the script to run.
                if len(script_data) > 0:
                    # Functional Utility: Executes the assigned script with the collected data,
                    # simulating sensor data processing.
                    result = script.run(script_data)

                    # Block Logic: Updates the sensor data of neighboring devices with the script's result.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Updates the local device's sensor data with the script's result.
                    self.device.set_data(location, result)

                # Functional Utility: Releases the lock for the current location, allowing other
                # threads to access the sensor data.
                self.device.locks[location].release()

                # Functional Utility: Marks the current task as done in the queue,
                # signaling completion to the `DeviceThreadMaster`.
                self.device.tasks.task_done()
            except Empty:
                # Block Logic: Catches the `Empty` exception if no tasks are available when
                # `get(block=False)` is called. The worker then re-enters the loop to wait
                # for `tasks_ready` to be set again or for the simulation to end.
                pass
