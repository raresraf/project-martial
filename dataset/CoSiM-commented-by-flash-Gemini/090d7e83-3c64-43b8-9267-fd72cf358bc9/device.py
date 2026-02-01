


from threading import Event, Thread, Lock, Semaphore, Condition
class ReusableBarrierCond(object):
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
        Initializes a new instance of the ReusableBarrierCond.

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
        script_received (threading.Event): Event to signal when new scripts are assigned.
        scripts (list): A list to temporarily store assigned scripts (script, location) tuples.
        timepoint_done (threading.Event): Event to signal that all scripts for a timepoint are ready for processing.
        loc_lock (list): A list of `threading.Lock` objects, one for each data location,
                         to ensure exclusive access during script execution.
        crt_nb_scripts (int): Counter for the number of scripts assigned in the current timepoint.
        crt_script (int): Index of the current script being processed within the `scripts` list.
        script_sem (threading.Semaphore): Semaphore to control access to `crt_script` and `scripts`.
        done_processing (threading.Semaphore): Semaphore to signal that processing for a timepoint is complete.
        wait_for_next_timepoint (threading.Event): Event to control script assignment rate.
        thread (DeviceThread): The dedicated master orchestrating thread for this device.
        barr (ReusableBarrierCond): A shared reusable barrier for inter-device synchronization.
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
        # Event for signaling when new scripts have been received.
        self.script_received = Event()
        # List to hold (script, location) tuples assigned to this device.
        self.scripts = []
        # Event for signaling that all scripts for a timepoint are ready for processing.
        self.timepoint_done = Event()
        # List of location-specific locks, initialized by setup_devices.
        self.loc_lock = []
        # Counter for total scripts assigned for the current timepoint.
        self.crt_nb_scripts = 0
        # Index of the current script being processed.
        self.crt_script = 0
        # Semaphore to protect access to `crt_script` and `scripts` list.
        self.script_sem = Semaphore(value=1)
        # Semaphore used to signal that all scripts for a timepoint have been processed.
        self.done_processing = Semaphore(value=0)
        # Event to control the rate of script assignment, ensuring only one timepoint is active.
        self.wait_for_next_timepoint = Event()
        self.wait_for_next_timepoint.set() # Initially allow script assignment.
        self.barr = None # Will be set by setup_devices.

        # Create and start the master orchestrating thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization primitives (a reusable barrier and location-specific locks)
        across all devices. This method ensures these resources are only initialized once
        by the device with `device_id == 0`.

        Pre-condition: This method should be called by a central entity (e.g., supervisor)
                       after all devices are instantiated.
        Invariant: If the current device is the designated initializer, a new `ReusableBarrierCond`
                   is created, and a list of `Lock` objects (for 100 locations) is initialized. These shared
                   resources are then propagated to all `Device` instances.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Create a new reusable barrier. This instance will be shared.
        my_barrier = ReusableBarrierCond(len(devices))
        # Block Logic: Only the device with ID 0 initializes shared resources.
        if self.device_id == 0:
            # Propagate the created barrier to all devices.
            for i in range(0, len(devices)):
                devices[i].barr = my_barrier

            # Initialize 100 location-specific locks.
            for i in range(0, 100):
                custom_lock = Lock()
                self.loc_lock.append(custom_lock)
            # Propagate the created location locks to all other devices.
            for devs in devices:
                if devs.device_id != 0:
                    devs.loc_lock = self.loc_lock
        else:
            # If not device 0, wait until device 0 has initialized the shared barrier.
            while not hasattr(self, 'barr'):
                continue

    def assign_script(self, script, location):
        """
        Assigns a script to be executed on data at a specific location for this device.
        If `script` is None, it signals that script assignments for the current timepoint are complete.

        Pre-condition: `wait_for_next_timepoint` event must be set, controlling script assignment rate.
        Invariant: If a script is assigned, it's appended to `scripts`, `script_received` is set,
                   and `crt_nb_scripts` is incremented. If `script` is None, `timepoint_done` is set.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (int): The data location where the script should be applied.
        """
        # Wait for the event to allow script assignment for the next timepoint.
        self.wait_for_next_timepoint.wait()
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that new scripts have been received.
            self.crt_nb_scripts += 1 # Increment script count for this timepoint.
        else:
            # Functional Utility: Clear the event to block further script assignments until processing is done.
            self.wait_for_next_timepoint.clear()
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
        Joins the device's dedicated orchestrating thread, effectively waiting for it
        to complete its execution before the program exits.
        """
        self.thread.join()


class MyWorker(Thread):
    """
    A worker thread responsible for executing individual scripts from a device's script queue.

    This thread repeatedly tries to process scripts, acquiring location-specific locks
    to ensure data consistency, executing the script, and updating relevant device data.

    Attributes:
        my_dev (DeviceThread): The `DeviceThread` instance that created this worker,
                               providing access to the parent `Device` and shared resources.
        worker_bar (ReusableBarrierCond): A local barrier for synchronizing `MyWorker`s
                                          within a `DeviceThread`.
        my_id (int): A unique identifier for this worker thread.
        neighbours (list): A cached list of neighboring Device objects.
    """

    def __init__(self, device_thread, worker_barrier, my_id):
        """
        Initializes a new MyWorker instance.

        Args:
            device_thread (DeviceThread): The `DeviceThread` instance that created this worker.
            worker_barrier (ReusableBarrierCond): A local barrier for worker synchronization.
            my_id (int): A unique ID for this worker thread.
        """
        Thread.__init__(self, name="Worker Thread %d" % my_id)
        self.my_dev = device_thread
        self.worker_bar = worker_barrier
        self.my_id = my_id

    def run(self):
        """
        The main execution loop for the MyWorker thread.

        Invariant: The loop continues until the supervisor signals termination (by setting
                   `timepoint_done` and `neighbours` becoming None). Within each iteration,
                   it processes scripts from the device's `scripts` list, handling
                   synchronization with other workers via `worker_bar` and `script_sem`,
                   and ensuring thread-safe data access with `loc_lock`.
        """
        while True:
            # Wait for `timepoint_done` event to be set, signaling scripts are ready.
            self.my_dev.device.timepoint_done.wait()
            neighbours = self.my_dev.neighbours # Get cached neighbors from DeviceThread.
            # Block Logic: If no neighbors are returned (e.g., simulation termination signal), break the loop.
            if neighbours is None:
                break

            # Block Logic: Handle state transition for timepoint completion and cleanup.
            if self.my_dev.inner_state == 1:
                self.worker_bar.wait() # Wait for all workers to reach this point.
                # Only thread 0 performs cleanup actions.
                if self.my_id == 0:
                    self.my_dev.device.timepoint_done.clear() # Clear the event.
                    self.my_dev.device.done_processing.release() # Signal that processing is done.
                    self.my_dev.inner_state = 0 # Reset inner state.
                self.worker_bar.wait() # Second wait for barrier.
                continue

            while True:
                # Acquire semaphore to get access to shared script counter.
                self.my_dev.device.script_sem.acquire()
                # Block Logic: Check if all scripts for the current timepoint have been assigned.
                if self.my_dev.device.crt_script == self.my_dev.device.crt_nb_scripts:
                    self.my_dev.device.script_sem.release()
                    # If all scripts processed by this worker, and this is worker 0,
                    # set inner_state to signal completion of timepoint.
                    if self.my_id == 0:
                        self.my_dev.inner_state = 1
                    break

                # Get the next script to process.
                my_script = self.my_dev.device.scripts[self.my_dev.device.crt_script]
                self.my_dev.device.crt_script += 1 # Increment shared script counter.
                self.my_dev.device.script_sem.release() # Release semaphore.

                # Acquire a location-specific lock to ensure exclusive access for data at this location.
                self.my_dev.device.loc_lock[my_script[1]].acquire()

                script_data = []

                # Block Logic: Collect data from neighboring devices for the current location.
                for device in neighbours:
                    data = device.get_data(my_script[1])
                    if data is not None:
                        script_data.append(data)
                # Collect data from the current device for the current location.
                data = self.my_dev.device.get_data(my_script[1])
                if data is not None:
                    script_data.append(data)

                # Block Logic: If data is available, execute the script and update results.
                if script_data != []:
                    # Execute the script with the collected data.
                    result = my_script[0].run(script_data)

                    # Update the sensor data in neighboring devices.
                    for device in neighbours:
                        device.set_data(my_script[1], result)

                    # Update the sensor data in the current device.
                    self.my_dev.device.set_data(my_script[1], result)

                # Release the location-specific lock.
                self.my_dev.device.loc_lock[my_script[1]].release()


class DeviceThread(Thread):
    """
    The dedicated master orchestrating thread for a `Device` object.

    This thread is responsible for:
    1. Interacting with the supervisor to get neighbor information.
    2. Managing a pool of `MyWorker` worker threads.
    3. Synchronizing with other `DeviceThread` instances using shared barriers.
    4. Resetting events and counters for the next timepoint.

    Attributes:
        device (Device): The Device object associated with this thread.
        inner_state (int): Internal state flag for managing worker synchronization.
        thread_p (list): A list of `MyWorker` worker thread instances.
        w_bar (ReusableBarrierCond): A local barrier for synchronizing `MyWorker`s.
        neighbours (list): A cached list of neighboring Device objects.
        inner_state_lock (threading.Lock): Lock to protect `inner_state` (currently unused).
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        # Initialize the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        self.inner_state = 0 # Internal state for workers.
        self.thread_p = [] # List of worker threads.
        self.w_bar = ReusableBarrierCond(8) # Local barrier for workers.
        self.neighbours = [] # Cached neighbors.
        self.inner_state_lock = Lock() # Lock for inner state (currently unused).

        # Create and start 8 MyWorker threads.
        for i in range(0, 8):
            self.thread_p.append(MyWorker(self, self.w_bar, i))
        for i in range(0, 8):
            self.thread_p[i].start()

    def run(self):
        """
        The main execution loop for the DeviceThread.

        Invariant: The loop continues until the supervisor signals termination.
                   Within each iteration, it retrieves neighbor information,
                   waits for workers to complete processing for the current timepoint,
                   resets counters and events for the next timepoint, and then
                   synchronizes with other `DeviceThread` instances via a global barrier.
        """
        while True:
            # Retrieve information about neighboring devices from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: If no neighbors are returned (e.g., simulation termination signal),
            # set `timepoint_done` to break workers' wait and then break the loop.
            if self.neighbours is None:
                self.device.timepoint_done.set() # Signal workers to check for termination.
                break

            # Wait on `done_processing` semaphore, released by a worker thread (worker 0)
            # after all scripts for the current timepoint have been processed.
            self.device.done_processing.acquire()

            # Reset script counters and allow new script assignments for the next timepoint.
            self.device.crt_script = 0
            self.device.crt_nb_scripts = 0 # Functional utility: Reset number of scripts.
            self.device.scripts = [] # Functional utility: Clear the processed scripts list.
            self.device.wait_for_next_timepoint.set() # Allow `assign_script` to proceed.

            # Synchronize with other `DeviceThread` instances using the global barrier.
            self.device.barr.wait()

        # Block Logic: After the main simulation loop ends, wait for all worker threads to join.
        for i in range(0, 8):
            self.thread_p[i].join()
