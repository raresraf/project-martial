"""
This module implements a distributed device simulation framework, focusing on concurrent script
execution and robust synchronization using a reusable barrier.

Algorithm:
- ReusableBarrier: A classic barrier synchronization mechanism using semaphores and locks
  to ensure all participating threads reach a certain point before any can proceed. It operates
  in two phases to allow repeated reuse.
- Device: Represents a simulated physical device with sensor data and script processing capabilities.
- MyThread: A worker thread responsible for executing a single script on collected data.
- DeviceThread: The orchestrating thread for each `Device`, which distributes scripts to `MyThread`
  instances and manages overall synchronization using shared locks and barriers.
- Concurrent Script Execution: Scripts are executed in parallel by `MyThread` instances, with
  a controlled step-wise execution to manage thread overhead.
"""

from threading import Event, Thread, Lock, Semaphore
from multiprocessing import cpu_count


class ReusableBarrier():
    """
    Implements a reusable barrier synchronization mechanism.

    This barrier ensures that a specified number of threads (`num_threads`)
    all reach a synchronization point before any of them are allowed to proceed.
    It uses a double-phased approach (using `count_threads1` and `count_threads2`
    along with `threads_sem1` and `threads_sem2`) to allow the barrier to be
    reused multiple times without requiring reinitialization.

    Attributes:
        num_threads (int): The total number of threads expected to participate in the barrier.
        count_threads1 (list): A list containing the current count of threads waiting in phase 1.
                               (Wrapped in a list to allow modification within methods).
        count_threads2 (list): A list containing the current count of threads waiting in phase 2.
                               (Wrapped in a list to allow modification within methods).
        count_lock (threading.Lock): A lock to protect access to the `count_threads` variables.
        threads_sem1 (threading.Semaphore): A semaphore used to release threads from phase 1.
        threads_sem2 (threading.Semaphore): A semaphore used to release threads from phase 2.
    """
    def __init__(self, num_threads):
        """
        Initializes a new instance of the ReusableBarrier.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can pass.
        """
        self.num_threads = num_threads
        # Initialize thread counts for two phases, allowing barrier reuse.
        # Wrapped in lists to enable in-place modification within the methods.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # Mutex to protect access to the thread count.
        self.count_lock = Lock()
        # Semaphores to block and release threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait until all `num_threads` threads
        have reached this point. This method orchestrates the two-phase
        synchronization.
        """
        # First phase of synchronization.
        self.phase(self.count_threads1, self.threads_sem1)
        # Second phase of synchronization for barrier reuse.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Manages a single phase of the barrier synchronization.

        Pre-condition: `count_threads` holds the number of remaining threads for this phase.
        Invariant: `count_threads` is decremented atomically. When it reaches zero,
                   all threads are released, and `count_threads` is reset.

        Args:
            count_threads (list): The shared counter for the current phase.
            threads_sem (threading.Semaphore): The semaphore associated with the current phase.
        """
        # Acquire a lock to safely decrement the thread count.
        with self.count_lock:
            count_threads[0] -= 1
            # If this thread is the last to reach the barrier in this phase:
            if count_threads[0] == 0:
                # Release all waiting threads for this phase.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the thread count for the next use of this phase.
                count_threads[0] = self.num_threads
        threads_sem.acquire() # Wait until released by the last thread.


class Device(object):
    """
    Represents a simulated device in a distributed environment.

    Each device has a unique ID, manages its sensor data, and can receive
    and execute scripts. It uses shared locks and reusable barriers for
    thread-safe operations and synchronization with other devices.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings,
                            where keys represent locations.
        supervisor (Supervisor): A reference to the central supervisor managing devices.
        scripts (list): A list to store assigned scripts (script, location) tuples.
        lock1 (threading.Lock): A general lock for protecting shared resources
                                (e.g., supervisor interactions).
        lock2 (threading.Lock): A general lock for protecting shared resources
                                (e.g., data updates).
        script_received (ReusableBarrier): A shared barrier to signal when
                                           scripts have been assigned.
        timepoint_done (ReusableBarrier): A shared barrier to signal completion
                                          of a timepoint's tasks.
        thread (DeviceThread): The dedicated orchestrating thread for this device.
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

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def set_lock(self, lock1, lock2, barrier1, barrier2):
        """
        Sets the shared locks and barriers for this device.

        Pre-condition: This method is called by `setup_devices` to propagate
                       initialized synchronization primitives.
        Invariant: The device's `lock1`, `lock2`, `script_received`, and
                   `timepoint_done` attributes are assigned the provided shared objects.

        Args:
            lock1 (threading.Lock): The first shared lock.
            lock2 (threading.Lock): The second shared lock.
            barrier1 (ReusableBarrier): The shared barrier for script reception.
            barrier2 (ReusableBarrier): The shared barrier for timepoint completion.
        """
        self.lock1 = lock1
        self.lock2 = lock2
        self.script_received = barrier1
        self.timepoint_done = barrier2

    def setup_devices(self, devices):
        """
        Initializes shared synchronization primitives (locks and barriers)
        across all devices. This method is typically called once by the
        first device (`device_id == devices[0].device_id`) to create these
        primitives, which are then shared with all other devices.

        Pre-condition: This method should be called by a central entity (e.g., supervisor)
                       after all devices are instantiated.
        Invariant: If the current device is the designated initializer, new locks and
                   reusable barriers are created and then propagated to all other devices.
                   Each device then starts its `DeviceThread`.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Only the device with the lowest ID (first in the list)
        # initializes shared synchronization primitives.
        if self.device_id == devices[0].device_id:
            lock1 = Lock()
            lock2 = Lock()
            barrier1 = ReusableBarrier(len(devices))
            barrier2 = ReusableBarrier(len(devices))
            # Propagate the newly created locks and barriers to all devices.
            for dev in devices:
                dev.set_lock(lock1, lock2, barrier1, barrier2)

        # Start the dedicated thread for this device's operations.
        self.thread = DeviceThread(self)
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed on data at a specific location for this device.

        Args:
            script (Script or None): The script object to execute.
            location (int): The data location where the script should be applied.
        """
        if script is not None:
            self.scripts.append((script, location))

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
        Joins the device's dedicated thread, effectively waiting for it to complete
        its execution before the program exits.
        """
        self.thread.join()


class MyThread(Thread):
    """
    A worker thread responsible for executing a single script on collected data
    and updating device states.

    Attributes:
        script (Script): The script object to execute.
        script_data (list): The collected data on which the script will operate.
        result (any): The result of the script execution.
        device (Device): The Device object associated with this worker thread.
        neighbours (list): A list of neighboring Device objects.
        location (int): The data location relevant to this script execution.
    """
    def __init__(self, script, script_date, device, neighbours, location):
        """
        Initializes a new MyThread instance.

        Args:
            script (Script): The script to be executed.
            script_date (list): The data for the script to process.
            device (Device): The Device associated with this worker.
            neighbours (list): List of neighboring Devices.
            location (int): The data location being processed.
        """
        Thread.__init__(self)
        self.script = script
        self.script_data = script_date
        self.result = None # Stores the result of script execution.
        self.device = device
        self.neighbours = neighbours
        self.location = location

    def run(self):
        """
        The main execution method for the MyThread.

        Invariant: Executes the assigned script on the provided data, then
                   acquires `device.lock2` to safely update the sensor data
                   in both the current device and its neighbors, finally
                   releasing the lock.
        """
        result = self.script.run(self.script_data)
        self.result = result

        # Acquire lock to ensure thread-safe updates to sensor data in devices.
        self.device.lock2.acquire()
        # Update sensor data in neighboring devices.
        for device in self.neighbours:
            device.set_data(self.location, result)
        # Update sensor data in the current device.
        self.device.set_data(self.location, result)
        self.device.lock2.release() # Release the lock.

class DeviceThread(Thread):
    """
    The dedicated orchestrating thread for a `Device` object.

    This thread is responsible for:
    1. Interacting with the supervisor to get neighbor information.
    2. Waiting for script assignments.
    3. Spawning `MyThread` instances to execute scripts concurrently.
    4. Managing the execution of `MyThread` instances in controlled steps.
    5. Synchronizing with other `DeviceThread` instances using shared barriers.

    Attributes:
        device (Device): The Device object associated with this thread.
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

    def run(self):
        """
        The main execution loop for the DeviceThread.

        Invariant: The loop continues until the supervisor signals termination.
                   Within each iteration, it retrieves neighbor information,
                   waits for new script assignments, creates and manages a pool
                   of `MyThread` workers to execute these scripts, and then
                   synchronizes with other `DeviceThread` instances via a barrier.
        """
        while True:
            # Acquire lock to ensure thread-safe access to supervisor interaction.
            self.device.lock1.acquire()
            # Retrieve information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            self.device.lock1.release()

            # Block Logic: If no neighbors are returned (e.g., simulation termination signal), break the loop.
            if neighbours is None:
                break

            # Wait on the `script_received` barrier, signaling that scripts have been assigned.
            self.device.script_received.wait()

            threads = [] # List to hold MyThread instances.

            # Block Logic: Prepare MyThread instances for each assigned script.
            for (script, location) in self.device.scripts:
                script_data = []

                # Block Logic: Collect data from neighboring devices for the current location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Collect data from the current device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: If data is available, create a MyThread for script execution.
                if script_data != []:
                    threads.append(MyThread(script, script_data, self.device, neighbours, location))

            # Clear the list of scripts after creating worker threads for them.
            self.device.scripts = []

            # Functional Utility: Determine the step size for concurrent thread execution.
            # This allows running a controlled number of MyThreads simultaneously.
            step = cpu_count() * 2
            # Block Logic: Execute MyThreads in batches to manage concurrency.
            for i in range(0, len(threads), step):
                # Start a batch of threads.
                for j in range(step):
                    if i + j < len(threads):
                        threads[i + j].start()
                # Wait for the current batch of threads to complete.
                for j in range(step):
                    if i + j < len(threads):
                        threads[i + j].join()

            # Wait on the `timepoint_done` barrier, synchronizing all DeviceThreads
            # before proceeding to the next timepoint.
            self.device.timepoint_done.wait()
