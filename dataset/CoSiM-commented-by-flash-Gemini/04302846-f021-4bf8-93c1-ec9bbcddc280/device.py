


"""
This module implements a simulation framework for distributed devices, including synchronization mechanisms, device representation, thread management for device execution, and worker processes for script execution on sensor data.

Algorithm:
- ReusableBarrier: A classic barrier synchronization mechanism using semaphores and locks to ensure all participating threads reach a certain point before any can proceed. It operates in two phases to allow reuse.
- DeviceThread/Worker: Implements a producer-consumer-like pattern where DeviceThread orchestrates the distribution of scripts to Worker threads, which then process sensor data.
"""

from threading import Thread, Semaphore, Lock, Event


class ReusableBarrier(object):
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
            # Decrement the count of threads waiting for this phase.
            count_threads[0] -= 1
            # If this thread is the last to reach the barrier in this phase:
            if count_threads[0] == 0:
                # Release all waiting threads for this phase.
                nr_threads = self.num_threads
                while nr_threads > 0:
                    threads_sem.release()
                    nr_threads -= 1
                # Reset the thread count for the next use of this phase.
                count_threads[0] = self.num_threads
        # Wait until released by the last thread.
        threads_sem.acquire()


class Device(object):
    """
    Represents a simulated device in a distributed environment.

    Each device has a unique ID, sensor data, and communicates with a supervisor.
    It can receive and execute scripts, and it participates in synchronized
    operations with other devices using a shared barrier and locks.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing sensor readings,
                            where keys represent locations.
        supervisor (Supervisor): A reference to the central supervisor managing devices.
        script_received (threading.Event): Event to signal when a script has been assigned.
        scripts (list): A list to store assigned scripts (script, location) tuples.
        timepoint_done (threading.Event): Event to signal completion of a timepoint's tasks.
        thread (DeviceThread): The dedicated thread for this device's operations.
        locks (list): A list of threading.Lock objects, one for each data location,
                      to ensure exclusive access during data manipulation.
        barrier (ReusableBarrier): A shared barrier for synchronizing all devices.
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
        # Event to signal the receipt of new scripts.
        self.script_received = Event()
        # List to hold (script, location) tuples assigned to this device.
        self.scripts = []
        # Event to signal the completion of processing for a given timepoint.
        self.timepoint_done = Event()
        # Create and start a dedicated thread for this device's operations.
        self.thread = DeviceThread(self)
        self.thread.start()

        # These will be set by the supervisor or a coordinating function.
        self.locks = None
        self.barrier = None

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Static-like method to initialize shared synchronization primitives
        (barrier and locks) across all devices. This method ensures that
        these resources are only initialized once.

        Pre-condition: Called by the supervisor or a central entity.
        Invariant: If `barrier` or `locks` are not yet initialized for this device,
                   a new `ReusableBarrier` and a list of `Lock` objects
                   (one for each distinct data location) are created and
                   assigned to all participating devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Check if synchronization primitives are already initialized for this device.
        if all(element is None for element in [self.barrier, self.locks]):
            # Initialize a reusable barrier for all devices.
            barrier = ReusableBarrier(len(devices))
            # Initialize a list of locks for each potential data location.
            locks = []

            # Determine the maximum location index to create an appropriate number of locks.
            max_locations = 0
            for device in devices:
                for location in device.sensor_data.keys():
                    if location > max_locations:
                        max_locations = location

            # Create a lock for each data location, up to max_locations.
            for location in range(max_locations + 1):
                locks.append(Lock())

            # Assign the newly created barrier and locks to all devices.
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed on data at a specific location for this device.
        If `script` is None, it signals that no more scripts are coming for the current timepoint,
        and triggers the `timepoint_done` event.

        Args:
            script (Script or None): The script object to execute, or None to signal completion.
            location (int): The data location where the script should be applied.
        """
        if script is not None:
            # Add the script and its target location to the device's script queue.
            self.scripts.append((script, location))
        else:
            # If script is None, it indicates the end of script assignment for a timepoint.
            self.script_received.set()
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
        Joins the device's dedicated thread, effectively waiting for it to complete
        its execution before the program exits.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the lifecycle and script execution for a single Device.

    This thread continuously checks for new scripts, distributes them to worker threads,
    and synchronizes with other device threads using a barrier. It acts as the main
    processing loop for a device in the simulation.

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

        Invariant: The loop continues until the supervisor signals termination
                   by returning None for neighbours.
                   Within each iteration, the thread waits for a timepoint to be done,
                   processes assigned scripts using worker threads, and then
                   synchronizes with other devices via a barrier before clearing
                   its timepoint completion signal.
        """
        while True:
            # Retrieve information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If no neighbors are returned (e.g., simulation termination signal), break the loop.
            if neighbours is None:
                break

            # Wait for the current timepoint's script assignments to be complete.
            self.device.timepoint_done.wait()

            number_of_scripts = len(self.device.scripts)

            # Block Logic: If there are scripts to process, prepare worker threads.
            if number_of_scripts != 0:

                # Determine the number of worker threads to use, capping at 8.
                if number_of_scripts < 8:
                    number_of_threads = number_of_scripts
                else:
                    number_of_threads = 8

                workers_list = []

                # Block Logic: Instantiate and prepare worker threads.
                # Each worker is initialized with a reference to the device and its neighbors.
                for i in range(number_of_threads):
                    worker = Worker(self.device, neighbours)
                    workers_list.append(worker)

                current_thread = 0
                average_scripts = 0

                # Calculate average scripts per worker and distribute scripts.
                if number_of_threads > 0:
                    average_scripts = len(self.device.scripts) / number_of_threads
                aux_average = average_scripts

                # Block Logic: Distribute the assigned scripts among the worker threads.
                # This ensures a balanced workload for parallel processing.
                for (script, location) in self.device.scripts:
                    if aux_average > 0:
                        workers_list[current_thread].scripts.append((script, location))
                        aux_average -= 1
                    # If current worker has enough scripts, move to the next worker.
                    if aux_average == 0:
                        aux_average = average_scripts

                        # Cycle through worker threads for script assignment.
                        if current_thread < number_of_threads - 1:
                            current_thread += 1
                        else:
                            current_thread = 0

                # Block Logic: Start all worker threads for parallel script execution.
                for i in range(number_of_threads):
                    workers_list[i].start()

                # Block Logic: Wait for all worker threads to complete their assigned scripts.
                for i in range(number_of_threads):
                    workers_list[i].join()

            # Clear the timepoint_done event, signaling readiness for the next timepoint.
            self.device.timepoint_done.clear()
            # Synchronize with other devices using the shared barrier.
            self.device.barrier.wait()


class Worker(Thread):
    """
    A worker thread responsible for executing assigned scripts on sensor data
    for a specific device and its neighbors.

    Attributes:
        device (Device): The Device object for which this worker is processing scripts.
        scripts (list): A list of (script, location) tuples assigned to this worker.
        neighbours (list): A list of neighboring Device objects from which to fetch data.
    """

    def __init__(self, device, neighbours):
        """
        Initializes a new Worker instance.

        Args:
            device (Device): The parent device associated with this worker.
            neighbours (list): A list of neighboring devices to interact with.
        """
        # Initialize the base Thread class.
        Thread.__init__(self)
        self.device = device
        # Initialize an empty list to store scripts assigned to this worker.
        self.scripts = []

        self.neighbours = neighbours

    def run(self):
        """
        The main execution method for the Worker thread.

        Invariant: Iterates through each assigned script, acquiring a lock for the
                   data location, collecting data from the device and its neighbors,
                   executing the script, updating data, and finally releasing the lock.
        """
        # Block Logic: Iterate through each script assigned to this worker.
        for (script, location) in self.scripts:
            script_data = []

            # Acquire a lock for the specific data location to ensure exclusive access.
            self.device.locks[location].acquire()

            # Block Logic: Collect data from neighboring devices for the current location.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Collect data from the current device for the current location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: If there is data collected, execute the script and update results.
            if script_data != []:
                # Execute the script with the collected data.
                result = script.run(script_data)
                # Update the sensor data in neighboring devices.
                for device in self.neighbours:
                    device.set_data(location, result)
                # Update the sensor data in the current device.
                self.device.set_data(location, result)

            # Release the lock for the specific data location.
            self.device.locks[location].release()
