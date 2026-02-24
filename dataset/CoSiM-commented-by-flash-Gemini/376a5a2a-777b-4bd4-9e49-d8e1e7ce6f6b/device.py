


from Queue import Queue
from threading import Semaphore, Lock
from threading import Event, Thread


class Device(object):
    """
    Represents a simulated device within a multi-device system.

    Each device manages its own sensor data, processes assigned scripts via
    a queue and worker threads, and synchronizes with other devices using
    a reusable barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor readings,
                                 where keys are locations and values are data.
            supervisor (Supervisor): A reference to the supervisor object
                                     that manages device interactions.
        """
        self.device_id = device_id
        self.read_data = sensor_data
        self.supervisor = supervisor
        # A queue to hold scripts that are ready to be processed by worker threads.
        self.active_queue = Queue()
        # A list to store scripts assigned to this device for the current timepoint.
        self.scripts = []
        # The thread dedicated to managing this device's worker threads.
        self.thread = DeviceThread(self)
        # A counter for tracking simulation time (or timepoints).
        self.time = 0

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources, specifically the `ReusableBarrierSem`, for all devices.
        Only device with device_id 0 initializes the barrier and shares it.

        Args:
            devices (list): A list of all Device instances in the system.
        """
        # Device with device_id 0 initializes the shared barrier.
        if self.device_id == 0:
            # Create a new reusable barrier with the total number of devices.
            self.new_round = ReusableBarrierSem(len(devices))
            # Store a reference to all devices.
            self.devices = devices
            # Share the initialized barrier with all other devices.
            for device in self.devices:
                device.new_round = self.new_round
        # Start the main thread for this device.
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be processed. If `script` is not None, it's added
        to a temporary list. If `script` is None, it signals that all scripts
        for the current timepoint have been assigned and moves them to the
        `active_queue` for processing by worker threads.

        Args:
            script (object or None): The script object to assign, or None to signal
                                     the end of scripts for a timepoint.
            location (int): The location associated with the script.
        """
        if script is not None:
            # If a script is provided, add it to the temporary scripts list.
            self.scripts.append((script, location))
        else:
            # If script is None, it means all scripts for this timepoint have been collected.
            # Move all collected scripts into the active_queue for worker threads to pick up.
            for (script, location) in self.scripts:
                self.active_queue.put((script, location))
            # Add sentinel values to the queue to signal worker threads to terminate
            # after processing all scripts for the current timepoint.
            for x in range(8): # Assuming 8 worker threads, though this should ideally be dynamic.
                self.active_queue.put((-1, -1))

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The integer identifier for the location for which to retrieve data.

        Returns:
            Any: The sensor data if the location exists in read_data, otherwise None.
        """
        return self.read_data[location] if location in self.read_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Args:
            location (int): The integer identifier for the location where the data should be set.
            data (Any): The new data to set for the location.
        """
        if location in self.read_data:
            self.read_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its associated thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the lifecycle of WorkerThread instances for a Device.

    This thread is responsible for continuously fetching neighbor information,
    creating and starting a pool of `WorkerThread`s to process scripts
    from the device's `active_queue`, waiting for them to complete,
    and synchronizing with other devices using the `ReusableBarrierSem`.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Defines the fixed number of worker threads to be created.
        self.workers_number = 8

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously performs the following steps:
        1. Retrieves up-to-date neighbor information from the supervisor.
        2. If no neighbors are returned (e.g., simulation end), the loop breaks.
        3. Creates and starts a fixed number of `WorkerThread` instances.
        4. Waits for all `WorkerThread` instances to complete their tasks.
        5. Synchronizes with other DeviceThreads using the `ReusableBarrierSem`.
        """
        # Initial fetch of neighbors for the first round.
        neighbours = self.device.supervisor.get_neighbours()
        while True:
            # List to hold references to the active worker threads for the current round.
            self.workers = []
            # Update the device's view of its neighbors for the current round.
            self.device.neighbours = neighbours
            # If no neighbors are available, it implies the simulation should terminate.
            if neighbours is None:
                break

            # Create and start a pool of WorkerThread instances.
            for i in range(self.workers_number):
                new_worker = WorkerThread(self.device)
                self.workers.append(new_worker)
                new_worker.start()

            # Wait for all worker threads to complete their execution.
            for worker in self.workers:
                worker.join()
            # Synchronize with other DeviceThreads at the end of the round.
            self.device.new_round.wait()
            # Fetch neighbors again for the next round.
            neighbours = self.device.supervisor.get_neighbours()


class WorkerThread(Thread):
    """
    A worker thread responsible for processing individual scripts from a device's active queue.

    Each worker thread continuously retrieves a script and its associated location,
    collects relevant sensor data from the device and its neighbors, executes the
    script, and updates the sensor data based on the script's outcome.
    """

    def __init__(self, device):
        """
        Initializes a new WorkerThread instance.

        Args:
            device (Device): The Device object associated with this worker thread.
        """
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for WorkerThread.

        It continuously performs the following steps:
        1. Retrieves a script and its location from the `device.active_queue`.
        2. If a sentinel value (-1, -1) is retrieved, it breaks the loop (terminates).
        3. Collects sensor data from the device and its current neighbors for the given location.
        4. If sufficient data is collected, it executes the script.
        5. Updates the sensor data on the device and its neighbors based on the script's result,
           specifically if the new result is greater than the old value (e.g., for maximum aggregation).
        """
        while True:
            # Get a script and its location from the device's active queue.
            script, location = self.device.active_queue.get()
            # Check for a sentinel value to terminate the worker thread.
            if script == -1:
                break
            
            script_data = [] # List to store data collected for the script.
            matches = []     # List to store devices from which data was collected.

            # Collect data from neighboring devices for the current location.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    matches.append(device)
                    script_data.append(data)
            
            # Collect data from the current device itself for the current location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
                matches.append(self.device)

            # If there's enough data (more than one piece, indicating participation from at least two sources),
            # execute the script and update data.
            if len(script_data) > 1:
                # Execute the script with the collected data.
                result = script.run(script_data)
                # Update the data in all matching devices (neighbors and self).
                for device in matches:
                    old_value = device.get_data(location)
                    # Only update if the new result is greater than the existing value.
                    if old_value < result:
                        device.set_data(location, result)


class ReusableBarrierSem():
    """
    A reusable barrier synchronization primitive that uses semaphores to coordinate multiple threads.
    It ensures that a fixed number of threads all reach a certain point before any can proceed,
    and can then be reset for subsequent synchronization points. This implementation uses a
    two-phase approach to allow for barrier reuse.
    """
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrierSem with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that must reach the
                                barrier before any can proceed.
        """
        self.num_threads = num_threads
        # Counter for the first phase of the barrier.
        self.count_threads1 = self.num_threads
        # Counter for the second phase of the barrier.
        self.count_threads2 = self.num_threads
        # A lock to protect access to the thread counters.
        self.counter_lock = Lock()
        # Semaphore for releasing threads in the first phase.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for releasing threads in the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait until all other threads have also
        called this method. This method orchestrates the two phases of the barrier.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        Manages the first phase of the barrier synchronization.
        Threads decrement a shared counter and the last thread releases all others
        through a semaphore.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # If this is the last thread, release all waiting threads.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset the counter for the next use of phase1.
                self.count_threads1 = self.num_threads

        # Acquire the semaphore, effectively waiting until all threads are released in this phase.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        Manages the second phase of the barrier synchronization.
        Similar to phase1, but uses a separate counter and semaphore for reuse.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # If this is the last thread, release all waiting threads.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset the counter for the next use of phase2.
                self.count_threads2 = self.num_threads

        # Acquire the semaphore, effectively waiting until all threads are released in this phase.
        self.threads_sem2.acquire()
