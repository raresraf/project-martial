

# @file 03166f8e-0aca-47b3-820e-31894c88dc74/device.py
# @brief Implements simulated device functionality with multithreaded script execution and synchronization.
#
# This module defines classes for a simulated 'Device' that processes sensor data,
# interacts with neighboring devices, and executes scripts using a pool of worker threads.
# It includes mechanisms for inter-thread communication, synchronization using a reusable barrier,
# and managing the device's operational lifecycle within a larger simulation.

from Queue import Queue
from threading import Semaphore, Lock
from threading import Event, Thread


class Device(object):
    """
    @brief Represents a simulated device with sensor data, script processing capabilities,
           and interactions within a network of devices.
    
    Manages its own sensor data, accepts scripts for execution, and orchestrates
    worker threads to process these scripts. It also participates in synchronization
    with other devices using a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device object.

        @param device_id: A unique integer identifier for this device.
        @param sensor_data: A dictionary containing the initial sensor data for this device.
                            Keys represent data locations, values are sensor readings.
        @param supervisor: A reference to the supervisor object responsible for managing the network of devices.
        @post The device is initialized with its ID, data, and a thread for execution.
        @post An active queue for scripts and an empty list for pending scripts are created.
        """
        # Functional Utility: Unique identifier for the device.
        self.device_id = device_id
        # Functional Utility: Stores the device's sensor data.
        self.read_data = sensor_data
        # Functional Utility: Reference to the simulation supervisor for network interactions.
        self.supervisor = supervisor
        # Functional Utility: Queue for active scripts to be processed by worker threads.
        self.active_queue = Queue()
        # Functional Utility: List to hold scripts assigned to the device before they are added to the active queue.
        self.scripts = []
        # Functional Utility: The dedicated thread for this device's operational logic.
        self.thread = DeviceThread(self)
        # Functional Utility: Represents the current simulation time-step for the device.
        self.time = 0

    def __str__(self):
        """
        @brief Provides a string representation of the Device.

        @return: A formatted string indicating the device's ID.
        """
        # Functional Utility: Returns a human-readable string representation of the device.
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the synchronization barrier for a group of devices.

        @param devices: A list of all device objects in the simulation.
        @pre This method is typically called only by device with ID 0 to initialize the barrier.
        @post A ReusableBarrierSem is created (if device_id is 0) and assigned to all devices.
        @post The device's dedicated thread is started.
        """
        # Block Logic: Ensures that only the device with ID 0 initializes the shared barrier.
        if self.device_id == 0:
            # Functional Utility: Creates a reusable semaphore-based barrier for all devices.
            self.new_round = ReusableBarrierSem(len(devices))
            # Functional Utility: Stores a reference to all devices in the simulation.
            self.devices = devices
            # Block Logic: Assigns the newly created barrier to all devices in the simulation.
            # Invariant: Each device in 'self.devices' receives the same barrier instance.
            for device in self.devices:
                device.new_round = self.new_round
        # Functional Utility: Starts the device's dedicated thread, initiating its operational cycle.
        self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for processing or signals the start of script execution.

        @param script: The script object to execute, or None to signal that pending scripts should be activated.
        @param location: The data location relevant to the script.
        @post If 'script' is not None, it is added to the pending scripts.
        @post If 'script' is None, all pending scripts are moved to the active queue,
              and termination signals are added for worker threads.
        """
        # Block Logic: Differentiates between receiving a new script and initiating script processing for the current time step.
        if script is not None:
            # Functional Utility: Appends the received script and its location to a list of pending scripts.
            self.scripts.append((script, location))
        else:
            # Block Logic: Transfers all pending scripts to the active queue for worker processing.
            # Invariant: All scripts in 'self.scripts' are moved to 'self.active_queue'.
            for (script, location) in self.scripts:
                self.active_queue.put((script, location))
            # Block Logic: Adds termination signals to the active queue for worker threads.
            # Inline: '8' represents the fixed number of worker threads, ensuring each receives a termination signal.
            for x in range(8):
                self.active_queue.put((-1, -1))

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The data location for which to retrieve the sensor reading.
        @return: The sensor data for the specified location, or None if the location is not found.
        """
        # Functional Utility: Safely retrieves data associated with a given location from the device's sensor readings.
        return self.read_data[location] if location in self.read_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data for a specific location.

        @param location: The data location to update.
        @param data: The new sensor data value.
        @post The sensor data at the specified location is updated if the location exists.
        """
        # Block Logic: Updates the sensor data only if the specified location is valid and exists in the device's readings.
        if location in self.read_data:
            self.read_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's dedicated thread.

        @post The device's thread is joined, ensuring its completion.
        """
        # Functional Utility: Waits for the device's operational thread to complete its tasks and terminate.
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The dedicated thread for a simulated device, responsible for its operational lifecycle.
    
    This thread manages the device's interaction with the supervisor, orchestrates
    worker threads to process scripts, and synchronizes its operations with other
    devices using a reusable barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.

        @param device: The Device object that this thread is responsible for.
        @post The thread is initialized with a descriptive name and associated with the device.
        @post The number of worker threads is set.
        """
        # Functional Utility: Calls the base Thread class constructor, assigning a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        # Functional Utility: Stores a reference to the Device object this thread manages.
        self.device = device
        # Functional Utility: Defines the fixed number of worker threads to be created for script processing.
        self.workers_number = 8

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        @post The thread continuously fetches neighbors, spawns worker threads to process scripts,
              and synchronizes with other devices until a termination signal (None neighbors) is received.
        """
        # Functional Utility: Retrieves the initial set of neighboring devices from the supervisor.
        neighbours = self.device.supervisor.get_neighbours()
        # Block Logic: Main operational loop for the device, continuing until the supervisor signals termination.
        while True:
            # Functional Utility: Initializes an empty list to store worker thread instances for the current round.
            self.workers = []
            # Functional Utility: Updates the device's internal representation of its neighbors.
            self.device.neighbours = neighbours
            # Block Logic: Checks if the supervisor has signaled the end of the simulation by returning None for neighbours.
            if neighbours is None:
                # Functional Utility: Exits the main operational loop upon receiving a termination signal.
                break

            # Block Logic: Creates and starts a fixed number of worker threads for processing scripts.
            # Invariant: Exactly 'self.workers_number' worker threads are created and started.
            for i in range(self.workers_number):
                new_worker = WorkerThread(self.device)
                self.workers.append(new_worker)
                new_worker.start()

            # Block Logic: Waits for all worker threads to complete their current tasks.
            # Invariant: All worker threads must finish their execution before proceeding.
            for worker in self.workers:
                worker.join()
            # Functional Utility: Synchronizes with other DeviceThreads using the shared barrier, waiting for all to complete.
            self.device.new_round.wait()
            # Functional Utility: Fetches the next set of neighbors for the upcoming simulation round.
            neighbours = self.device.supervisor.get_neighbours()


class WorkerThread(Thread):
    """
    @brief A worker thread responsible for executing individual scripts assigned to a device.
    
    Each worker continuously fetches scripts from the device's active queue,
    gathers data from relevant neighbors and the local device, executes the script,
    and updates device data based on the script's output.
    """

    def __init__(self, device):
        """
        @brief Initializes a WorkerThread.

        @param device: The Device object that this worker thread is associated with.
        @post The thread is initialized with a descriptive name and associated with the device.
        """
        # Functional Utility: Calls the base Thread class constructor, assigning a descriptive name.
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        # Functional Utility: Stores a reference to the Device object whose scripts this worker will execute.
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the WorkerThread.

        @post The thread continuously retrieves and executes scripts from the device's active queue
              until a termination signal (-1, -1) is received.
        @post After script execution, relevant device data is updated.
        """
        # Block Logic: Main loop for the worker thread, processing scripts until a termination signal is encountered.
        while True:
            # Functional Utility: Retrieves a script and its associated location from the device's active queue.
            script, location = self.device.active_queue.get()
            # Block Logic: Checks if the retrieved item is a termination signal for the worker thread.
            if script == -1:
                # Functional Utility: Exits the worker thread's execution loop upon receiving a termination signal.
                break
            # Functional Utility: Initializes an empty list to store data relevant to the script.
            script_data = []
            # Functional Utility: Initializes an empty list to keep track of devices that provided data.
            matches = []
            # Block Logic: Gathers data from neighboring devices for the script.
            # Invariant: Data is collected only from neighbors where data exists for the specified location.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    matches.append(device)
                    script_data.append(data)
            # Functional Utility: Gathers data from the local device for the script.
            data = self.device.get_data(location)
            # Block Logic: Appends local device data if available.
            if data is not None:
                script_data.append(data)
                matches.append(self.device)

            # Block Logic: Executes the script if sufficient data (more than one source) is available.
            if len(script_data) > 1:
                # Functional Utility: Executes the script with the collected data and stores the result.
                result = script.run(script_data)
                # Block Logic: Propagates the script result to all participating devices (neighbors and local device).
                # Invariant: Each participating device's data is updated if the new result is greater than its current value.
                for device in matches:
                    old_value = device.get_data(location)
                    if old_value < result:
                        device.set_data(location, result)


class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier using semaphores for synchronizing multiple threads in two phases.
    
    This barrier ensures that a specified number of threads wait for each other at a synchronization
    point before any of them can proceed. It supports multiple uses (reusable) across different
    synchronization cycles.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrierSem.

        @param num_threads: The total number of threads that must reach the barrier to release.
        @pre num_threads > 0.
        @post Internal counters and semaphores are initialized.
        """
        # Functional Utility: Stores the total number of threads expected to reach the barrier.
        self.num_threads = num_threads
        # Functional Utility: Counter for the first phase of the barrier.
        self.count_threads1 = self.num_threads
        # Functional Utility: Counter for the second phase of the barrier.
        self.count_threads2 = self.num_threads
        # Functional Utility: A lock to protect access to the thread counters.
        self.counter_lock = Lock()
        # Functional Utility: Semaphore for controlling the release of threads in phase 1.
        self.threads_sem1 = Semaphore(0)
        # Functional Utility: Semaphore for controlling the release of threads in phase 2.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all expected threads have reached this barrier.

        @post The thread completes both phase1 and phase2 of the synchronization.
        """
        # Functional Utility: Executes the first phase of the barrier synchronization.
        self.phase1()
        # Functional Utility: Executes the second phase of the barrier synchronization.
        self.phase2()

    def phase1(self):
        """
        @brief The first phase of the reusable barrier synchronization.

        @post The calling thread waits until all threads have reached this phase.
        @post The counter for phase 1 is reset.
        """
        # Block Logic: Atomically decrements the counter and checks if all threads have arrived.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Block Logic: If this is the last thread to reach phase 1, release all waiting threads.
            if self.count_threads1 == 0:
                # Functional Utility: Releases all threads waiting on the first semaphore.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Functional Utility: Resets the counter for the next use of phase 1.
                self.count_threads1 = self.num_threads

        # Functional Utility: Acquires the semaphore, blocking until released by the last thread in phase 1.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief The second phase of the reusable barrier synchronization.

        @post The calling thread waits until all threads have reached this phase.
        @post The counter for phase 2 is reset.
        """
        # Block Logic: Atomically decrements the counter and checks if all threads have arrived.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Block Logic: If this is the last thread to reach phase 2, release all waiting threads.
            if self.count_threads2 == 0:
                # Functional Utility: Releases all threads waiting on the second semaphore.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Functional Utility: Resets the counter for the next use of phase 2.
                self.count_threads2 = self.num_threads

        # Functional Utility: Acquires the semaphore, blocking until released by the last thread in phase 2.
        self.threads_sem2.acquire()
