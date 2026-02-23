"""
@376a5a2a-777b-4bd4-9e49-d8e1e7ce6f6b/device.py
@brief Implements a distributed device simulation framework with a focus on concurrent script execution
using a work queue and worker threads. This variant uses a simplified reusable barrier for
synchronization and processes scripts via a `Queue` for better load balancing among worker threads.
* Algorithm: Producer-Consumer pattern (scripts as tasks, worker threads as consumers) within a
             device simulation, synchronized by a reusable barrier.
* Concurrency: Uses `Queue` for inter-thread communication, `threading.Thread` for device and worker
               threads, `threading.Semaphore`, and `threading.Lock` for synchronization.
"""

from Queue import Queue # Python 2.x Queue module for thread-safe queue.
from threading import Semaphore, Lock
from threading import Event, Thread


class Device(object):
    """
    @brief Represents a simulated device, managing sensor data, scripts, and orchestrating
    worker threads to process these scripts.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor readings relevant to this device.
        @param supervisor: A reference to the supervisor object managing the devices.
        """
        self.device_id = device_id
        self.read_data = sensor_data # Stores the device's sensor data.
        self.supervisor = supervisor
        self.active_queue = Queue() # Queue for scripts to be processed by worker threads.
        self.scripts = [] # List to store assigned scripts.
        self.thread = DeviceThread(self) # The main thread for this device.
        self.time = 0 # Not explicitly used in provided methods, possibly for time-stepping.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the synchronization barrier and distributes it among all devices.
        Only device 0 acts as the coordinator for barrier initialization.
        @param devices: A list of all Device objects in the simulation.
        """
        # Pre-condition: This block executes only for device with ID 0 to coordinate setup.
        if self.device_id == 0:
            # Invariant: Device 0 initializes a new reusable barrier for all devices.
            self.new_round = ReusableBarrierSem(len(devices))
            self.devices = devices
            # Distribute the initialized barrier to all devices.
            for device in self.devices:
                device.new_round = self.new_round
        self.thread.start() # Start the device's main thread.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific data location.
        If `script` is None, it signals the end of script assignment for a timepoint,
        and all accumulated scripts are put into the active queue for processing.
        @param script: The script object to be executed.
        @param location: The data location relevant to the script.
        """
        # Pre-condition: A script is provided, or None to signal end of timepoint.
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Invariant: If script is None, transfer all collected scripts to the active queue.
            for (script, location) in self.scripts:
                self.active_queue.put((script, location))
            # Put sentinel values (-1, -1) into the queue to signal worker threads to terminate.
            for x in range(8): # Number of worker threads.
                self.active_queue.put((-1, -1))

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key for the sensor data.
        @return: The sensor data at the specified location, or None if not found.
        """
        return self.read_data[location] if location in self.read_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.
        @param location: The key for the sensor data.
        @param data: The new data to be set.
        """
        # Pre-condition: `location` must exist in `read_data` to be updated.
        if location in self.read_data:
            self.read_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's associated thread.
        """
        self.thread.join() # Wait for the device's main thread to complete.


class DeviceThread(Thread):
    """
    @brief The main thread for a Device, responsible for coordinating the execution of worker threads,
    synchronizing with other devices, and managing timepoints.
    """
    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device: The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers_number = 8 # Number of worker threads to spawn.

    def run(self):
        """
        @brief The main execution loop for the device thread.
        It continuously gets neighbors, spawns worker threads to process scripts,
        and synchronizes with other devices using the barrier.
        """
        neighbours = self.device.supervisor.get_neighbours() # Initial fetch of neighbors.
        # Invariant: Loop continuously until the simulation signals termination.
        while True:
            self.workers = [] # List to hold worker thread instances.
            self.device.neighbours = neighbours # Update device's neighbors.
            # Pre-condition: If `neighbours` is None, it signals the end of the simulation.
            if neighbours is None:
                break # Exit the loop and terminate the thread.

            # Block Logic: Spawn and start worker threads to process scripts from the active queue.
            for i in range(self.workers_number):
                new_worker = WorkerThread(self.device)
                self.workers.append(new_worker)
                new_worker.start()

            # Invariant: Wait for all worker threads to complete their current tasks.
            for worker in self.workers:
                worker.join()
            
            # Synchronize all device threads at the barrier before proceeding to the next timepoint.
            self.device.new_round.wait()
            # Fetch updated neighbors for the next timepoint.
            neighbours = self.device.supervisor.get_neighbours()


class WorkerThread(Thread):
    """
    @brief A worker thread that fetches scripts from the device's active queue,
    executes them, and updates data based on neighbor information.
    """
    def __init__(self, device):
        """
        @brief Initializes a WorkerThread.
        @param device: The parent Device object from which to get tasks and data.
        """
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the worker thread.
        It continuously gets scripts from the active queue, performs data collection,
        executes the script, and applies updates if conditions are met.
        """
        # Invariant: Loop continuously until a sentinel value is received from the queue.
        while True:
            script, location = self.device.active_queue.get() # Fetch a script task from the queue.
            # Pre-condition: If `script` is -1, it's a sentinel value signaling termination.
            if script == -1:
                break # Exit the loop and terminate the thread.
            
            script_data = [] # List to collect data for the script.
            matches = [] # List to store devices that provided data.

            # Block Logic: Collect data from neighboring devices for the current `location`.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    matches.append(device)
                    script_data.append(data)
            
            # Collect data from the current device itself for the current `location`.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
                matches.append(self.device)

            # Pre-condition: Execute the script only if more than one data point was collected.
            if len(script_data) > 1:
                result = script.run(script_data) # Execute the script.
                # Invariant: Update data on matching devices if the new result is greater than the old value.
                for device in matches:
                    old_value = device.get_data(location)
                    if old_value < result:
                        device.set_data(location, result)


class ReusableBarrierSem():
    """
    @brief A simpler reusable barrier implementation using semaphores for a two-phase synchronization.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrierSem.
        @param num_threads: The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for the first phase.
        self.count_threads2 = self.num_threads # Counter for the second phase.
        self.counter_lock = Lock() # Lock to protect the counters.
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase.

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier, completing both phases.
        """
        self.phase1() # Execute the first phase of synchronization.
        self.phase2() # Execute the second phase of synchronization.

    def phase1(self):
        """
        @brief First phase of the barrier: threads decrement a counter and the last one releases all others.
        """
        # Pre-condition: Acquire lock to safely decrement the counter.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Invariant: If this is the last thread, release all waiting threads.
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset counter for next use.

        # Block the current thread until released by the last thread in this phase.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Second phase of the barrier: similar to phase1, allowing for reusability.
        """
        # Pre-condition: Acquire lock to safely decrement the counter.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Invariant: If this is the last thread, release all waiting threads.
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset counter for next use.

        # Block the current thread until released by the last thread in this phase.
        self.threads_sem2.acquire()