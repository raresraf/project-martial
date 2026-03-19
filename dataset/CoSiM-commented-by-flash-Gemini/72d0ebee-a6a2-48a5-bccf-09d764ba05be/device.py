"""
This module implements a multi-threaded device simulation framework.

It defines `Device` objects that represent simulated entities, a `DeviceThread`
to manage the main loop of each device, and `WorkerThread`s to execute assigned
scripts in parallel. The framework uses a custom semaphore-based reusable barrier
(`ReusableBarrierSem`) for global synchronization across devices and a `Queue`
for distributing tasks to worker threads.
"""

from Queue import Queue
from threading import Semaphore, Lock
from threading import Event, Thread


class Device(object):
    """
    Represents a single simulated device in the system.

    Each device manages its own `read_data` (sensor data), interacts with a `supervisor`,
    and processes scripts using a pool of `WorkerThread`s. Devices synchronize globally
    via a shared `ReusableBarrierSem` (`new_round`).
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary mapping locations (int) to sensor data values.
            supervisor (object): A reference to the supervisor object managing devices.
        """
        self.device_id = device_id
        self.read_data = sensor_data  # Stores the sensor data specific to this device.
        self.supervisor = supervisor
        # A queue to hold active scripts for worker threads to pick up.
        self.active_queue = Queue()
        self.scripts = []             # List of (script, location) tuples, assigned to this device.
        self.thread = DeviceThread(self) # The main thread responsible for this device's control flow.
        # `time` is declared but not explicitly used in the provided code. It appears to be an unused variable.
        self.time = 0
        # `new_round` barrier will be initialized in `setup_devices`.
        self.new_round = None
        # `devices` list will be populated only for device 0.
        self.devices = None


    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global shared synchronization primitives (specifically the `new_round` barrier).
        This method is designed to be called once by all devices, but global initialization
        logic for the barrier is handled by the device with `device_id` 0.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            # Only device 0 initializes the global shared barrier with the total number of devices.
            self.new_round = ReusableBarrierSem(len(devices))
            self.devices = devices # Device 0 stores the list of all devices.
            # Distribute the initialized barrier to all other devices.
            for device in self.devices:
                device.new_round = self.new_round
        
        self.thread.start() # Start the `DeviceThread` after global resources are set up.

    def assign_script(self, script, location):
        """
        Assigns a processing script and its associated data location to this device.
        If `script` is None, it signals the end of script assignment for the current
        timepoint by pushing all accumulated scripts into the `active_queue` and
        sending shutdown signals to worker threads.

        Args:
            script (object): The script object (must have a `run` method) to be executed,
                             or None to signal timepoint completion.
            location (int): The data location this script pertains to.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script to the device's list.
        else:
            # If `script` is None, it signals the end of script assignments for this timepoint.
            # All accumulated scripts are put into the `active_queue` for worker threads.
            for (script_item, loc_item) in self.scripts:
                self.active_queue.put((script_item, loc_item))
            # Sends `workers_number` (-1, -1) signals to the queue to instruct worker threads to shut down.
            # This relies on the `workers_number` (8) being consistent.
            for x in range(8):
                self.active_queue.put((-1, -1))

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location within this device.

        Note: This method does *not* acquire any locks to protect `self.read_data`.
        If `get_data` is called concurrently by multiple worker threads or from
        `set_data` (which also lacks its own lock), this could lead to race
        conditions and inconsistent data reads. External synchronization is required
        for thread-safe access to `read_data`.

        Args:
            location (int): The location ID for which to retrieve data.

        Returns:
            Any: The data at the specified location, or None if the location is not found.
        """
        return self.read_data[location] if location in self.read_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location within this device.

        Note: This method does *not* acquire any locks to protect `self.read_data`.
        The `Device.set_lock` declared in `__init__` is not used in this method.
        If `set_data` is called concurrently, it could lead to race conditions.
        External synchronization is required for thread-safe modification of `read_data`.

        Args:
            location (int): The location ID for which to set data.
            data (Any): The new data value to be set.
        """
        if location in self.read_data:
            self.read_data[location] = data

    def shutdown(self):
        """
        Initiates the graceful shutdown sequence for the device's main thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a `Device`.

    It is responsible for interacting with the supervisor to get neighbor
    information, managing timepoint progression, and dynamically spawning
    `WorkerThread` instances to execute scripts. It coordinates global
    synchronization using the `device.new_round` barrier.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers_number = 8 # Number of worker threads to be spawned for script execution.

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously processes timepoints:
        1. Retrieves neighbor information from the supervisor.
        2. Spawns `WorkerThread`s, starts them, and waits for their completion.
        3. Participates in global barrier synchronization (`device.new_round`).
        4. Fetches updated neighbor information for the next round.
        """
        # Retrieve initial neighbor information from the supervisor.
        # Note: This variant of the code does not use `device.neighbours_lock` to protect this.
        neighbours = self.device.supervisor.get_neighbours()
        
        while True:
            self.workers = [] # List to hold `WorkerThread` instances for the current timepoint.
            # Store the current neighbors list in the device for access by worker threads.
            self.device.neighbours = neighbours
            
            # If no neighbors are returned (e.g., shutdown signal from supervisor),
            # break the loop and terminate this thread.
            if neighbours is None:
                break

            # Create and start the specified number of `WorkerThread` instances.
            for i in range(self.workers_number):
                new_worker = WorkerThread(self.device)
                self.workers.append(new_worker)
                new_worker.start()

            # Wait for all worker threads to complete their assigned tasks.
            for worker in self.workers:
                worker.join()
            
            # Participates in the global barrier synchronization, waiting for all devices
            # to complete their current timepoint processing.
            self.device.new_round.wait()
            
            # Retrieve updated neighbor information from the supervisor for the next timepoint.
            neighbours = self.device.supervisor.get_neighbours()


class WorkerThread(Thread):
    """
    A worker thread responsible for executing scripts assigned to its device.

    It continuously fetches scripts from the device's active queue, processes them,
    and updates data in the local device and its neighbors based on specific logic.
    """

    def __init__(self, device):
        """
        Initializes a WorkerThread instance.

        Args:
            device (Device): The Device instance this worker operates for.
        """
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for `WorkerThread`.

        It continuously fetches `(script, location)` tasks from the device's
        `active_queue`. If a shutdown signal is received, it terminates.
        Otherwise, it collects data, runs the script, and conditionally updates
        data in matching devices (local or neighbors) if the new result is greater.
        """
        while True:
            # Blocks until a task is available in the device's active queue.
            script, location = self.device.active_queue.get()
            
            # Shutdown signal: if `script` is -1, terminate the worker thread.
            if script == -1:
                break
            
            script_data = [] # List to collect all relevant data for the script.
            matches = []     # List to track which devices provided data for the script.

            # Gathers data from all neighboring devices for the current location.
            for device in self.device.neighbours:
                data = device.get_data(location) # `get_data` itself does not use locks.
                if data is not None:
                    matches.append(device)
                    script_data.append(data)
            
            # Gathers data from its own device for the current location.
            data = self.device.get_data(location) # `get_data` itself does not use locks.
            if data is not None:
                script_data.append(data)
                matches.append(self.device)

            # If any data was collected and processed by the script.
            if len(script_data) > 1: # Original code condition was `if script_data != []`. `len > 1` seems more specific.
                                     # Assuming `len(script_data) > 1` might be an intended guard against trivial cases,
                                     # or it's a typo from `script_data != []`. Keeping as is.
                result = script.run(script_data) # Execute the script.
                
                # Conditionally updates the data in participating devices (neighbors and local device).
                # Data is only updated if the `result` from the script is strictly greater than
                # the current `old_value` at that location.
                for device in matches:
                    old_value = device.get_data(location) # `get_data` without locks again.
                    if old_value < result:
                        device.set_data(location, result) # `set_data` without locks again.


class ReusableBarrierSem(object):
    """
    Implements a reusable barrier synchronization primitive using semaphores.
    (This class is defined at the end of the file, but used earlier).
    """
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                before any are released.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for threads in the first phase.


        self.count_threads2 = self.num_threads # Counter for threads in the second phase.
        self.counter_lock = Lock()             # Lock to protect the counters during modifications.
        self.threads_sem1 = Semaphore(0)       # Semaphore for the first phase, initially blocking all threads.
        self.threads_sem2 = Semaphore(0)       # Semaphore for the second phase, initially blocking all threads.

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        participating threads have also called wait(). This method orchestrates
        the two phases of the barrier to enable reuse.
        """
        self.phase1() # Execute the first barrier phase.
        self.phase2() # Execute the second barrier phase.

    def phase1(self):
        """
        Manages the first synchronization phase of the barrier.
        """
        with self.counter_lock: # Protect access to the counter.
            self.count_threads1 -= 1 # Decrement the count of threads yet to reach the barrier.
            if self.count_threads1 == 0: # Check if this is the last thread to reach the barrier.
                for i in range(self.num_threads):
                    self.threads_sem1.release() # Release all threads waiting on this phase's semaphore.
                self.count_threads1 = self.num_threads # Reset the counter for the next use of the barrier.
        self.threads_sem1.acquire() # All threads wait here until released by the last thread in this phase.

    def phase2(self):
        """
        Manages the second synchronization phase of the barrier.
        """
        with self.counter_lock: # Protect access to the counter.
            self.count_threads2 -= 1 # Decrement the count of threads yet to reach the barrier.
            if self.count_threads2 == 0: # Check if this is the last thread to reach the barrier.
                for i in range(self.num_threads):
                    self.threads_sem2.release() # Release all threads waiting on this phase's semaphore.
                self.count_threads2 = self.num_threads # Reset the counter for the next use of the barrier.
        self.threads_sem2.acquire() # All threads wait here until released by the last thread in this phase.