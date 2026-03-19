"""
This module implements a multi-threaded device simulation framework.

It defines a reusable barrier for thread synchronization, a `Device` class
representing simulated entities, a `DeviceThread` to manage device operations,
and `DeviceWorker` threads to execute scripts. The framework supports
inter-device communication and script processing in a concurrent manner,
utilizing a shared barrier and per-device locks for coordination.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue # Standard Python queue, although the usage is for worker management, not directly Queue.Queue objects being passed around.


class ReusableBarrier(object):
    """
    Implements a reusable barrier synchronization primitive.

    This barrier ensures that a fixed number of participating threads
    wait at a designated point until all threads have arrived. Once all
    threads reach the barrier, they are all released simultaneously.
    It uses a two-phase mechanism (`phase1`, `phase2` or general `phase` method)
    to allow for reuse without busy-waiting.
    """

    def __init__(self, num_threads):
        """
        Initializes the reusable barrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                before any are released.
        """
        self.num_threads = num_threads
        # Counters for threads in phase 1 and phase 2.
        # Wrapped in a list `[value]` to allow modification of the value from within nested scopes.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()       # Lock to protect the counters during modifications.
        
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase, initially blocking all threads.
        
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase, initially blocking all threads.

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        participating threads have also called wait(). This method orchestrates
        the two phases of the barrier to enable reuse.
        """
        self.phase(self.count_threads1, self.threads_sem1) # Execute the first barrier phase.
        self.phase(self.count_threads2, self.threads_sem2) # Execute the second barrier phase.

    def phase(self, count_threads, threads_sem):
        """
        Manages a single synchronization phase of the barrier.

        Threads acquire the `count_lock`, decrement their count, and if they
        are the last thread to reach this phase, they release all waiting threads
        and reset the counter. Otherwise, they wait on the phase's semaphore.

        Args:
            count_threads (list): A list containing the counter for the current phase (e.g., `[value]`).
            threads_sem (Semaphore): The semaphore associated with the current phase.
        """
        with self.count_lock: # Protect access to the counter.
            count_threads[0] -= 1 # Decrement the count of threads yet to reach the barrier.
            
            if count_threads[0] == 0: # Check if this is the last thread to reach the barrier.
                for i in range(self.num_threads):
                    threads_sem.release() # Release all threads waiting on this phase's semaphore.
                
                count_threads[0] = self.num_threads # Reset the counter for the next use of the barrier.
        threads_sem.acquire() # All threads wait here until released by the last thread in this phase.


class Device(object):
    """
    Represents a single simulated device in the system.

    Each device manages its own `sensor_data`, interacts with a `supervisor`,
    and coordinates script execution. Devices synchronize globally via a shared
    barrier (`neighbours_barrier`) and manage data consistency with `set_lock`.
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
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # `result_queue` is declared but not explicitly used in the provided code.
        self.result_queue = Queue.Queue()
        self.set_lock = Lock()          # Lock to protect `self.sensor_data` during `set_data` operations.
        self.neighbours_lock = None     # Global lock for neighbor access, initialized by device 0.
        self.neighbours_barrier = None  # Global barrier for device synchronization, initialized by device 0.

        self.script_received = Event()  # Event to signal when new scripts have been assigned.
        self.scripts = []               # List of (script, location) tuples assigned to this device.
        self.timepoint_done = Event()   # Event to signal that all scripts for a timepoint have been assigned.

        # The main thread responsible for this device's control flow and script delegation.
        self.thread = DeviceThread(self)

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global shared synchronization primitives (neighbours_lock and neighbours_barrier).
        This method is designed to be called once by all devices, but global initialization
        logic is handled by the device with the lowest `device_id` (assumed to be `devices[0]`).

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Global initialization: Only the first device (assumed device_id 0) creates
        # and distributes the global locks and barrier.
        if self.device_id == devices[0].device_id:
            self.neighbours_lock = Lock()                       # Create the global neighbors lock.
            self.neighbours_barrier = ReusableBarrier(len(devices)) # Create the global barrier.
        
        else:
            # Other devices receive the shared locks and barrier from the first device.
            self.neighbours_lock = devices[0].neighbours_lock
            self.neighbours_barrier = devices[0].neighbours_barrier

        self.thread.start() # Start the DeviceThread after global resources are set up.

    def assign_script(self, script, location):
        """
        Assigns a processing script and its associated data location to this device.
        If `script` is None, it signals that script assignment for the current
        timepoint is complete.

        Args:
            script (object): The script object (must have a `run` method) to be executed,
                             or None to signal timepoint completion.
            location (int): The data location this script pertains to.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the new script to the list.
            self.script_received.set()              # Signal that a new script has been received.
        else:
            self.script_received.set() # Signal that script processing is done (even if no new script).
            self.timepoint_done.set()  # Signal that all scripts for the current timepoint have been assigned.

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location within this device.

        Note: This method does *not* acquire any locks to protect `self.sensor_data`.
        If `get_data` is called concurrently with `set_data` (which uses `set_lock`),
        or by multiple worker threads on the same device, this could lead to race
        conditions and inconsistent data reads. External synchronization is required.

        Args:
            location (int): The location ID for which to retrieve data.

        Returns:
            Any: The data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location within this device.
        Protects access to `self.sensor_data` using `self.set_lock`.

        Args:
            location (int): The location ID for which to set data.
            data (Any): The new data value to be set.
        """
        self.set_lock.acquire() # Acquire lock before modifying sensor_data.
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.set_lock.release() # Release lock after modification.

    def shutdown(self):
        """
        Initiates the graceful shutdown sequence for the device's main thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a `Device`.

    It is responsible for interacting with the supervisor to get neighbor
    information, managing timepoint progression, and dispatching scripts
    to a pool of `DeviceWorker` threads. It also participates in global
    synchronization using a shared barrier.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = [] # List to hold `DeviceWorker` threads.

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously processes timepoints:
        1. Retrieves neighbor information from the supervisor.
        2. Waits for scripts to be assigned.
        3. Initializes and manages a pool of `DeviceWorker` threads, distributing
           scripts among them for concurrent execution.
        4. Waits for all worker threads to complete their tasks.
        5. Participates in global barrier synchronization.
        6. Clears the `script_received` event for the next timepoint.
        """
        while True:
            # Acquire global neighbors_lock, retrieve neighbors, then release lock.
            self.device.neighbours_lock.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.neighbours_lock.release()

            # If no neighbors are returned (e.g., shutdown signal from supervisor),
            # break the loop and terminate this thread.
            if neighbours is None:
                break

            # Waits until new scripts have been assigned to this device for the current timepoint.
            self.device.script_received.wait()

            # Initializes a pool of 8 DeviceWorker threads.
            self.workers = []
            for i in range(8):
                self.workers.append(DeviceWorker(self.device, i, neighbours))

            # Distributes scripts to workers in a load-balancing fashion.
            # It attempts to assign scripts to workers based on locations they already handle,
            # or to the worker with the fewest assigned locations.
            for (script, location) in self.device.scripts:
                added = False
                for worker in self.workers:
                    if location in worker.locations: # If worker already handles this location, assign it.
                        worker.add_script(script, location)
                        added = True
                        break # Script assigned, move to next script.

                if added == False: # If no worker was found that already handles this location.
                    # Assign the script to the worker that currently has the fewest assigned locations.
                    minimum = len(self.workers[0].locations)
                    chosen_worker = self.workers[0]
                    for worker in self.workers:
                        if minimum > len(worker.locations):
                            minimum = len(worker.locations)
                            chosen_worker = worker

                    chosen_worker.add_script(script, location)

            # Start all worker threads.
            for worker in self.workers:
                worker.start()

            # Wait for all worker threads to complete their assigned tasks.
            for worker in self.workers:
                worker.join()

            # Participates in the global barrier synchronization, waiting for all devices
            # to complete their current timepoint processing of scripts.
            self.device.neighbours_barrier.wait()
            self.device.script_received.clear() # Clear the event for the next script assignment.


class DeviceWorker(Thread):
    """
    A worker thread responsible for executing a subset of scripts assigned to a device.
    These threads are created by `DeviceThread` for each timepoint's scripts.
    """
    
    def __init__(self, device, worker_id, neighbours):
        """
        Initializes a DeviceWorker instance.

        Args:
            device (Device): The Device instance this worker operates for.
            worker_id (int): A unique identifier for this worker within its device's pool.
            neighbours (list): The list of neighboring devices for the current timepoint.
        """
        Thread.__init__(self)
        self.device = device
        self.worker_id = worker_id
        self.scripts = []          # List of script objects assigned to this worker.
        self.locations = []        # List of locations corresponding to the assigned scripts.
        self.neighbours = neighbours # Neighbors for data exchange.

    def add_script(self, script, location):
        """
        Assigns a script and its associated data location to this worker.

        Args:
            script (object): The script object to be executed.
            location (int): The data location relevant to this script.
        """
        self.scripts.append(script)
        self.locations.append(location)

    def run_scripts(self):
        """
        Executes all scripts assigned to this worker.

        For each script, it collects data from neighboring devices and the local device,
        runs the script, and then updates the data in relevant devices.
        """
        for (script, location) in zip(self.scripts, self.locations):
            script_data = [] # List to collect all relevant data for the script.
            
            # Gathers data from all neighboring devices for the current location.
            # `get_data` itself does not use a lock; external locking is expected for consistency.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Gathers data from its own device for the current location.
            # `get_data` itself does not use a lock; external locking is expected for consistency.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # If any data was collected, run the script and update devices.
            if script_data != []:
                res = script.run(script_data) # Execute the script.

                # Updates the data in neighboring devices with the script's result.
                # `set_data` itself uses `device.set_lock`.
                for device in self.neighbours:
                    device.set_data(location, res)
                
                # Updates its own device's data with the script's result.
                # `set_data` itself uses `device.set_lock`.
                self.device.set_data(location, res)

    def run(self):
        """
        The main execution logic for `DeviceWorker`.
        It simply calls `run_scripts` to process its assigned tasks.
        """
        self.run_scripts()