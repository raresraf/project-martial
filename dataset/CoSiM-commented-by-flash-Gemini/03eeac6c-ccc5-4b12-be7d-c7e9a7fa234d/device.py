

# @file 03eeac6c-ccc5-4b12-be7d-c7e9a7fa234d/device.py
# @brief Implements a simulated device for distributed computation with multi-threaded script execution and barrier synchronization.
#
# This module defines the core components for simulating a network of computational devices.
# Each `Device` manages its own sensor data, processes scripts, and interacts with neighbors.
# `DeviceThread` orchestrates the device's operational logic, spawning `ScriptThread` workers
# to execute individual scripts. Synchronization across devices is managed by a `ReusableBarrier`.
# The `MyObjects` class serves as a data container for passing information to worker threads.

from threading import Event, Thread, Lock
from reusable_barrier_semaphore import ReusableBarrier
import Queue # Renamed from 'Queue' to 'queue' for Python 3 compatibility if needed, but keeping original for now.
NUMBER_OF_THREADS = 8 # Functional Utility: Defines the fixed number of script execution threads per device.

class Device(object):
    """
    @brief Represents a simulated computational device within a distributed environment.
    
    Each device has a unique ID, sensor data, a supervisor for network topology,
    and manages scripts for local execution. It uses a dedicated thread for its
    main loop and synchronizes with other devices via a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device object.

        @param device_id: A unique integer identifier for this device.
        @param sensor_data: A dictionary containing the initial sensor data for this device.
                            Keys represent data locations, values are sensor readings.
        @param supervisor: A reference to the supervisor object responsible for managing the network of devices.
        @post The device is initialized with its ID, data, and a thread for execution.
        @post Event flags for script arrival and timepoint completion are set up.
        @post A lock for protecting shared data access is initialized.
        """
        # Functional Utility: Unique identifier for the device.
        self.device_id = device_id
        # Functional Utility: Stores the device's sensor data, mapping locations to values.
        self.sensor_data = sensor_data
        # Functional Utility: Reference to the central entity managing device interactions and network state.
        self.supervisor = supervisor
        # Functional Utility: Event flag signaling that new scripts have been assigned to the device.
        self.script_received = Event()
        # Functional Utility: List to hold scripts assigned to the device for the current timepoint.
        self.scripts = []
        # Functional Utility: Event flag signaling the completion of processing for the current simulation timepoint.
        self.timepoint_done = Event()
        # Functional Utility: The dedicated thread responsible for the device's main operational logic.
        self.thread = DeviceThread(self)
        # Functional Utility: Initiates the execution of the device's dedicated thread.
        self.thread.start()
        # Functional Utility: Reference to the shared synchronization barrier used by all devices.
        self.barrier = None
        # Functional Utility: A lock to protect access to the device's sensor data from concurrent modifications.
        self.data_lock = Lock()

    def __str__(self):
        """
        @brief Provides a string representation of the Device.

        @return: A formatted string indicating the device's ID.
        """
        # Functional Utility: Returns a human-readable identifier for the device.
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the shared synchronization barrier across all devices.

        This method ensures that all devices in the simulation share the same barrier
        instance, crucial for synchronized time-step progression.
        @param devices: A list of all Device objects participating in the simulation.
        @pre `devices` is a non-empty list of Device instances.
        @post A `ReusableBarrier` instance is created (if not already present) and assigned to all devices.
        """
        # Functional Utility: Creates a single `ReusableBarrier` instance to be shared by all devices.
        # It's initialized with the total number of devices in the simulation.
        barrier = ReusableBarrier(len(devices))

        # Block Logic: Ensures that the barrier is initialized only once for the first device calling this method.
        if self.barrier is None:
            self.barrier = barrier

        # Block Logic: Iterates through all devices to ensure each one is assigned the same shared barrier instance.
        # Invariant: Each device in `devices` will have its `barrier` attribute set to the common barrier.
        for device in devices:
            if device.barrier is None: # Inline: Checks if the device has not yet received a barrier.
                device.barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for processing or signals timepoint completion.

        @param script: The script object to execute. If None, it signals that no more scripts
                       are expected for the current timepoint, and processing can proceed.
        @param location: The data location relevant to the script.
        @post If a script is provided, it's added to the device's script list and `script_received` is set.
        @post If `script` is None, `timepoint_done` is set to signal completion for the current timepoint.
        """
        # Block Logic: Differentiates between receiving an actual script and a signal for timepoint completion.
        if script is not None:
            # Functional Utility: Adds the new script and its associated data location to the device's queue of scripts to be executed.
            self.scripts.append((script, location))
            # Functional Utility: Signals to the `DeviceThread` that new scripts are available for processing.
            self.script_received.set()
        else:
            # Functional Utility: Signals to the `DeviceThread` that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The data location for which to retrieve the sensor reading.
        @return: The sensor data for the specified location, or None if the location is not found.
        @pre The `data_lock` should ideally be acquired by the caller before accessing this method
             to ensure thread-safe read operations on `sensor_data` if direct dictionary access
             is not sufficiently atomic for the use case. (Note: Current implementation acquires lock inside `ScriptThread`).
        """
        # Functional Utility: Safely accesses the `sensor_data` dictionary to retrieve the value for a given `location`.
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data for a specific location.

        @param location: The data location to update.
        @param data: The new sensor data value.
        @post The sensor data at the specified location is updated if the location exists.
        @pre The `data_lock` should ideally be acquired by the caller before accessing this method
             to ensure thread-safe write operations on `sensor_data`. (Note: Current implementation acquires lock inside `ScriptThread`).
        """
        # Block Logic: Updates the sensor data only if the specified `location` is a valid key in `sensor_data`.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's dedicated thread.

        @post The `DeviceThread` is joined, ensuring its graceful termination and completion of all tasks.
        """
        # Functional Utility: Blocks the calling thread until the `DeviceThread` has completed its execution.
        self.thread.join()

class DeviceThread(Thread):
    """
    @brief The dedicated control thread for a simulated device.
    
    This thread manages the device's operational lifecycle, including fetching
    neighbor information, processing assigned scripts concurrently using a pool
    of `ScriptThread` workers, and synchronizing with other devices via a barrier.
    """
    
    # Functional Utility: A class-level dictionary to store locks for specific data locations,
    # ensuring thread-safe access when multiple script threads might operate on the same location.
    location_locks = {}

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.

        @param device: The `Device` object that this thread is responsible for.
        @post The thread is initialized with a descriptive name and associated with the device.
        @post An empty list for worker threads and a queue for scripts are created.
        """
        # Functional Utility: Calls the base `Thread` class constructor, assigning a unique name to the thread.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        # Functional Utility: Stores a reference to the `Device` instance that this thread manages.
        self.device = device
        # Functional Utility: An empty list to hold references to the `ScriptThread` worker instances.
        self.threads = []
        # Functional Utility: A thread-safe queue to distribute scripts to the `ScriptThread` workers.
        self.scripts_queue = Queue.Queue() # Using Queue.Queue for thread-safe FIFO.

    def run(self):
        """
        @brief The main execution loop for the `DeviceThread`.

        This loop continuously manages the device's state: it fetches neighbors,
        spawns a fixed pool of `ScriptThread` workers, dispatches scripts to them,
        waits for all scripts to complete, and then synchronizes with other devices
        using a barrier before moving to the next simulation timepoint.
        @pre The `device` object is fully initialized.
        @post The thread runs indefinitely until a termination signal (None neighbors) is received.
        """
        # Block Logic: Initializes a pool of `ScriptThread` workers.
        # Invariant: Exactly `NUMBER_OF_THREADS` worker threads are created and added to `self.threads`.
        for _ in range(NUMBER_OF_THREADS):
            self.threads.append(ScriptThread(self.scripts_queue))

        # Block Logic: Starts all `ScriptThread` workers.
        # Invariant: Each `script_thread` in `self.threads` is started.
        for script_thread in self.threads:
            script_thread.start()

        # Block Logic: Main simulation loop for the device.
        # Invariant: Continues until the supervisor signals termination.
        while True:
            # Functional Utility: Retrieves information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Checks if a termination signal (no neighbors) has been received from the supervisor.
            if neighbours is None:
                # Block Logic: Sends termination signals to all `ScriptThread` workers.
                # Invariant: Each `script_thread` receives a `MyObjects` instance with `stop` set to `False`.
                for script_thread in self.threads:
                    self.scripts_queue.put(MyObjects(None, None, None, None,
                                                     False, None)) # Inline: `stop=False` acts as a termination signal.
                break # Functional Utility: Exits the main simulation loop.

            # Functional Utility: Blocks until the `timepoint_done` event is set, indicating scripts are assigned.
            self.device.timepoint_done.wait()
            # Block Logic: Dispatches each assigned script to the `ScriptThread` workers via the queue.
            # Invariant: Each script in `self.device.scripts` is placed into the `scripts_queue`.
            for (script, location) in self.device.scripts:
                # Block Logic: If a lock for this location doesn't exist, create one.
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()

                # Functional Utility: Puts the script details into the queue for a worker thread to pick up.
                self.scripts_queue.put(MyObjects(self.device, location, script,
                                                 neighbours, True, # Inline: `stop=True` indicates a regular script to be processed.
                                                 self.location_locks),
                                       block=True, timeout=None) # Inline: Blocks indefinitely until space is available in the queue.
            # Functional Utility: Clears the `timepoint_done` event for the next simulation timepoint.
            self.device.timepoint_done.clear()

            # Functional Utility: Waits at the shared barrier until all devices have completed their script processing for the current timepoint.
            self.device.barrier.wait()

        # Block Logic: Joins all `ScriptThread` workers after the main loop has terminated.
        # Invariant: Each `script_thread` is joined, ensuring their complete shutdown.
        for script_thread in self.threads:
            script_thread.join()

class ScriptThread(Thread):
    """
    @brief A worker thread responsible for executing individual scripts on behalf of a device.
    
    This thread continuously retrieves script execution tasks from a shared queue,
    gathers necessary data, executes the script, and applies the results back to
    the device's sensor data, ensuring thread-safe access where needed.
    """

    def __init__(self, queue):
        """
        @brief Initializes a ScriptThread.

        @param queue: The shared `Queue.Queue` instance from which script tasks are retrieved.
        @post The thread is initialized with a descriptive name and a reference to the script queue.
        """
        # Functional Utility: Calls the base `Thread` class constructor, assigning a fixed name.
        Thread.__init__(self, name="Script Thread")
        # Functional Utility: Stores a reference to the shared `scripts_queue` from `DeviceThread`.
        self.queue = queue

    def run(self):
        """
        @brief The main execution loop for the `ScriptThread`.

        The thread continuously retrieves `MyObjects` instances from the queue.
        If a termination signal is received (`stop == False`), the thread exits.
        Otherwise, it acquires necessary locks, gathers data from the local device
        and neighbors, executes the encapsulated script, and updates sensor data.
        @pre The `queue` is properly initialized and populated by the `DeviceThread`.
        @post The thread processes scripts until a termination signal is received.
        """
        # Block Logic: Main loop for the script worker thread.
        # Invariant: Continuously processes scripts or terminates upon signal.
        while True:
            # Functional Utility: Blocks indefinitely until a `MyObjects` task is available in the queue.
            my_objects = self.queue.get(block=True, timeout=None)

            # Block Logic: Checks if the retrieved object is a termination signal.
            if my_objects.stop == False: # Inline: `stop=False` is the termination flag for worker threads.
                break # Functional Utility: Exits the worker thread's execution loop.

            # Functional Utility: Acquires a location-specific lock to ensure exclusive access to data at `my_objects.location`.
            my_objects.location_locks[my_objects.location].acquire()

            script_data = [] # Functional Utility: List to aggregate data for script execution.
            
            # Block Logic: Gathers data from neighboring devices.
            # Invariant: Data is appended to `script_data` only if it exists for the given `location`.
            for device in my_objects.neighbours:
                # Functional Utility: Retrieves sensor data from a neighboring device.
                data = device.get_data(my_objects.location)

                if data is not None: # Inline: Checks if valid data was retrieved from the neighbor.
                    script_data.append(data)

            # Functional Utility: Acquires the device's main data lock for thread-safe access to its sensor data.
            my_objects.device.data_lock.acquire()
            # Functional Utility: Retrieves sensor data from the local device.
            data = my_objects.device.get_data(my_objects.location)
            # Functional Utility: Releases the device's main data lock.
            my_objects.device.data_lock.release()

            if data is not None: # Inline: Checks if valid data was retrieved from the local device.
                script_data.append(data)

            # Block Logic: Executes the script if any data has been collected.
            if script_data != []:
                # Functional Utility: Executes the assigned script with the aggregated data and stores the result.
                result = my_objects.script.run(script_data)

                # Block Logic: Propagates the script result to all participating neighboring devices.
                # Invariant: Each neighbor's data is updated with the result, protected by its `data_lock`.
                for device in my_objects.neighbours:
                    device.data_lock.acquire() # Functional Utility: Acquires the neighbor's data lock.
                    device.set_data(my_objects.location, result) # Functional Utility: Updates the neighbor's sensor data.
                    device.data_lock.release() # Functional Utility: Releases the neighbor's data lock.

                # Functional Utility: Propagates the script result to the local device.
                my_objects.device.data_lock.acquire() # Functional Utility: Acquires the local device's data lock.
                my_objects.device.set_data(my_objects.location, result) # Functional Utility: Updates the local device's sensor data.
                my_objects.device.data_lock.release() # Functional Utility: Releases the local device's data lock.

            # Functional Utility: Releases the location-specific lock, allowing other threads to access data at this location.
            my_objects.location_locks[my_objects.location].release()

class MyObjects():
    """
    @brief A simple data class used to encapsulate and pass multiple objects as a single unit
           through a queue to `ScriptThread` workers.
    
    This helps in organizing the parameters required for script execution.
    """

    def __init__(self, device, location, script, neighbours, stop, location_locks):
        """
        @brief Initializes a MyObjects instance.

        @param device: The `Device` object on which the script is to be executed.
        @param location: The data location pertinent to the script.
        @param script: The script object to be executed.
        @param neighbours: A list of neighboring `Device` objects.
        @param stop: A boolean flag; `True` for a regular task, `False` for a termination signal.
        @param location_locks: A dictionary of locks for different locations, ensuring thread-safe data access.
        @post All provided parameters are stored as attributes of the `MyObjects` instance.
        """
        # Functional Utility: Stores the `Device` instance associated with this script execution.
        self.device = device
        # Functional Utility: Stores the specific data location the script pertains to.
        self.location = location
        # Functional Utility: Stores the script object to be executed.
        self.script = script
        # Functional Utility: Stores a list of neighboring `Device` objects for data interaction.
        self.neighbours = neighbours
        # Functional Utility: Flag indicating whether this is a regular script task (`True`) or a thread termination signal (`False`).
        self.stop = stop
        # Functional Utility: Reference to the dictionary of location-specific locks managed by `DeviceThread`.
        self.location_locks = location_locks
# Functional Utility: Imports all members from the `threading` module, although not all are explicitly used here.

# ReusableBarrier class definition (as provided in reusable_barrier_semaphore.py)
class ReusableBarrier():
    """
    @brief Implements a reusable barrier for synchronizing multiple threads using semaphores in two phases.
    
    This barrier ensures that a specified number of threads wait for each other at a synchronization
    point before any of them can proceed. It is designed to be reusable across multiple synchronization cycles.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier.

        @param num_threads: The total number of threads that must reach the barrier to release.
        @pre `num_threads` is a positive integer.
        @post Internal counters and semaphores (`threads_sem1`, `threads_sem2`) are initialized.
        @post A `count_lock` is created to protect access to shared counters.
        """
        # Functional Utility: Stores the total number of threads required to reach the barrier.
        self.num_threads = num_threads
        # Functional Utility: Counter for the first synchronization phase. Wrapped in a list to allow modification within `with` statement.
        self.count_threads1 = [self.num_threads]
        # Functional Utility: Counter for the second synchronization phase. Wrapped in a list for mutability.
        self.count_threads2 = [self.num_threads]
        # Functional Utility: A lock to protect `count_threads1` and `count_threads2` from race conditions.
        self.count_lock = Lock()                 
        # Functional Utility: Semaphore for controlling the release of threads in the first phase.
        self.threads_sem1 = Semaphore(0)         
        # Functional Utility: Semaphore for controlling the release of threads in the second phase.
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        """
        @brief Blocks the calling thread until all expected threads have completed both synchronization phases.

        @post The thread has successfully passed through both `phase1` and `phase2` of the barrier.
        """
        # Functional Utility: Invokes the first phase of the barrier.
        self.phase(self.count_threads1, self.threads_sem1)
        # Functional Utility: Invokes the second phase of the barrier.
        self.phase(self.count_threads2, self.threads_sem2)
    
    def phase(self, count_threads, threads_sem):
        """
        @brief Implements a single phase of the reusable barrier synchronization.

        A thread decrements a shared counter. If it's the last thread to do so,
        it releases all waiting threads via a semaphore and resets the counter.
        Otherwise, it waits on the semaphore.
        @param count_threads: A mutable container (e.g., list) holding the counter for this phase.
        @param threads_sem: The `Semaphore` instance used for this phase.
        @post The calling thread has been released from this phase of the barrier.
        """
        # Block Logic: Ensures atomic decrement of the counter and checking for the last thread.
        with self.count_lock:
            count_threads[0] -= 1 # Functional Utility: Decrements the thread count for the current phase.
            # Block Logic: Checks if the current thread is the last one to reach this phase.
            if count_threads[0] == 0:            
                # Functional Utility: Releases all `num_threads` that are waiting on this semaphore.
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads # Functional Utility: Resets the counter for the next use of this phase.
        threads_sem.acquire() # Functional Utility: Blocks the current thread until it's released by the last thread in the phase.                    
                                                 

# Example usage of MyThread (not part of the core simulation logic, but for demonstration)
class MyThread(Thread):
    """
    @brief An example thread class demonstrating the usage of the ReusableBarrier.
    
    This thread repeatedly waits on a barrier and prints its ID and step number.
    """
    def __init__(self, tid, barrier):
        """
        @brief Initializes a MyThread instance.

        @param tid: The unique ID of this thread.
        @param barrier: A reference to the `ReusableBarrier` instance to synchronize with.
        @post The thread is initialized with its ID and a reference to the barrier.
        """
        # Functional Utility: Calls the base `Thread` class constructor.
        Thread.__init__(self)
        # Functional Utility: Stores the unique ID of this thread.
        self.tid = tid
        # Functional Utility: Stores a reference to the shared `ReusableBarrier`.
        self.barrier = barrier
    
    def run(self):
        """
        @brief The main execution loop for the `MyThread`.

        The thread repeatedly waits at the barrier and prints a message.
        @post The thread repeatedly synchronizes with other threads via the barrier and logs its progress.
        """
        # Block Logic: Loops a fixed number of times to demonstrate barrier usage over multiple steps.
        for i in xrange(10): # Inline: Performs 10 iterations of waiting at the barrier.
            self.barrier.wait() # Functional Utility: Blocks until all threads have reached this point in the current iteration.
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",