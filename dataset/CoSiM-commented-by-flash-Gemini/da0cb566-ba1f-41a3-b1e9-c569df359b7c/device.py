"""
@da0cb566-ba1f-41a3-b1e9-c569df359b7c/device.py
@brief Defines core components for simulating a distributed sensor network or device system.
This module provides classes for devices, their operational threads, and a reusable
barrier synchronization mechanism, enabling simulation of concurrent operations
and data exchange across multiple simulated entities. This version uses global
locks for location-specific data access and coordinates device thread startup
using `threading.Event` objects.

Domain: Distributed Systems, Concurrency, Simulation.
"""

from rr import ReusableBarrier # Assuming 'rr' module contains ReusableBarrier definition.
from threading import Event, Thread, Lock


# Global dictionary to store locks for specific data locations.
# Functional Utility: Ensures only one thread can modify data at a given location at a time.
L_LOCKS = {}
# Global lock to protect access to the L_LOCKS dictionary itself.
LOCK = Lock()
# Global barrier instance, initialized by the first device (device_id 0).
BARRIER = None

class Device(object):
    """
    @brief Represents a single device in the simulated distributed system.
    Each device manages its own sensor data, processes scripts, and interacts
    with a supervisor. Device threads coordinate their startup using an Event
    and synchronize operations via a global barrier.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data readings
                            keyed by location.
        @param supervisor: A reference to the supervisor object that
                           manages inter-device communication and coordination.
        """
        # Event to signal when this device's thread is allowed to start.
        self.event = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to this device (unused in this version).
        self.script_received = Event()
        # List to store assigned scripts, each paired with its execution location.
        self.scripts = []
        # Event to signal completion of a timepoint's processing for this device.
        self.timepoint_done = Event()
        # The dedicated thread responsible for this device's operations.
        self.thread = DeviceThread(self)

    def __str__(self):
        """
        @brief Returns a string representation of the device.
        @return A string in the format "Device {device_id}".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures shared resources (global barrier) across devices and starts threads.
        This method ensures that all devices share the same global synchronization barrier.
        Only device_id 0 initializes the barrier and signals other devices to start their
        respective threads.
        @param devices: A list of all Device objects in the simulation.
        """
        # Block Logic: Device threads (except device_id 0) wait for a signal before starting.
        if self.device_id > 0:
            self.event.wait() # Wait for device 0 to signal.
            self.thread.start()
        else:
            # Block Logic: Device 0 initializes the global barrier and signals other devices.
            global BARRIER
            BARRIER = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id > 0:
                    device.event.set() # Signal other devices to start.
            self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device or signals timepoint completion.
        If a script is provided, it's added to the device's queue. If no script
        is provided (None), it indicates that the current timepoint's assignments
        are complete.
        @param script: The script object to be executed, or None to signal completion.
        @param location: The data location pertinent to the script's execution.
        """
        # Block Logic: Processes script assignment or signals timepoint completion.
        if script is None:
            # Signal that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()
        else:
            self.scripts.append((script, location))


    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.
        Note: Caller is responsible for acquiring the appropriate lock for `location`
        before calling this method to ensure thread safety.
        @param location: The location for which to retrieve data.
        @return The sensor data at the given location, or None if the location
                does not exist in the device's `sensor_data`.
        """
        # Inline: Safely retrieve data using dictionary's get method to handle missing locations.
        return self.sensor_data[location] if location in self.sensor_data \
	else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data for a specified location.
        Note: Caller is responsible for releasing the appropriate lock for `location`
        after calling this method to ensure thread safety.
        @param location: The location whose data needs to be updated.
        @param data: The new data value to set for the location.
        """
        # Block Logic: Updates sensor data only if the location already exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread.
        Ensures proper termination by waiting for the device's thread to complete its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the execution lifecycle of a single Device.
    This thread is responsible for coordinating with the supervisor to get
    neighbor information, executing assigned scripts, and synchronizing
    with other device threads using a global barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread for a given device.
        @param device: The Device object that this thread will manage.
        """
        # Functional Utility: Initializes the base Thread class with a descriptive name.
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the device thread.
        This loop continuously retrieves neighbor information, waits for
        script assignments, executes them while managing per-location locks,
        and synchronizes with other device threads via a global barrier.
        """
        # Block Logic: Initial retrieval of neighbors from the supervisor.
        neighbours = self.device.supervisor.get_neighbours()
        
        # Block Logic: The main simulation loop, continuing until no neighbors are found
        # (indicating simulation termination or a paused state).
        while True:
            # Pre-condition: All DeviceThreads must reach this global barrier before proceeding
            # to process the current timepoint's operations.
            global BARRIER
            BARRIER.wait()
            
            # Pre-condition: If no neighbors are returned, the simulation for this device ends.
            if neighbours is None:
                break

            # Block Logic: Waits until the supervisor signals that all scripts for
            # the current timepoint have been assigned.
            self.device.timepoint_done.wait()
            # Unused variable; `self.device.scripts` is used directly below.
            cs = self.device.scripts

            # Block Logic: Iterates through assigned scripts and executes them sequentially.
            for (script, location) in self.device.scripts:
                # Block Logic: Acquires the global lock to safely access/modify the L_LOCKS dictionary.
                global LOCK
                LOCK.acquire()

                # Block Logic: Retrieves or creates a lock for the specific data location.
                global L_LOCKS
                if not location in L_LOCKS.keys():
                    L_LOCKS[location] = Lock()

                # Functional Utility: Acquires the specific lock for the data location.
                # Invariant: This ensures exclusive access to the data at 'location' during script execution.
                L_LOCKS[location].acquire()
                LOCK.release() # Release the global lock once the specific lock is acquired.

                script_data = []
                
                # Block Logic: Collects sensor data from neighboring devices at the specified location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collects sensor data from the current device itself.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script only if there is any collected data.
                if script_data != []:
                    # Functional Utility: Executes the assigned script with the aggregated data.
                    result = script.run(script_data)

                    # Block Logic: Propagates the script's result to neighboring devices.
                    # Pre-condition: `device.set_data` updates the value, assuming the lock for `location`
                    # has been acquired by this thread (via `L_LOCKS[location].acquire()`).
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Updates the current device's sensor data with the script's result.
                    self.device.set_data(location, result)
                # Functional Utility: Releases the specific lock for the data location.
                L_LOCKS[location].release()

            # Reset the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()

# Assuming 'rr' module contains this class, but it's redefined here.
# If 'rr' module already exists, this is a redundant definition.
class ReusableBarrier():
    """
    @brief Implements a reusable barrier synchronization mechanism using semaphores.
    This barrier allows a fixed number of threads to synchronize multiple times,
    ensuring that no thread proceeds past the barrier until all participating
    threads have reached it. It uses a two-phase approach to allow reusability.
    This version uses a mutable list `[self.num_threads]` for counters to allow
    in-place modification.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that must reach the
                            barrier for it to be lifted.
        """
        self.num_threads = num_threads
        # Counter for the first phase of the barrier, stored in a mutable list.
        self.count_threads1 = [self.num_threads]
        # Counter for the second phase of the barrier, enabling reusability, stored in a mutable list.
        self.count_threads2 = [self.num_threads]
        # Lock to protect access to the thread counters.
        self.count_lock = Lock()
        # Semaphore for the first synchronization phase. Initialized to 0
        # so threads wait until all have arrived.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for the second synchronization phase. Initialized to 0
        # for reusability, ensures threads wait for reset.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this point.
        Orchestrates the two-phase barrier synchronization.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements one phase of the reusable barrier logic.
        Threads decrement a counter and the last thread to reach zero
        releases all waiting threads via a semaphore, then resets the counter.
        @param count_threads: The mutable list holding the thread counter for this phase.
        @param threads_sem: The semaphore used for synchronization in this phase.
        """
        # Block Logic: Critical section for safely decrementing the thread counter.
        with self.count_lock:
            count_threads[0] -= 1
            # Invariant: If this is the last thread, release all waiting threads.
            if count_threads[0] == 0:
                # Release all threads waiting on the semaphore.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of the barrier.
                count_threads[0] = self.num_threads
	
        # Wait until all other threads have reached this phase and the semaphore is released.
        threads_sem.acquire()

class MyThread(Thread):
    """
    @brief Example/Test class for demonstrating the ReusableBarrier.
    This class is not directly used in the main Device simulation logic.
    """
    
    def __init__(self, tid, barrier):
        """
        @brief Initializes an example thread.
        @param tid: Thread ID.
        @param barrier: The barrier instance to use for synchronization.
        """
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier

    def run(self):
        """
        @brief Example run method that repeatedly waits on the barrier.
        """
        # Block Logic: Iterates to demonstrate repeated barrier synchronization.
        for i in xrange(10): # xrange is Python 2 syntax, equivalent to range in Python 3.
            self.barrier.wait()
            # Functional Utility: Prints a message after passing the barrier.
            print "I'm Thread " + str(self.tid) + \
            " after barrier, in step " + str(i) + "\n",
