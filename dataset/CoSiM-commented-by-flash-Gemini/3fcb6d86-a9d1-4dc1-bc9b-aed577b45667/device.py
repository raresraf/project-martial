"""
@3fcb6d86-a9d1-4dc1-bc9b-aed577b45667/device.py
@brief Implements a simulated device with a custom reusable barrier for thread synchronization in a distributed system.
This module defines a `Device` that processes sensor data and executes scripts,
a `DeviceThread` for its operational logic, and a `ReusableBarrier` for coordinating
multiple threads through a two-phase semaphore mechanism. It also includes a `MyThread`
class for demonstrating the barrier's functionality.
"""

from threading import *


class ReusableBarrier():
    """
    @brief Implements a reusable barrier for synchronizing a fixed number of threads.
    This barrier ensures that all participating threads wait at a synchronization point
    until every thread has reached it, after which all are released simultaneously.
    It uses a two-phase approach with semaphores to allow for repeated use.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that will participate in this barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Counter for the first phase of the barrier.
        self.count_threads2 = [self.num_threads] # Counter for the second phase of the barrier.
        self.count_lock = Lock()                 # Lock to protect access to thread counters.
        self.threads_sem1 = Semaphore(0)         # Semaphore for releasing threads from the first phase.
        self.threads_sem2 = Semaphore(0)         # Semaphore for releasing threads from the second phase.
 
    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this barrier.
        Orchestrates the two-phase synchronization process.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single phase of the barrier synchronization.
        Threads decrement a shared counter; the last thread to decrement releases all others
        waiting on the semaphore for this phase.
        @param count_threads: A list containing the counter for the current phase (used as mutable int).
        @param threads_sem: The semaphore associated with the current phase.
        Invariant: All threads are held at `threads_sem.acquire()` until `count_threads[0]` reaches zero.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                # Block Logic: The last thread to reach this point releases all waiting threads for this phase.
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  # Reset counter for reuse.
        threads_sem.acquire()                    # All threads wait here.
                                                 
 
class MyThread(Thread):
    """
    @brief A demonstration thread class for testing the `ReusableBarrier`.
    Each thread iterates, prints its ID and step, and then waits on the barrier.
    """
    def __init__(self, tid, barrier):
        """
        @brief Initializes a `MyThread` instance.
        @param tid: The thread's unique identifier.
        @param barrier: The `ReusableBarrier` instance to synchronize with.
        """
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
 
    def run(self):
        """
        @brief The main execution loop for `MyThread`.
        It repeatedly waits on the barrier and prints a message.
        """
        # Block Logic: Iterates a fixed number of times, synchronizing at the barrier in each step.
        for i in range(10): # Using range(10) instead of xrange(10) for Python 3 compatibility.
            self.barrier.wait()
            print ("I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n"),


class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread and a shared barrier mechanism.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's local sensor readings.
        @param supervisor: The supervisor object responsible for managing the overall simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the list of all devices and initializes a shared reusable barrier.
        Only the device with `device_id == 0` initializes the barrier.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        self.devices = devices
        nrDeviceuri = len(devices)
        
        # Block Logic: Initializes a shared barrier if this is the first device (device_id == 0).
        # Invariant: A single `ReusableBarrier` instance is created and implicitly shared among devices.
        if self.device_id == 0: # Changed `self is devices[0]` to `self.device_id == 0` for correctness.
            self.bar = ReusableBarrier(nrDeviceuri)
        if self.device_id != 0: # Changed `self is not devices[0]` for correctness.
            print ("Nu e primu")
            

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        Signals that a script has been received, or that a timepoint is done if no script.
        @param script: The script object to be executed, or `None` to signal completion.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Block Logic: Signals completion of timepoint setup if no script is assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        @param location: The key for the sensor data to be modified.
        @param data: The new data value to store.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread, waiting for its graceful completion.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    executing scripts, and synchronizing with other device threads using a reusable barrier.
    Time Complexity: O(T * S * (N + D)) where T is the number of timepoints, S is the number of scripts per device,
    N is the number of neighbors, and D is the data retrieval/setting operations.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a `DeviceThread` instance.
        @param device: The `Device` instance that this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        
    def run(self):
        """
        @brief The main loop for the device's operational thread.
        Block Logic:
        1. Continuously fetches neighbor information from the supervisor.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        2. Waits for the `timepoint_done` event to be set, indicating that scripts are ready to be processed.
        3. Executes assigned scripts, collecting data from neighbors and its own device,
           executing the script, and propagating the results.
           Invariant: For each script, all necessary data is gathered, the script is run,
           and results are disseminated to relevant devices.
        4. Synchronizes with all other device threads using a shared barrier.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        5. Clears the `timepoint_done` event for the next timepoint.
        """
        while True:    
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits for the `timepoint_done` event to be set, indicating that
            # all scripts for the current timepoint have been assigned or processed.
            self.device.timepoint_done.wait()
          
            
            # Block Logic: Executes the scripts assigned to the device for the current timepoint.
            # Invariant: Each script retrieves data from neighbors and itself, executes, and updates data.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Block Logic: Collects data from neighboring devices for the specified location.
                # Note: This implementation does not use explicit locks for neighbor data access,
                # which might lead to race conditions if not handled by higher-level mechanisms.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collects data from its own device for the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if any data was collected and propagates the result.
                if script_data != []:
                    
                    result = script.run(script_data)

                    # Block Logic: Updates neighboring devices with the script's result.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Updates its own device's data with the script's result.
                    self.device.set_data(location, result)
         
            # Block Logic: Synchronizes with other device threads using a shared barrier,
            # ensuring all devices complete their processing before proceeding.
            # Removed redundant `if` check around `barrier.wait()` as it doesn't return a boolean.
            self.device.devices[0].bar.wait()
            # Block Logic: Clears the `timepoint_done` event for the next timepoint.
            # Removed redundant `if` check around `timepoint_done.clear()` as it doesn't return a boolean.
            self.device.timepoint_done.clear()
