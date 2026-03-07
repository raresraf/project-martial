"""
@file device.py
@brief Implements components for a distributed system, likely a simulation or sensor network,
focusing on concurrent data processing, synchronization, and task management.
This module defines Device objects that manage sensor data and execute scripts
using multiple worker threads, employing various synchronization primitives
like events, locks, and a reusable barrier.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier for thread synchronization using semaphores.
    This barrier ensures that a specified number of threads all reach a certain point
    before any of them are allowed to proceed. It uses two phases for reusability.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.

        @param num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        # Counters for threads reaching the barrier in each phase.
        # Stored in a list to make them mutable within the 'phase' method.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.counter_lock = Lock()       # Lock to protect the counters.
        self.threads_sem1 = Semaphore(0) # Semaphore for threads waiting in phase 1.
        self.threads_sem2 = Semaphore(0) # Semaphore for threads waiting in phase 2.

    def wait(self):
        """
        @brief Blocks the calling thread until all 'num_threads' have reached the barrier
        and then allows them to proceed. This method executes both phases of the barrier.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements one phase of the barrier synchronization.

        @param count_threads (list): A list containing the counter for the current phase.
                                     (Using a list allows modification within the function scope).
        @param threads_sem (Semaphore): The semaphore associated with this phase.
        """
        with self.counter_lock: # Ensures exclusive access to the counter.
            count_threads[0] -= 1 # Decrements the thread count for this phase.
            # Conditional Logic: If this is the last thread to reach the barrier in this phase.
            if count_threads[0] == 0:
                # Releases all waiting threads from the semaphore.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Resets the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        threads_sem.acquire() # Threads wait here until released by the last thread.

class Device(object):
    """
    @brief Represents a single device in the distributed system.
    Each device has a unique ID, manages its sensor data, and interacts with a supervisor.
    It processes assigned scripts using a dedicated thread that spawns worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id (int): A unique identifier for the device.
        @param sensor_data (dict): A dictionary holding sensor readings for different locations.
        @param supervisor (Supervisor): A reference to the central supervisor managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Signals when scripts have been assigned.
        self.scripts = [] # List to hold (script, location) tuples assigned to this device.
        self.timepoint_done = Event() # Signals that script assignment for a timepoint is complete.
        self.thread = DeviceThread(self) # The main thread responsible for this device's lifecycle.

        self.neighbours = [] # List of neighboring devices (updated by DeviceThread).
        self.barrier = None # Reference to the global ReusableBarrierSem.
        self.threads = [] # List to hold MyThread instances (worker threads).
        self.locks = [None] * 100 # List of 100 Locks, potentially for data locations.

        self.thread.start() # Starts the main DeviceThread.

    def __str__(self):
        """
        @brief Provides a string representation of the Device.

        @return str: A formatted string indicating the device ID.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs initial setup for the device, specifically initializing the global barrier.
        The first device to call this sets up the barrier for all devices.

        @param devices (list): A list of all Device instances in the system.
        """
        # Conditional Logic: If the barrier is not yet initialized (i.e., this is the first device to set it up).
        if self.barrier is None:
            barrier = ReusableBarrierSem(len(devices)) # Initializes the global barrier.
            self.barrier = barrier # Assigns the barrier to this device.
            # Block Logic: Propagates the initialized barrier to all other devices.
            for d in devices:
                if d.barrier is None:
                    d.barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location or signals timepoint completion.

        @param script (callable): The script (function or object with a run method) to execute.
                                  If None, it signals that script assignment for the timepoint is done.
        @param location (int): The identifier of the data location the script operates on.
        """
        # (Commented out section seems to be old or experimental logic for lock assignment.)
        # have_lock_for_location = 0
        if script is not None:
            self.scripts.append((script, location)) # Appends the script and its location.
            
            # This logic below is problematic: locks should be acquired/released per access, not assigned here.
            # Assigning a new lock for each location on each script assignment will overwrite previous locks.
            # self.locks[location] = Lock() 
            # for device in self.neighbours:
            #     device.locks[location] = Lock()
            # self.script_received.set() # This event seems to be intended for when scripts are assigned.
        else:
            self.timepoint_done.set() # Signals that script assignment for the timepoint is complete.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location (int): The identifier of the data location.
        @return any: The sensor data at the specified location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location (int): The identifier of the data location to update.
        @param data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's main operational thread.
        """
        self.thread.join() # Waits for the main DeviceThread to terminate.


class MyThread(Thread):
    """
    @brief A worker thread responsible for executing a single script for a Device.
    It collects data from its device and its neighbors, runs the assigned script,
    and then propagates the updated data back.
    """
    
    def __init__(self, device, location, script, neighbours):
        """
        @brief Initializes a new MyThread instance.

        @param device (Device): The Device object this worker thread belongs to.
        @param location (int): The data location the script operates on.
        @param script (callable): The script (function or object with a run method) to execute.
        @param neighbours (list): A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for the MyThread.
        Collects data, runs the script, and updates data on devices, protected by a location-specific lock.
        """
        # Synchronization: Acquires the lock for the specific data location.
        self.device.locks[self.location].acquire() # This assumes the lock for self.location was correctly initialized beforehand.
        script_data = [] # List to collect data for the script.
        
        # Block Logic: Collects data from neighboring devices.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
        # Collects data from its own device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Conditional Logic: If any data was collected, execute the script and propagate results.
        if script_data != []:
            
            result = self.script.run(script_data) # Executes the script.

            # Block Logic: Propagates the new data to all neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result) # Updates data on its own device.
        self.device.locks[self.location].release() # Releases the lock for the specific data location.


class DeviceThread(Thread):
    """
    @brief The main thread of execution for a Device.
    It is responsible for fetching neighbor information, spawning `MyThread` worker
    threads for script processing, and managing synchronization across devices.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device (Device): The Device object this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop of the DeviceThread.
        It continuously fetches neighbor information, waits for scripts to be assigned,
        spawns worker threads to execute them, waits for worker completion,
        clears events, and then synchronizes with other devices via a global barrier.
        """
        while True:
            # Block Logic: Fetches neighbor devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Conditional Logic: If no neighbors are returned (supervisor signals shutdown), terminates.
            if neighbours is None:
                break

            self.device.timepoint_done.wait() # Waits until scripts for the timepoint are assigned.

            self.device.neighbours = neighbours # Updates the device's neighbor list.

            # Block Logic: Spawns a MyThread worker for each assigned script.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, location, script, neighbours)
                self.device.threads.append(thread) # Adds the worker thread to the device's list.
            
            # Block Logic: Starts all worker threads.
            for thread in self.device.threads:
                thread.start()
            # Block Logic: Joins all worker threads, waiting for their completion.
            for thread in self.device.threads:
                thread.join()
            self.device.threads=[] # Clears the list of worker threads.
            
            # Clears events for the next timepoint.
            self.device.timepoint_done.clear()
            self.device.barrier.wait() # Synchronizes all devices at the global barrier.
