"""
@6ddff196-fece-4f9c-9248-43d8e567d02d/device.py
@brief Implements a simulated device for a distributed sensor network, with sequential script execution and dynamic, global location-based locking.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` for operational logic and uses a `ReusableBarrierSem`
for global time-step synchronization. Data access is protected by `Lock` objects
stored in a global dictionary `dic`, which are dynamically created and managed
on a per-location basis.
"""

import sys
from threading import *


class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier for synchronizing a fixed number of threads using Locks and Semaphores.
    This barrier ensures that all participating threads wait at a synchronization point
    until every thread has reached it, after which all are released simultaneously.
    It uses a two-phase semaphore approach for reusability.
    """
    
    
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that will participate in this barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for the first phase of the barrier.


        self.count_threads2 = self.num_threads # Counter for the second phase of the barrier.
        self.counter_lock = Lock()               # Lock to protect access to thread counters.
        self.threads_sem1 = Semaphore(0)         # Semaphore for releasing threads from the first phase.
        self.threads_sem2 = Semaphore(0)         # Semaphore for releasing threads from the second phase.
    
    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have completed both phases of the barrier.
        Orchestrates the two-phase synchronization process.
        """
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """
        @brief The first phase of the barrier synchronization.
        Threads decrement a counter; the last thread to decrement releases all others for this phase.
        Invariant: All threads are held at `threads_sem1.acquire()` until `count_threads1` reaches zero.
        """
        # Block Logic: Atomically decrements the counter for phase 1.
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Block Logic: The last thread to reach this point releases all waiting threads from phase 1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Inline: Resets the counter for phase 1 for subsequent uses.
                self.count_threads1 = self.num_threads
        
        # Block Logic: Threads wait here until released by the last thread of phase 1.
        self.threads_sem1.acquire()
    
    def phase2(self):
        """
        @brief The second phase of the barrier synchronization, allowing for reuse.
        Threads decrement a second counter; the last thread to decrement releases all others.
        Invariant: All threads are held at `threads_sem2.acquire()` until `count_threads2` reaches zero.
        """
        # Block Logic: Atomically decrements the counter for phase 2.
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Block Logic: The last thread to reach this point releases all waiting threads from phase 2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Inline: Resets the counter for phase 2 for subsequent uses.
                self.count_threads2 = self.num_threads
        
        # Block Logic: Threads wait here until released by the last thread of phase 2.
        self.threads_sem2.acquire()

class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its local sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread and a shared barrier, leveraging global locks for location-specific data.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing the device's local sensor readings.
        @param supervisor: The supervisor object responsible for managing the overall simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal that a script has been assigned.
        self.scripts = [] # List to store assigned scripts.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        self.thread = DeviceThread(self)
        self.thread.start()

        self.barrier = None # Shared barrier for global time step synchronization, to be set by device 0.

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared `ReusableBarrierSem` for synchronization among all devices.
        Only the device with `device_id == 0` is responsible for initializing the barrier
        and a global dictionary for location locks.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: The device with `device_id == 0` initializes the shared barrier and global lock dictionary.
        # Invariant: A single `ReusableBarrierSem` instance is created and shared across all devices.
        if self.device_id == 0:
            num_threads = len(devices) # `num_threads` is local to this scope.
            
            bar = ReusableBarrierSem(len(devices)) # Creates the shared barrier.
            
            # Block Logic: Distributes the initialized shared barrier to all devices.
            for d in devices:
                d.barrier = bar
            
            # Inline: Initializes a global dictionary to hold location-specific locks.
            # This is a global variable, accessible by all `DeviceThread`s.
            global dic
            dic = {}
                       
        pass

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        Dynamically creates and adds a new `Lock` to the global `dic` for that location
        if one doesn't already exist.
        @param script: The script object to assign.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Block Logic: Signals completion of the timepoint if no script is assigned.
            self.timepoint_done.set()


        # Block Logic: Dynamically creates a lock for the given location if it doesn't exist in the global dictionary `dic`.
        # Invariant: Each unique `location` will have one shared `Lock` object in `dic`.
        if location in dic.keys():
            return
        else:
            dic[location] = Lock()
        

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        `DeviceThread` will acquire the appropriate lock from `dic` before calling this method.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        Note: This method does not acquire any locks directly. It is expected that the calling
        `DeviceThread` will acquire the appropriate lock from `dic` before calling this method.
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
    executing scripts sequentially, and coordinating with other device threads using
    a shared `ReusableBarrierSem` and global location-specific locks.
    Time Complexity: O(T * S * (N * D_access + D_script_run)) where T is the number of timepoints,
    S is the number of scripts per device, N is the number of neighbors, D_access is data access
    time, and D_script_run is script execution time.
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
        2. Synchronizes with all other device threads using the shared `ReusableBarrierSem`.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           proceed, ensuring synchronized advancement of the simulation.
        3. Processes each assigned script: for each script, it acquires the location-specific lock from `dic`,
           collects data from neighbors and itself, runs the script, updates data on neighbors and itself,
           and then releases the lock.
           Invariant: Data access and modification for a given location are protected by its corresponding lock.
        4. Waits for the `timepoint_done` event to be set, indicating that all processing for the current timepoint is complete.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Block Logic: Synchronizes all device threads using the shared barrier at the start of each timepoint.
            self.device.barrier.wait()

            # Block Logic: Processes each script assigned to the device for the current timepoint.
            # Invariant: Each script retrieves data from neighbors and itself, executes, and updates data,
            # all while holding the appropriate location-specific lock.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Block Logic: Acquires the global location-specific lock from `dic` before accessing or modifying data.
                dic[location].acquire()
                
                # Block Logic: Collects data from neighboring devices for the specified location.
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
                
                # Block Logic: Releases the global location-specific lock from `dic` after all data operations for this script are complete.
                dic[location].release()
            
            # Block Logic: Waits for the `timepoint_done` event to be set, indicating that all processing for the current timepoint is complete.
            self.device.timepoint_done.wait()
       