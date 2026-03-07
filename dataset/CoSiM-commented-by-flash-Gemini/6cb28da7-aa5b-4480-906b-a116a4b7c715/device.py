"""
@file device.py
@brief Implements core components for a distributed system, likely a simulation or sensor network.
This module defines Device objects that can communicate and process data concurrently
using multiple threads per device, employing synchronization primitives like locks,
conditions, and barriers for coordinated operation and data consistency.
"""

from threading import Event, Thread, Condition, Lock


class Barrier(object):
    """
    @brief Implements a simple, single-use barrier for thread synchronization.
    All participating threads must reach the barrier before any can proceed.
    This version is reset after all threads pass, but its reusability is implicit
    and depends on external management of its `count_threads`.
    """

    def __init__(self, num_threads=0):
        """
        @brief Initializes the barrier with a specified number of threads.

        @param num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads  # Counter for threads reaching the barrier.
        
        self.cond = Condition()  # Condition variable to signal and wait for threads.

    def wait(self):
        """
        @brief Blocks the calling thread until all 'num_threads' have reached this barrier.
        When the last thread arrives, all waiting threads are released.
        """
        self.cond.acquire()  # Acquires the lock associated with the condition variable.
        self.count_threads -= 1  # Decrements the count of threads yet to reach.
        # Conditional Logic: If this is the last thread to reach the barrier.
        if self.count_threads == 0:
            
            self.cond.notify_all()  # Notifies all waiting threads to proceed.
            self.count_threads = self.num_threads  # Resets the counter for future use.
        else:
            
            self.cond.wait()  # Waits (releases lock and blocks) until notified.
        
        self.cond.release()  # Releases the lock.

class Device(object):
    """
    @brief Represents a single device in the distributed system.
    Each device has a unique ID, manages its sensor data, interacts with a supervisor,
    and processes assigned scripts using a pool of dedicated threads. It uses global
    and local synchronization mechanisms.
    """
    
    # Static class variable: A global barrier for all devices to synchronize.
    bariera_devices = Barrier()
    # Static class variable: A global list of locks, typically one for each data location.
    locks = []

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

        self.scripts = []  # List to hold scripts assigned to this device.
        self.locations = [] # List to hold locations corresponding to the scripts.
        
        self.nr_scripturi = 0  # Counter for the total number of scripts assigned for the current timepoint.
        
        self.script_crt = 0  # Index of the current script being processed by a DeviceThread.

        self.timepoint_done = Event()  # Event to signal when script assignment for a timepoint is complete.

        self.neighbours = []  # List of neighboring Device objects.
        self.event_neighbours = Event()  # Event to signal when neighbors are updated.
        self.lock_script = Lock()  # Lock to protect access to self.script_crt and self.scripts.
        self.bar_thr = Barrier(8)  # A local barrier for the 8 DeviceThread instances within this Device.

        # Creates a pool of 8 DeviceThread instances for this Device.
        # One thread (first=1) has special responsibility (e.g., fetching neighbors).
        self.thread = DeviceThread(self, 1) # The main DeviceThread.
        self.thread.start()
        self.threads = []
        for _ in range(7): # Additional 7 DeviceThreads.
            tthread = DeviceThread(self, 0)
            self.threads.append(tthread)
            tthread.start()

    def __str__(self):
        """
        @brief Provides a string representation of the Device.

        @return str: A formatted string indicating the device ID.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs global setup for devices, including initializing the static barrier and locks.
        This method is typically called once by a coordinating entity.

        @param devices (list): A list of all Device instances in the system.
        """
        # Initializes the static global barrier with the total number of devices.
        Device.bariera_devices = Barrier(len(devices))
        
        # Conditional Logic: Initializes the static list of locks only if it's empty.
        # This ensures locks are created once for all locations across all devices.
        if Device.locks == []:
            # The number of locations is obtained from the supervisor's testcase configuration.
            for _ in range(self.supervisor.supervisor.testcase.num_locations):
                Device.locks.append(Lock())

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        @param script (callable): The script (function or object with a run method) to execute.
                                  If None, it signals that script assignment for the timepoint is done.
        @param location (int): The identifier of the data location the script operates on.
        """
        # Conditional Logic: If a script is provided, it's added to the lists.
        if script is not None:
            self.scripts.append(script)
            self.locations.append(location)
            
            self.nr_scripturi += 1 # Increments the count of assigned scripts.
        else:
            # If script is None, it means no more scripts for the current timepoint.
            # Signals that script assignment for the timepoint is complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location (int): The identifier of the data location.
        @return any: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in \
        self.sensor_data else None

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
        @brief Shuts down all DeviceThread instances associated with this device.
        """
        self.thread.join() # Waits for the main DeviceThread to finish.
        for tthread in self.threads: # Waits for all other DeviceThreads to finish.
            tthread.join()


class DeviceThread(Thread):
    """
    @brief A thread of execution dedicated to a Device. Multiple instances of this
    thread run concurrently for a single Device, cooperating to process scripts.
    One instance (`first=1`) has additional responsibilities, like fetching neighbors.
    """

    def __init__(self, device, first):
        """
        @brief Initializes a new DeviceThread instance.

        @param device (Device): The Device object this thread is responsible for.
        @param first (int): A flag (1 for main thread, 0 for auxiliary) indicating
                             special responsibilities for this thread instance.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.first = first # Flag for special thread responsibilities.

    def run(self):
        """
        @brief The main execution loop of a DeviceThread.
        It participates in fetching neighbors (if `first=1`), waits for assigned scripts,
        processes a portion of them, and synchronizes with other threads within the same
        Device and with other Devices via global barriers.
        """
        while True:
            # Block Logic: The main DeviceThread (first=1) fetches neighbors and resets script index.
            # Invariant: Each iteration processes a timepoint.
            
            if self.first == 1:
                # Fetches neighbor devices from the supervisor.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.script_crt = 0  # Resets the script index for the new timepoint.
                self.device.event_neighbours.set() # Signals that neighbors information is updated.

            # Synchronization: All DeviceThreads wait until neighbor information is available.
            self.device.event_neighbours.wait()

            # Conditional Logic: If supervisor signals shutdown (neighbors is None), terminates the thread.
            if self.device.neighbours is None:
                break

            # Synchronization: All DeviceThreads wait until all scripts for the current timepoint are assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Each DeviceThread concurrently picks and processes scripts from the Device's queue.
            while True:
                # Synchronization: Acquires a lock to safely get the next script index.
                self.device.lock_script.acquire()
                index = self.device.script_crt # Gets the current script index.
                self.device.script_crt += 1    # Increments the index for the next thread.
                self.device.lock_script.release() # Releases the lock.

                # Conditional Logic: If all scripts have been processed by this Device's threads, breaks the loop.
                if index >= self.device.nr_scripturi:
                    break

                # Retrieves the script and its location based on the index.
                location = self.device.locations[index]
                script = self.device.scripts[index]

                # Synchronization: Acquires the global lock for the specific data location.
                Device.locks[location].acquire()

                script_data = [] # List to collect data for the current script.
                    
                # Block Logic: Collects data from neighboring devices.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Collects data from its own device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Conditional Logic: If any data was collected, executes the script and propagates results.
                if script_data != []:
                        
                    result = script.run(script_data) # Executes the script.

                    # Block Logic: Propagates the new data to all neighboring devices.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    # Updates the data on its own device.
                    self.device.set_data(location, result)

                # Releases the global lock for the specific data location.
                Device.locks[location].release()

            # Synchronization: All DeviceThreads within this Device wait at a local barrier.
            self.device.bar_thr.wait()
            
            # Block Logic: The main DeviceThread (first=1) resets event flags for the next timepoint.
            if self.first == 1:
                self.device.event_neighbours.clear() # Clears the event for neighbors.
                self.device.timepoint_done.clear()  # Clears the timepoint done event.
                self.device.scripts = []            # Clears the script list.
                self.device.locations = []          # Clears the locations list.
                self.device.nr_scripturi = 0        # Resets the script counter.
            
            self.device.bar_thr.wait() # All DeviceThreads wait at the local barrier again.
            
            # Conditional Logic: The main DeviceThread (first=1) waits at the global barrier.
            if self.first == 1:
                Device.bariera_devices.wait() # Global synchronization among all Devices.

