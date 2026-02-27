


"""
@file device.py
@brief This module defines the behavior and synchronization mechanisms for individual devices in a distributed simulation environment.

@details It includes classes for managing device state, handling sensor data,
         executing assigned scripts, and synchronizing operations across multiple
         devices using barriers and locks. The `Device` class represents an
         individual simulated entity, while `DeviceThread` encapsulates its
         concurrent execution logic. `ReusableBarrier` provides a synchronization
         primitive for coordinating multiple threads/devices.
"""

from threading import Event, Thread, Lock
from utils import ReusableBarrier


class Device(object):
    """
    @brief Represents a single device within the simulation environment.

    @details Manages the device's state, sensor data, communication with a supervisor,
             and execution of assigned scripts. Each device runs in its own thread
             (`DeviceThread`) and coordinates with other devices using shared
             synchronization primitives like barriers and locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id An integer representing the unique identifier for the device.
        @param sensor_data A dictionary containing initial sensor data for the device.
        @param supervisor An object responsible for overseeing and coordinating devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event() # Event to signal completion of a timepoint's tasks.
        self.thread = DeviceThread(self, 0) # The dedicated thread for this device.

        
        self.common_barrier = None # Barrier for synchronizing all devices.
        
        # Event to signal when the device's initialization is complete, especially for non-leader devices.
        self.wait_initialization = Event()

        
        self.locations_locks = None # Shared dictionary of locks for data locations.
        
        self.lock_location_dict = Lock() # Lock for protecting access to `locations_locks`.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the synchronization mechanisms for the devices in the simulation.

        @details This method initializes the common barrier and shared location locks
                 among all devices. Device 0 (the leader) is responsible for creating
                 these shared resources and signaling other devices to start.
        @param devices A list of all Device objects in the simulation.
        """

        if not self.device_id == 0:
            # Block Logic: Non-leader devices wait for the leader to complete initialization.
            self.wait_initialization.wait() # Wait for the leader device to set up shared resources.
            
            self.thread.start() # Start the device's execution thread.
        else:
            # Block Logic: Leader device (device_id == 0) initializes shared resources.

            # Initialize a dictionary to hold locks for different data locations.
            self.locations_locks = {}

            # Create a reusable barrier for synchronizing all devices.
            barrier_size = len(devices)
            self.common_barrier = ReusableBarrier(len(devices))

            # Assign the common barrier and shared locations_locks to all devices.
            for dev in devices:
                dev.common_barrier = self.common_barrier
                dev.locations_locks = self.locations_locks
            
            # Signal non-leader devices that shared resources are initialized.
            for dev in devices:
                if not dev.device_id == 0:
                    dev.wait_initialization.set() # Release non-leader devices to start their threads.

            self.thread.start() # Start the leader device's execution thread.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific data location.

        @details Scripts represent operations to be performed on sensor data. If no script
                 is provided, it signals that the device has no tasks for the current timepoint.
        @param script The script object to assign, or None if no script.
        @param location The data location (e.g., sensor ID) where the script will operate.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its target location.
        else:
            self.timepoint_done.set() # Signal completion if no scripts are assigned.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.

        @param location The data location (e.g., sensor ID).
        @return The sensor data for the location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.

        @param location The data location (e.g., sensor ID).
        @param data The new sensor data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's execution thread.

        @details Joins the device's thread, ensuring all operations are completed
                 before the program exits.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Represents the execution thread for a Device object.

    @details This thread manages the cyclical execution of scripts assigned to its
             associated `Device`. It uses a common barrier to synchronize with
             other device threads at the beginning of each timepoint and processes
             scripts, applying results to sensor data and communicating with neighbors.
    """

    def __init__(self, device, th_id):
        """
        @brief Initializes a new DeviceThread instance.

        @param device The `Device` object that this thread is responsible for.
        @param th_id An integer representing the thread's ID (typically 0 for the main device thread).
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.th_id = th_id

    def run(self):
        """
        @brief The main execution loop for the device thread.

        @details This method continuously loops, synchronizing at a common barrier
                 at the start of each timepoint. It then retrieves scripts, acquires
                 necessary locks for data locations, executes the scripts,
                 updates sensor data, and releases the locks. The loop breaks
                 when the supervisor signals the simulation to end.
        """
        while True:
            # Block Logic: Synchronize all device threads at the common barrier.
            # Invariant: All threads wait here until every device has reached this point.
            self.device.common_barrier.wait()

            # Precondition: Only the leader device (th_id == 0) communicates with the supervisor.
            if self.th_id == 0:
                neighbours = self.device.supervisor.get_neighbours() # Get neighboring devices from the supervisor.
                if neighbours is None:
                    break # Break the loop if the supervisor signals the end of the simulation.
            else:
                # Non-leader threads simply pass, they don't directly interact with the supervisor for neighbors.
                pass

            # Block Logic: Wait for scripts to be assigned and marked as ready for execution.
            # Invariant: All devices wait here until their `timepoint_done` event is set.
            self.device.timepoint_done.wait()

            # Retrieve the current batch of scripts assigned to this device.
            current_scripts = self.device.scripts

            # Block Logic: Iterate through assigned scripts and execute them.
            # Invariant: Each script is executed, acquiring necessary locks to prevent race conditions.
            for (script, location) in current_scripts:
                # Inline: Acquire a lock to safely access and potentially modify the `locations_locks` dictionary.
                self.device.lock_location_dict.acquire()

                # Block Logic: Ensure a lock exists for the current data location, create if not.
                # Invariant: Each data location has a dedicated lock for mutual exclusion.
                if not self.device.locations_locks.has_key(location):
                    self.device.locations_locks[location] = Lock() # Create a new lock for this location.

                # Inline: Acquire the specific lock for the current data location.
                self.device.locations_locks[location].acquire()

                # Inline: Release the lock for the `locations_locks` dictionary after modification/access.
                self.device.lock_location_dict.release()

                script_data = [] # List to accumulate data for the script.
                
                # Block Logic: Gather data from neighboring devices for the current location.
                # Invariant: Only available data from neighbors is collected.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Get data from the current device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Precondition: Execute the script only if there is data available.
                if script_data != []:
                    # Execute the assigned script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Update sensor data on neighboring devices and the current device.
                    # Invariant: The script's result is disseminated to all relevant devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                # Inline: Release the specific lock for the current data location.
                self.device.locations_locks[location].release()

            self.device.timepoint_done.clear() # Reset the event for the next timepoint.


from threading import Semaphore, Lock


class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.

    @details This barrier allows a fixed number of threads to wait until all
             threads have reached a common point, and then releases them all
              simultaneously. It is designed to be reusable for subsequent
             synchronization points.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrier instance.

        @param num_threads The total number of threads that will participate in the barrier.
        """

        self.num_threads = num_threads
        # Two counters are used to manage the two phases of the barrier, allowing reusability.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]

        
        self.count_lock = Lock() # A lock to protect access to the thread counters.

        # Two semaphores, one for each phase, to block and release threads.
        self.threads_sem1 = Semaphore(0)

        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all threads have reached the barrier.

        @details This method orchestrates the two-phase synchronization mechanism.
                 Threads are released only after all `num_threads` have called `wait()`.
        """

        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single phase of the barrier synchronization.

        @details Decrements a counter and, when the counter reaches zero, releases
                 all waiting threads using a semaphore. It then resets the counter
                 for the next phase.
        @param count_threads A list containing the counter for the current phase (list to make it mutable).
        @param threads_sem The semaphore associated with the current phase.
        """

        with self.count_lock: # Ensure atomic access to the counter.
            count_threads[0] -= 1
            if count_threads[0] == 0: # If this is the last thread to reach the barrier.
                # Release all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                
                count_threads[0] = self.num_threads # Reset the counter for reusability.
        
        threads_sem.acquire() # Acquire the semaphore, blocking until released by the last thread.
        
