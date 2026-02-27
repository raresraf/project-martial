


"""
@file device.py
@brief This module defines a simulated device environment utilizing condition variables for a reusable barrier and fine-grained concurrency for script execution.

@details It includes `ReusableBarrierCond` for thread synchronization, implemented
         using `threading.Condition`. The `Device` class represents individual
         simulation entities, and `DeviceThread` manages its main operational loop.
         `ScriptThread` instances are spawned by `DeviceThread` to execute
         scripts in parallel, employing global `Lock` objects for data consistency.
         Shared resources like the barrier and locks are managed as class-level attributes
         of the `Device` class.
"""

from threading import Condition, Lock, Event, Thread

class ReusableBarrierCond(object):
    """
    @brief Implements a reusable barrier for synchronizing multiple threads using `threading.Condition`.

    @details This barrier allows a fixed number of threads to wait until all
             threads have reached a common synchronization point, and then releases
             them all simultaneously. It is designed to be reusable for subsequent
             synchronization points.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrierCond instance.

        @param num_threads The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads      # Total number of threads to wait for.
        self.count_threads = self.num_threads # Counter for threads currently waiting at the barrier.
        self.cond = Condition()             # The condition variable used for signaling and waiting.

    def wait(self):
        """
        @brief Blocks the calling thread until all threads have reached the barrier.

        @details When a thread calls `wait()`, it acquires the condition variable's
                 lock, decrements a counter. If it's the last thread, it notifies
                 all other waiting threads and resets the counter. Otherwise, it
                 releases the lock and waits until it's notified.
        """
        
        self.cond.acquire()     # Acquire the lock associated with the condition variable.
        self.count_threads -= 1 # Decrement the count of threads yet to reach the barrier.
        if self.count_threads == 0:     # If this is the last thread to arrive at the barrier.
            self.cond.notify_all()      # Release all waiting threads.
            self.count_threads = self.num_threads # Reset the counter for the next use of the barrier.
        else:       # If not the last thread.
            self.cond.wait()    # Release the lock and wait for a notification.
        self.cond.release()     # Release the lock after being notified and proceeding.


class Device(object):
    """
    @brief Represents a single simulated device in the environment.

    @details Each device manages its own sensor data, interacts with a central supervisor,
             and executes assigned scripts. It coordinates its activities through a dedicated
             `DeviceThread` and utilizes a shared `ReusableBarrierCond` for global
             synchronization, and a global list of `Lock` objects for data consistency.
    """
    
    barrier = None      # Shared instance of ReusableBarrierCond for global synchronization.
    unique = []         # Global list of Locks, where each Lock protects a specific data location.

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
        self.scripts = []           # List to store (script, location) tuples assigned to this device.
        self.timepoint_done = Event() # Event to signal when all scripts for a timepoint are assigned.

        self.thread = DeviceThread(self) # The dedicated thread for this device's main loop.
        self.thread.start() # Start the main device thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d" % self.device_id.
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization resources (barrier and location locks).

        @details This method is called once at the beginning of the simulation.
                 If the shared `Device.barrier` is not yet initialized, this method
                 creates it. It also initializes a global list of `Lock` objects
                 (`Device.unique`) for a fixed number of data locations, making
                 them accessible to all devices.
        @param devices A list of all Device objects in the simulation.
        """
        
        # Block Logic: Initialize the global barrier if it hasn't been already.
        # Invariant: Device.barrier is a singleton, initialized once.
        if Device.barrier == None:
            Device.barrier = ReusableBarrierCond(len(devices))

        # Block Logic: Initialize a global list of locks for data locations.
        # This list's size is determined by the total number of locations in the test case.
        if len(Device.unique) != \
        self.supervisor.supervisor.testcase.num_locations: # Check if locks have already been created.
            for _ in \
            range(self.supervisor.supervisor.testcase.num_locations):
                Device.unique.append(Lock()) # Create a new lock for each location.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific data location.

        @details If a script is provided, it's added to the device's script queue.
                 If `script` is None, it signals the `timepoint_done` event,
                 indicating that all scripts for the current timepoint have been
                 assigned to this device.
        @param script The script object to assign, or None.
        @param location The data location (e.g., sensor ID) where the script will operate.
        """
        
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its target location.
        else:
            self.timepoint_done.set() # Signal that all script assignments for this timepoint are done.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location from the device's internal state.

        @param location The data location (e.g., sensor ID).
        @return The sensor data for the location, or None if the location is not found.
        """
        
        return self.sensor_data[location] if location in self.sensor_data \
                                          else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location in the device's internal state.

        @param location The data location (e.g., sensor ID).
        @param data The new sensor data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's dedicated thread.

        @details Joins the `DeviceThread`, ensuring all ongoing operations are completed
                 before the device fully shuts down.
        """
        
        self.thread.join() # Wait for the main device thread to finish its execution.


class ScriptThread(Thread):
    """
    @brief An auxiliary thread responsible for executing a single script within a device's context.

    @details This thread encapsulates the logic for running a specific script at
             a given data `location`. It coordinates with other threads by acquiring
             a global lock for the `location` to ensure data consistency, gathers
             data from neighboring devices, executes the script, and propagates the results.
    """

    def __init__(self, device, scripts, neighbours):
        """
        @brief Initializes a new ScriptThread instance.

        @param device The parent `Device` object to which this script belongs.
        @param scripts A list of (script, location) tuples to be executed by this thread.
                       (Note: The current implementation implies it expects a list of one script).
        @param neighbours A list of neighboring `Device` objects for data interaction.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for the ScriptThread.

        @details This method iterates through its assigned scripts. For each script,
                 it acquires a global lock for the data `location`, collects sensor
                 data from the current device and its neighbors, executes the assigned
                 script with this data, and then updates the sensor data on the current
                 device and its neighbors with the script's result. Finally, it releases
                 the location lock.
        """
        
        for (script, location) in self.scripts: # Iterate through scripts (typically one per thread in this design).
            Device.unique[location].acquire() # Acquire the global lock for the specific data location.

            script_data = [] # List to accumulate data for the script.
            
            # Block Logic: Gather data from neighboring devices for the current location.
            if not self.neighbours is None: # Ensure neighbors list is not None.
                for device in self.neighbours:
                    # Note: Neighbors' locks are acquired here, which could lead to deadlock
                    # if another thread tries to acquire the current device's lock
                    # while holding a neighbor's lock, and vice-versa.
                    device.locks[location].acquire() # Acquire lock for neighbor's data location.
                    data = device.get_data(location)
                    device.locks[location].release() # Release lock.

                    if data is not None:
                        script_data.append(data)

            # Block Logic: Gather data from the current device for the current location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Precondition: Execute the script only if there is data available.
            if script_data != []:
                # Execute the assigned script with the collected data.
                result = script.run(script_data)

                # Block Logic: Update sensor data on neighboring devices, acquiring locks.
                if not self.neighbours is None:
                    for device in self.neighbours:
                        device.set_data(location, result)

                # Block Logic: Update sensor data on the current device.
                self.device.set_data(location, result)

            Device.unique[location].release() # Release the global lock for the data location.


class DeviceThread(Thread):
    """
    @brief Manages the main operational loop for a Device object, coordinating script execution.

    @details This thread is responsible for the overall lifecycle of a device's operations
             within a timepoint. It fetches neighbor information from the supervisor,
             dispatches script execution to `ScriptThread`s, and handles global
             synchronization using the shared `ReusableBarrierCond`.
    """
    
    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device The `Device` object that this thread is responsible for.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device # Reference to the parent Device object.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        @details This loop continuously processes timepoints in the simulation.
                 It fetches neighbor information from the supervisor. If the
                 simulation is ongoing, it waits for scripts to be assigned,
                 creates and joins `ScriptThread`s for parallel script execution.
                 Finally, it synchronizes globally with other devices using the
                 `ReusableBarrierCond`. The loop terminates when the supervisor
                 signals the end of the simulation.
        """
        while True:
            # Block Logic: Query the supervisor for updated neighbor information for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            # Precondition: If supervisor returns None, it signals the end of the simulation.
            if neighbours is None:
                break # Exit the main loop.

            # Block Logic: Wait for the Device to signal that all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Create and start ScriptThreads for each assigned script to run in parallel.
            threads = []
            divide_scripts = [[] for _ in range(len(self.device.scripts))] # Initialize a list of lists for script distribution.

            # Note: This distribution logic (`divide_scripts[i].append(self.device.scripts[i])`)
            # effectively creates a separate list containing a single script for each ScriptThread,
            # indicating that each ScriptThread processes one script.
            for i in range(len(self.device.scripts)):
                divide_scripts[i].append(self.device.scripts[i])

            # Block Logic: Create and start a ScriptThread for each individual script.
            for i in range(len(divide_scripts)):
                threads.append(ScriptThread(self.device, divide_scripts[i], \
                                            neighbours))

            for thread in threads:
                thread.start() # Start each ScriptThread.

            for thread in threads:
                thread.join() # Wait for all ScriptThreads to complete their execution.

            Device.barrier.wait() # Global synchronization barrier.
            self.device.timepoint_done.clear() # Reset the event for the next timepoint.
