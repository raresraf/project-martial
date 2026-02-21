"""
@file device.py
@brief Implements a simulated distributed device with a custom two-phase barrier and multi-threaded script execution.

This module defines the `ReusableBarrier`, `MyThread`, `Device`, and `DeviceThread` classes.
The `Device` class represents a node in a simulated sensing network, managing
local sensor data and coordinating script execution. `DeviceThread` orchestrates
the timepoint simulation, while `MyThread` instances are spawned to concurrently
execute individual scripts. A shared `ReusableBarrier` ensures global timepoint
synchronization.

Architecture:
- `ReusableBarrier`: A custom implementation of a reusable barrier that uses a
  two-phase protocol with semaphores to synchronize multiple threads.
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds a reference to the global barrier.
- `DeviceThread`: A dedicated thread per `Device` that orchestrates the
  execution for each timepoint, including neighborhood discovery, spawning
  `MyThread`s, and global timepoint synchronization.
- `MyThread`: Worker threads spawned by `DeviceThread` to execute
  individual scripts. Each script thread gathers data from neighbors and itself,
  runs the script, and updates data.

Patterns:
- Two-Phase Barrier Synchronization: Ensures all threads/devices complete a
  phase of computation before any proceed to the next, preventing race conditions
  and maintaining simulation integrity.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `MyThread` instances (spawned by `DeviceThread`) act as consumers,
  processing them.
- Concurrent Script Execution: Scripts are executed in parallel by multiple
  `MyThread` instances within each device.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    @brief A reusable two-phase barrier for thread synchronization.

    This barrier ensures that all participating threads complete two distinct
    phases of execution before any thread proceeds to the next iteration.
    It uses semaphores and a counter to manage synchronization.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier instance.

        @param num_threads: The total number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Counter for the first phase, wrapped in list for pass-by-reference
        self.count_threads2 = [self.num_threads] # Counter for the second phase
        self.count_lock = Lock()           	# Lock to protect access to counters
        self.threads_sem1 = Semaphore(0)        # Semaphore for the first phase
        self.threads_sem2 = Semaphore(0)        # Semaphore for the second phase

    def wait(self):
        """
        @brief Blocks until all threads have completed both phases of the barrier.

        This method orchestrates the two phases of synchronization.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements one phase of the barrier synchronization.

        A thread decrements a shared counter. The last thread to make the counter
        zero releases all waiting threads via a semaphore and resets the counter.

        @param count_threads: The counter (list containing an integer) for the current phase.
        @param threads_sem: The semaphore used to release waiting threads for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # Block Logic: If this is the last thread to reach the barrier in this phase,
            # release all threads waiting on the semaphore and reset the counter.
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release() # Release a permit for each thread

                count_threads[0] = self.num_threads # Reset counter for reuse
        threads_sem.acquire() # Wait for all threads to reach this point and for the semaphore to be released.

class MyThread(Thread):
    """
    @brief A worker thread responsible for executing a single script.

    Each `MyThread` processes one assigned script, gathers necessary
    sensor data from the local device and its neighbors, executes the script,
    and updates the results back to the relevant devices.
    """
    def __init__(self, neighbours, script, location, device):
        """
        @brief Initializes a new MyThread instance.

        @param neighbours: A list of neighboring `Device` instances.
        @param script: The script object to execute.
        @param location: The data location (e.g., sensor ID) for the script.
        @param device: A reference to the parent `Device` instance.
        """
        Thread.__init__(self)
        self.neighbours = neighbours # List of neighboring devices

        self.script = script # The script to execute
        self.location = location # The sensor data location for the script
        self.device = device # Reference to the parent device
        self.script_data = [] # Buffer to hold collected sensor data

    def run(self):
        """
        @brief The main execution logic for the `MyThread`.

        Gathers data from neighbors and the local device, executes the script,
        and updates the sensor data on affected devices.
        """
        # Block Logic: Gathers relevant sensor data from neighboring devices for script input.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                self.script_data.append(data)

        # Block Logic: Includes the current device's own sensor data in the script input.
        data = self.device.get_data(self.location)
        if data is not None:
            self.script_data.append(data)
        # Block Logic: Executes the script only if there is any data to process.
        if self.script_data != []:

            # Functional Utility: Executes the assigned script with the collected data.
            result = self.script.run(self.script_data)

            # Block Logic: Propagates the script's result back to neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)

            # Block Logic: Updates the current device's own sensor data with the script's result.
            self.device.set_data(self.location, result)
        self.script_data = [] # Clears the buffer for the next execution


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensing network.

    Manages local sensor data, assigned scripts, and coordinates its operation
    within the simulated environment using a dedicated thread and a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing initial sensor readings
                            for various locations.
        @param supervisor: A reference to the central supervisor managing
                           the distributed system.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when scripts are assigned
        self.scripts = [] # List to store assigned scripts
        self.timepoint_done = Event() # Event to signal that timepoint script assignment is complete
        self.thread = DeviceThread(self) # Dedicated thread for this device's operations
        self.thread.start() # Starts the device's main thread
        self.barrier = None # Reference to the global ReusableBarrier

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization resources among devices.

        If this device is the first device (device_id 0), it initializes the
        global `ReusableBarrier` for all devices. It then propagates this
        shared barrier reference to all other devices.

        @param devices: A list of all Device instances in the simulation.
        """
        # Block Logic: Initializes the global barrier if this is the designated master device (device_id 0).
        # Pre-condition: This block ensures a single barrier instance is created and shared.
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices)) # Create a new ReusableBarrier
            self.barrier = barrier # Store local reference
            # Block Logic: Propagates the initialized barrier to all devices in the simulation.
            for i in xrange(len(devices)):
                devices[i].barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script object is provided, it is added to the device's list of scripts
        to be processed, and `script_received` event is set. If `script` is None,
        it signals that all scripts for the current timepoint have been assigned
        by setting the `timepoint_done` event.

        @param script: The script object to execute, or None to signal timepoint completion.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """
        # Block Logic: Differentiates between assigning a script and signaling end-of-assignment.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a script has been received
        else:
            self.timepoint_done.set() # Signal that all scripts for this timepoint are assigned

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The identifier for the data location.
        @return The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location: The identifier for the data location.
        @param data: The new data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its main thread to complete.

        Functional Utility: Ensures proper termination and cleanup of resources
        associated with the device's dedicated execution thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief A dedicated thread for each Device, managing its simulation logic.

    This thread is responsible for discovering neighbors, spawning `MyThread`s
    for concurrent script execution, and synchronizing with other devices
    at timepoint boundaries using a shared barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: A reference to the parent Device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Continuously performs neighborhood discovery, waits for script assignments
        to complete for the current timepoint, spawns `MyThread`s to process scripts
        concurrently, waits for all script threads to finish, and then synchronizes
        globally using the shared barrier before starting the next timepoint.
        """
        while True:
            # Block Logic: Discovers neighboring devices for the current timepoint.
            # Pre-condition: `self.device.supervisor` is available to provide neighborhood information.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Checks for a shutdown condition (None neighbors indicates termination).
            if neighbours is None:
                break

            # Block Logic: Waits for the supervisor to signal that script assignments for
            # the current timepoint are complete.
            # Invariant: All scripts for the current timepoint are in `self.device.scripts` after this wait.
            self.device.timepoint_done.wait()

            # Block Logic: Spawns a `MyThread` for each assigned script to execute them concurrently.
            freds = [] # List to hold MyThread instances
            for (script, location) in self.device.scripts:
                fred = MyThread(neighbours, script, location, self.device)
                freds.append(fred)

            # Block Logic: Starts all `MyThread` instances.
            for i in freds:
                i.start()
            # Block Logic: Waits for all `MyThread` instances to complete their execution.
            for i in freds:
                i.join()

            # Block Logic: Clears the timepoint_done event, resetting it for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been processed.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            self.device.barrier.wait()
