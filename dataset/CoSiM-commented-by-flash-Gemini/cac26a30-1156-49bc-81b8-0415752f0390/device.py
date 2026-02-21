"""
@file device.py
@brief Implements a simulated distributed device with a custom two-phase barrier and a single thread for script execution per device.

This module defines the `ReusableBarrier`, `Device`, and `DeviceThread` classes.
The `Device` class represents a node in a simulated sensing network, managing
local sensor data and coordinating script execution through a dedicated
`DeviceThread`. A custom `ReusableBarrier` provides a two-phase synchronization
mechanism to coordinate all `DeviceThread` instances across the simulation.
A single shared lock is used to protect sensor data during concurrent access
from different devices.

Architecture:
- `ReusableBarrier`: A custom implementation of a reusable barrier that uses a
  two-phase protocol with semaphores to synchronize multiple threads.
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds references to shared
  synchronization objects (global barrier and data lock).
- `DeviceThread`: A dedicated thread per `Device` that executes the simulation
  logic for each timepoint, including neighborhood discovery and script processing.

Patterns:
- Two-Phase Barrier Synchronization: Ensures all threads/devices complete a
  phase of computation before any proceed to the next, preventing race conditions
  and maintaining simulation integrity.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `DeviceThread` acts as a consumer, processing them.
- Shared Locking: A single global lock (`self.device.lock`) is acquired for
  processing each script, protecting access to sensor data and neighborhood
  updates across devices.
"""

from threading import Lock, Semaphore, Event, Thread

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
        self.count_lock = Lock() # Lock to protect access to counters
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase

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
                i = 0
                while i < self.num_threads:
                    threads_sem.release() # Release a permit for each thread
                    i += 1
                count_threads[0] = self.num_threads # Reset counter for reuse
        threads_sem.acquire() # Wait for all threads to reach this point
                              # and for the semaphore to be released by the last thread.


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensing network.

    Manages local sensor data, assigned scripts, and coordinates its operation
    within the simulated environment using a dedicated thread and shared
    synchronization primitives.
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
        self.barrier = None # Reference to the global ReusableBarrier
        self.lock = None # Reference to a shared global lock for data access
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when scripts are assigned
        self.scripts = [] # List to store assigned scripts
        self.timepoint_done = Event() # Event to signal when timepoint script assignment is complete
        self.thread = DeviceThread(self) # Dedicated thread for this device's operations
        self.thread.start() # Starts the device's main thread

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device synchronization for all devices.

        This method ensures that a single `ReusableBarrier` and a global `Lock`
        are initialized and shared among all devices in the simulation. This
        initialization is performed by the designated master device (devices[0]).

        @param devices: A list of all Device instances in the simulation.
        """
        # Block Logic: If the global barrier is not yet initialized (first device's perspective),
        # initialize it and the global lock.
        # Pre-condition: This block ensures a single barrier and lock instance are created and shared.
        if devices[0].barrier is None:
            # Block Logic: Checks if this is the designated master device (device_id matches devices[0].device_id).
            # The master device is responsible for creating the shared synchronization primitives.
            if self.device_id == devices[0].device_id:
                bariera = ReusableBarrier(len(devices)) # Initialize the global barrier
                my_lock = Lock() # Initialize the global lock
                # Block Logic: Propagates the initialized barrier and lock to all devices.
                for device in devices:
                    device.barrier = bariera
                    device.lock = my_lock


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script is provided, it's added to the device's script list, and
        the `script_received` event is set to signal its availability. If
        `script` is None, it signals that the current timepoint's script
        assignments are complete by setting the `timepoint_done` event.

        @param script: The script object to execute, or None to signal timepoint completion.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """
        # Block Logic: Handles script assignment and signals script availability or timepoint completion.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a script has been received
        else:
            self.timepoint_done.set() # Signal that all scripts for this timepoint are assigned

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        This method retrieves data from the device's local sensor_data.
        Note that this method itself does not acquire the global `self.lock`,
        as it's assumed to be handled by the caller (`DeviceThread`).

        @param location: The identifier for the data location.
        @return The sensor data at the specified location, or None if not found.
        """

        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        This method updates data in the device's local sensor_data.
        Note that this method itself does not acquire the global `self.lock`,
        as it's assumed to be handled by the caller (`DeviceThread`).

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

    This thread is responsible for discovering neighbors, waiting for script
    assignments, processing assigned scripts sequentially (under a global lock),
    and synchronizing globally at timepoint boundaries.
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

        Continuously performs neighborhood discovery, waits for timepoint script
        assignments to complete, processes all assigned scripts sequentially
        (acquiring and releasing a global lock for each script), and then
        synchronizes globally using the shared barrier before starting the
        next timepoint.
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

            # Block Logic: Processes all assigned scripts for the current timepoint sequentially.
            # Each script is executed on relevant sensor data, protected by a global lock.
            for (script, location) in self.device.scripts:
                # Block Logic: Acquires a shared global lock to protect concurrent access
                # to sensor data and neighbor information during script execution.
                self.device.lock.acquire()
                script_data = []
                # Block Logic: Gathers relevant sensor data from neighboring devices for script input.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                # Block Logic: Includes the device's own sensor data in the script input.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if there is any data to process.
                if script_data != []:
                    # Functional Utility: Executes the assigned script with the collected data.
                    result = script.run(script_data)

                    # Block Logic: Propagates the script's result back to neighboring devices.
                    for device in neighbours:
                        device.set_data(location, result)

                    # Block Logic: Updates the device's own sensor data with the script's result.
                    self.device.set_data(location, result)
                # Block Logic: Releases the shared global lock after script execution,
                # allowing other devices/threads to proceed.
                self.device.lock.release()

            # Block Logic: Clears the timepoint_done event, resetting it for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been processed.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            self.device.barrier.wait()