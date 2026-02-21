"""
@file device.py
@brief Implements a simulated distributed device using a custom two-phase reusable barrier and a single thread for script execution per device.

This module defines the `Device`, `DeviceThread`, and `MyReusableBarrier` classes.
The `Device` class represents a node in a simulated sensing network, managing
local sensor data and coordinating script execution through a dedicated
`DeviceThread`. A custom `MyReusableBarrier` provides a two-phase synchronization
mechanism to coordinate all `DeviceThread` instances across the simulation.

Architecture:
- `MyReusableBarrier`: A custom implementation of a reusable barrier that uses a
  two-phase protocol with semaphores to synchronize multiple threads.
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds a reference to the global barrier.
- `DeviceThread`: A dedicated thread per `Device` that executes the simulation
  logic for each timepoint, including neighborhood discovery, script processing,
  and global timepoint synchronization.

Patterns:
- Two-Phase Barrier Synchronization: Ensures all threads/devices complete a
  phase of computation before any proceed to the next, preventing race conditions
  and maintaining simulation integrity.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `DeviceThread` acts as a consumer, processing them.
"""

from threading import Event, Thread, Lock, Semaphore


class MyReusableBarrier():
    """
    @brief A reusable two-phase barrier for thread synchronization.

    This barrier ensures that all participating threads complete two distinct
    phases of execution before any thread proceeds to the next iteration.
    It uses semaphores and a counter to manage synchronization.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the MyReusableBarrier instance.

        @param num_threads: The total number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for the first phase
        self.count_threads2 = self.num_threads # Counter for the second phase

        self.counter_lock = Lock()       # Lock to protect access to counters
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase

    def wait(self):
        """
        @brief Blocks until all threads have completed both phases of the barrier.

        This method orchestrates the two phases of synchronization.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief First phase of the two-phase barrier synchronization.

        Threads decrement a counter. The last thread to reach zero
        releases all other threads waiting on `threads_sem1`.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            # Block Logic: If this is the last thread to reach the barrier in phase 1,
            # release all threads waiting on threads_sem1 and reset the counter for phase 2.
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            self.count_threads2 = self.num_threads # Reset count for phase 2 for the next wait cycle.

        self.threads_sem1.acquire() # Wait for all threads to reach this point

    def phase2(self):
        """
        @brief Second phase of the two-phase barrier synchronization.

        Threads decrement a counter. The last thread to reach zero
        releases all other threads waiting on `threads_sem2`.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            # Block Logic: If this is the last thread to reach the barrier in phase 2,
            # release all threads waiting on threads_sem2 and reset the counter for phase 1.
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            self.count_threads1 = self.num_threads # Reset count for phase 1 for the next wait cycle.

        self.threads_sem2.acquire() # Wait for all threads to reach this point


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
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when scripts are assigned
        self.scripts = [] # List to store assigned scripts
        self.timepoint_done = Event() # Event to signal that timepoint script assignment is complete
        self.thread = DeviceThread(self) # Dedicated thread for this device's operations
        self.thread.start() # Starts the device's main thread
        self.barrier = None # Reference to the global MyReusableBarrier
        self.lock = Lock() # Local lock for this device's sensor_data

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization resources among devices.

        If this device is the first (devices[0]), it initializes the global
        `MyReusableBarrier` for all devices. Other devices obtain a reference
        to this shared barrier. It also stores a reference to all devices.

        @param devices: A list of all Device instances in the simulation.
        """

        self.devices=devices
        # Block Logic: Initializes the global barrier instance if this is device[0].
        # This ensures a single barrier is shared across all devices.
        if self == devices[0]:
            self.bar = MyReusableBarrier(len(devices)) # Create a new MyReusableBarrier


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


class MyThread(Thread):
    """
    @brief A worker thread responsible for executing a single script on sensor data.

    Each `MyThread` processes one assigned script, gathers necessary
    sensor data from the local device and its neighbors, executes the script,
    and updates the results back to the relevant devices, acquiring and
    releasing locks on each device for data consistency.
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
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.device = device

    def run(self):
        """
        @brief The main execution logic for the `MyThread`.

        Gathers data from neighbors and the local device, executes the script,
        and updates the sensor data on affected devices, using per-device locks
        to ensure data integrity during updates.
        """
        script_data = []
        # Block Logic: Gathers relevant sensor data from neighboring devices for script input.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Block Logic: Includes the current device's own sensor data in the script input.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)


        # Block Logic: Executes the script only if there is any data to process.
        if script_data != []:

            # Functional Utility: Executes the assigned script with the collected data.
            result = self.script.run(script_data)
            # Block Logic: Propagates the script's result back to neighboring devices,
            # acquiring and releasing their individual locks for thread-safe updates.
            for device in self.neighbours:
                device.lock.acquire()
                device.set_data(self.location, result)
                device.lock.release()

            # Block Logic: Updates the current device's own sensor data with the script's result,
            # acquiring and releasing its local lock for thread-safe updates.
            self.device.lock.acquire()
            self.device.set_data(self.location, result)
            self.device.lock.release()

class DeviceThread(Thread):
    """
    @brief A dedicated thread for each Device, managing its simulation logic.

    This thread is responsible for discovering neighbors, waiting for script
    assignments, spawning `MyThread`s for concurrent script execution,
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
        assignments to complete, spawns `MyThread`s to process scripts concurrently,
        waits for all script threads to finish, and then synchronizes globally
        using the shared barrier before starting the next timepoint.
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


            # Block Logic: Clears the `timepoint_done` event, resetting it for the next timepoint.
            # This is done *before* starting threads to ensure it's clear for the next iteration.
            self.device.timepoint_done.clear()

            threads = []
            # Block Logic: Spawns a `MyThread` for each assigned script to execute them concurrently.
            # Pre-condition: `self.device.scripts` contains all scripts to be processed in this timepoint.
            for (script, location) in self.device.scripts:
                t = MyThread(neighbours, self.device, location, script)
                t.start()
                threads.append(t)

            # Block Logic: Waits for all `MyThread` instances to complete their execution.
            for i in range(len(threads)):
                threads[i].join()

            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            self.device.devices[0].bar.wait() # Accesses the barrier from devices[0]