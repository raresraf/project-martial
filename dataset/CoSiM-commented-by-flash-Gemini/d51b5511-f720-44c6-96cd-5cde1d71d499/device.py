"""
@file device.py
@brief Implements a simulated distributed device using a custom two-phase barrier and a single thread for script execution per device with global locking.

This module defines the `ReusableBarrierSem`, `Device`, and `DeviceThread` classes.
The `Device` class represents a node in a simulated sensing network, managing
local sensor data and coordinating script execution through a dedicated
`DeviceThread`. A custom `ReusableBarrierSem` provides a two-phase synchronization
mechanism to coordinate all `DeviceThread` instances across the simulation.
A global dictionary (`dic`) of `Lock` objects is used to protect sensor data
during concurrent access from different devices at specific locations.

Architecture:
- `ReusableBarrierSem`: A custom implementation of a reusable barrier that uses a
  two-phase protocol with semaphores to synchronize multiple threads/devices.
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds a reference to the global barrier.
- `DeviceThread`: A dedicated thread per `Device` that executes the simulation
  logic for each timepoint, including neighborhood discovery, waiting on
  synchronization barriers, and processing assigned scripts sequentially.

Patterns:
- Two-Phase Barrier Synchronization: Ensures all threads/devices complete a
  phase of computation before any proceed to the next, preventing race conditions
  and maintaining simulation integrity.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `DeviceThread` acts as a consumer, processing them.
- Global Fine-grained Locking: A global dictionary of `Lock` objects (`dic`) ensures
  exclusive access to sensor data at specific locations during script execution
  across concurrent worker threads from different devices.
"""

import sys

from threading import *


class ReusableBarrierSem():
    """
    @brief A reusable two-phase barrier for thread synchronization using semaphores.

    This barrier ensures that all participating threads complete two distinct
    phases of execution before any thread proceeds to the next iteration.
    It uses semaphores and a counter to manage synchronization.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrierSem instance.

        @param num_threads: The total number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for the first phase
        self.count_threads2 = self.num_threads # Counter for the second phase
        self.counter_lock = Lock() # Lock to protect access to counters
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
            # release all threads waiting on threads_sem1 and reset the counter for phase 1 for reuse.
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

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
            # release all threads waiting on threads_sem2 and reset the counter for phase 2 for reuse.
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire() # Wait for all threads to reach this point

class Device(object):
    """
    @brief Represents a simulated device in a distributed sensing network.

    Manages local sensor data, processes assigned scripts sequentially,
    and participates in global synchronization with other devices.
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
        self.script_received = Event() # Event to signal that scripts are ready to be processed
        self.scripts = [] # List to store assigned scripts
        self.timepoint_done = Event() # Event to signal that timepoint script assignment is complete
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
        @brief Sets up shared synchronization resources among devices.

        If this device is the first (device_id 0), it initializes the global
        `ReusableBarrierSem` and a global dictionary (`dic`) of `Lock` objects.
        These shared instances are then propagated to all other devices.

        @param devices: A list of all Device instances in the simulation.
        """

        # Block Logic: Initializes the global barrier and the dictionary of locks if this is device 0.
        # This ensures a single set of shared resources is created and distributed.
        if self.device_id == 0:
            num_threads = len(devices)

            bar = ReusableBarrierSem(len(devices)) # Create a single global barrier

            # Block Logic: Propagates the initialized barrier to all devices.
            for d in devices:
                d.barrier = bar

            # Block Logic: Initializes a global dictionary to store locks for data locations.
            global dic # Declares 'dic' as a global variable
            dic = {}

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script object is provided, it is added to the device's list of scripts,
        and the `script_received` event is set. If `script` is None, it signals that
        script assignments for the current timepoint are complete by setting the
        `timepoint_done` event.
        It also ensures a lock exists in the global `dic` for the given location.

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


        # Block Logic: Ensures that a lock exists in the global dictionary `dic` for the given location.
        # If the lock does not exist, it creates a new Lock and adds it.
        if location in dic.keys():
            return

        else:
            dic[location] = Lock()


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

    This thread is responsible for discovering neighbors, waiting for script
    assignments, processing assigned scripts sequentially (under global locks),
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

        Continuously performs neighborhood discovery, synchronizes with other
        devices using the global barrier, processes all assigned scripts sequentially
        (acquiring and releasing location-specific locks for each script from the
        global `dic`), and then waits for the `timepoint_done` event before looping.
        """
        while True:
            # Block Logic: Discovers neighboring devices for the current timepoint.
            # Pre-condition: `self.device.supervisor` is available to provide neighborhood information.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Checks for a shutdown condition (None neighbors indicates termination).
            if neighbours is None:
                break

            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their neighborhood discovery before proceeding.
            self.device.barrier.wait()

            # Block Logic: Processes all assigned scripts for the current timepoint sequentially.
            # Each script is executed on relevant sensor data, protected by its location-specific lock.
            for (script, location) in self.device.scripts:
                script_data = []

                # Block Logic: Acquires the location-specific lock from the global `dic`
                # to ensure exclusive access to the sensor data at this `location` during script execution.
                dic[location].acquire()

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

                # Block Logic: Releases the location-specific lock, allowing other threads to access it.
                dic[location].release()

            # Block Logic: Waits for the `timepoint_done` event to be set, likely by the supervisor
            # indicating the completion of all script assignments for the current timepoint.
            # This ensures that the device pauses until the timepoint is explicitly advanced.
            self.device.timepoint_done.wait()