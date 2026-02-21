"""
@file device.py
@brief Implements a simulated distributed device using an external reusable barrier and multi-threaded script execution with fine-grained locking.

This module defines the `Device`, `DeviceThread`, and `MyThread` classes,
which together simulate a node in a distributed sensing network. Each `Device`
manages its local sensor data and executes assigned scripts. A `DeviceThread`
per device coordinates the overall timepoint simulation, distributing scripts
among multiple `MyThread` instances for concurrent processing. A shared
external `ReusableBarrierCond` ensures global timepoint synchronization,
and a shared dictionary of `Lock` objects provides fine-grained control
over access to sensor data locations.

Architecture:
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds references to the global barrier
  and the shared dictionary of data locks.
- `DeviceThread`: A dedicated thread per `Device` that orchestrates the
  execution for each timepoint, including neighborhood discovery, distributing
  scripts into batches, and spawning `MyThread`s for concurrent execution.
- `MyThread`: Worker threads spawned by `DeviceThread` to execute a subset
  of assigned scripts. Each script within a `MyThread` acquires a lock for its
  target data location, gathers data from neighbors and itself, runs the script,
  updates data, and releases the lock.
- `barrier.ReusableBarrierCond`: An external reusable barrier used for global
  synchronization across all devices at each simulation step.

Patterns:
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `MyThread` instances (spawned by `DeviceThread`) act as consumers,
  processing them.
- Fine-grained Locking: A shared dictionary of `Lock` objects ensures
  exclusive access to sensor data at specific locations during script execution
  across concurrent worker threads.
- Script Distribution: Scripts are divided among a fixed number of worker threads
  to utilize concurrency effectively.
"""

import barrier
from threading import Event, Thread, Lock


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensing network.

    Manages local sensor data, processes assigned scripts concurrently,
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
        self.barrier = None # Reference to the global ReusableBarrierCond
        self.dictionary = {} # Shared dictionary of global locks for specific data locations

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device synchronization mechanisms.

        If this device is the designated master (the device with the largest device_id in this setup),
        it initializes the shared `ReusableBarrierCond` and the global `dictionary` of locks.
        These synchronization primitives are then propagated to all `Device` instances.

        @param devices: A list of all Device instances in the simulation.
        """

        # Block Logic: Initializes the global barrier and the dictionary of locks if this is the master device.
        # This implementation designates the device with the largest ID as the master for setup.
        if self.device_id == len(devices) - 1: # Assuming the last device in the list is the master
            # Functional Utility: Initializes a single global barrier for all devices.
            my_barrier = barrier.ReusableBarrierCond(len(devices))
            my_dictionary = dict()
            # Block Logic: Initializes a lock for each unique sensor data location across all devices.
            for dev in devices:
                for location, data in dev.sensor_data.iteritems(): # Python 2 syntax for iterating dictionary items
                    if location not in my_dictionary:
                        my_dictionary[location] = Lock()
            # Block Logic: Propagates the initialized shared barrier and dictionary of locks to all devices.
            for dev in devices:
                dev.barrier = my_barrier
                dev.dictionary = my_dictionary

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script object is provided, it is added to the device's list of scripts.
        If `script` is None, it signals that script assignments for the current
        timepoint are complete by setting the `script_received` event.

        @param script: The script object to execute, or None to signal script assignment completion.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """
        # Block Logic: Handles script assignment and signals script availability.
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
        return self.sensor_data[location] if location in self.sensor_data \
        else None

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
    assignments, distributing scripts among `MyThread` instances for concurrent
    execution, and synchronizing globally at timepoint boundaries.
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
        assignments to complete, distributes scripts among multiple `MyThread`s
        for processing, waits for all `MyThread`s to finish, and then synchronizes
        globally using the shared barrier before starting the next timepoint.
        """

        while True:
            # Block Logic: Discovers neighboring devices for the current timepoint.
            # Pre-condition: `self.device.supervisor` is available to provide neighborhood information.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Checks for a shutdown condition (None neighbors indicates termination).
            if neighbours is None:
                break
            
            threads = []
            div = len(self.device.scripts) // 8 # Calculate base number of scripts per worker
            mod = len(self.device.scripts) % 8  # Calculate remainder scripts for uneven distribution
            
            # Block Logic: Waits for the supervisor to signal that script assignments for
            # the current timepoint are complete.
            self.device.timepoint_done.wait()

            # Block Logic: Distributes scripts among 8 worker threads and creates `MyThread` instances.
            # This ensures concurrent processing of scripts for the current timepoint.
            for division in range(8): # Loop for 8 worker threads
                if div > 0:
                    # Assign a base number of scripts to the current worker.
                    list_of_scripts = \
                    self.device.scripts[division * div: (division+1) * div]
                else:
                    list_of_scripts = []
                # Distribute remaining scripts one by one to workers.
                if mod > 0:
                    list_of_scripts.append\
                    (self.device.scripts[len(self.device.scripts) - mod])
                    mod = mod - 1
                threads.append(MyThread(self.device, list_of_scripts, neighbours)) # Create a worker thread


            # Block Logic: Starts all worker threads.
            for thread in threads:
                thread.start()

            # Block Logic: Waits for all worker threads to complete their assigned scripts.
            for thread in threads:
                thread.join()

            # Block Logic: Clears the `timepoint_done` event, resetting it for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been processed.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            self.device.barrier.wait()

class MyThread(Thread):
    """
    @brief A worker thread responsible for executing a batch of scripts on sensor data.

    Each `MyThread` processes its assigned scripts, acquiring location-specific
    locks, gathering data from the local device and its neighbors, executing
    the script, and propagating the results back to the relevant devices.
    """

    def __init__(self, device, scripts, neighbours):
        """
        @brief Initializes a new MyThread instance.

        @param device: A reference to the parent `Device` instance.
        @param scripts: A list of (script, location) tuples assigned to this worker.
        @param neighbours: A list of neighboring `Device` instances.
        """
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for the `MyThread`.

        Iterates through its assigned scripts, and for each script: acquires
        the location-specific lock, gathers data from neighbors and the local
        device, executes the script, updates sensor data on affected devices,
        and then releases the lock.
        """

        # Block Logic: Iterates through each script assigned to this worker thread.
        for (script, location) in self.scripts:
            # Block Logic: Acquires the location-specific lock from the shared dictionary
            # to ensure exclusive access to the sensor data at this `location` during script execution.
            self.device.dictionary[location].acquire()
            script_data = []

            # Block Logic: Gathers relevant sensor data from neighboring devices for script input.
            for dev in self.neighbours:
                data = dev.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Block Logic: Includes the current device's own sensor data in the script input.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script only if there is any data to process.
            if script_data != []:
                # Functional Utility: Executes the assigned script with the collected data.
                result = script.run(script_data)

                # Block Logic: Propagates the script's result back to neighboring devices.
                for dev in self.neighbours:
                    dev.set_data(location, result)

                # Block Logic: Updates the current device's own sensor data with the script's result.
                self.device.set_data(location, result)
            # Block Logic: Releases the location-specific lock, allowing other threads to access it.
            self.device.dictionary[location].release()
