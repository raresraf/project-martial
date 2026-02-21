"@file device.py
@brief Implements a simulated distributed device using a reusable barrier and multi-threaded script execution with dynamic thread management and fine-grained locking.

This module defines the `Device` and `DeviceThread` classes, along with a global `execute` function.
Together, they simulate a node in a distributed sensing network. Each `Device` manages its
local sensor data and processes assigned scripts. A `DeviceThread` per device coordinates
the overall timepoint simulation, dynamically spawning worker threads (running `execute`) 
to process scripts concurrently. A shared `ReusableBarrier` ensures global timepoint
synchronization, and a shared list of `Lock` objects provides fine-grained control
over access to sensor data locations.

Architecture:
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds references to the global barrier,
  the shared list of data locks, and a semaphore for thread pool management.
- `DeviceThread`: A dedicated thread per `Device` that orchestrates the
  execution for each timepoint, including neighborhood discovery, and dynamically
  spawning worker threads (`execute`) for concurrent script processing.
- `execute`: A global function executed by worker threads. It acquires a lock
  for its target data location, gathers data from neighbors and itself, runs the script,
  updates data, and releases the lock, then releases a semaphore permit.
- `ReusableBarrier`: A shared barrier for global synchronization across all devices
  at each simulation step.

Patterns:
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  Worker threads (running `execute`) act as consumers, processing them.
- Fine-grained Locking: A shared list of `Lock` objects (`locations`) ensures
  exclusive access to sensor data at specific locations during script execution.
- Thread Pool Management: A `Semaphore` (`free_threads`) controls the number
  of concurrently active worker threads.
"

from threading import Event, Thread
from threading import Semaphore, Lock
from Barrier import ReusableBarrier


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

        self.initialization_event = Event() # Event to signal completion of device-specific initialization
        self.free_threads = Semaphore(value=8) # Semaphore to limit the number of active worker threads
        self.locations = [] # Shared list of Lock objects for fine-grained data access
        self.barrier = None # Reference to the global ReusableBarrier

        self.device_threads = [] # List to hold dynamically spawned worker threads

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device synchronization and shared resources.

        If this device is the master (device_id 0), it initializes the global
        `ReusableBarrier` and a shared list of `Lock` objects (`locations`).
        These shared resources are then propagated to all other devices,
        and their `initialization_event` is set.

        @param devices: A list of all Device instances in the simulation.
        """

        num_devices = len(devices)
        # Block Logic: If this is the master device (device_id 0), it initializes shared resources.
        if self.device_id is 0:
            # Block Logic: Initializes a fixed number (30) of `Lock` objects for data locations.
            # This implies a maximum of 30 distinct data locations.
            locations = []
            number_of_locations = 30
            while number_of_locations > 0:
                locations.append(Lock())
                number_of_locations = number_of_locations - 1

            barrier = ReusableBarrier(num_devices) # Initialize the global barrier

            # Block Logic: Propagates the initialized shared `locations` (locks) and `barrier` to all devices,
            # and sets their `initialization_event` to allow their threads to proceed.
            for i in range(0, num_devices):
                devices[i].initialization_event.set()
                devices[i].locations = locations
                devices[i].barrier = barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script object is provided, it is added to the device's list of scripts,
        and the `script_received` event is set. If `script` is None, it signals that
        script assignments for the current timepoint are complete by setting the
        `timepoint_done` event.

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

    def clear_threads(self):
        """
        @brief Joins all worker threads spawned by this device and clears the list.

        Functional Utility: Ensures all dynamically created threads complete their
        execution before proceeding and frees up resources.
        """
        for thread in self.device_threads:
            thread.join()

        self.device_threads = []

    def shutdown(self):
        """
        @brief Shuts down the device by joining all its worker threads and its main thread.

        Functional Utility: Ensures proper termination and cleanup of all resources
        associated with the device's concurrent execution.
        """
        self.clear_threads() # First, join all dynamically spawned script threads
        self.thread.join() # Then, join the main DeviceThread

# @brief Executes a single script on sensor data, protecting access with a location-specific lock.
#
# This function is designed to be run by dynamically spawned worker threads.
# It acquires a lock for the target data location, gathers data from neighbors
# and the calling device, runs the script, updates the data, releases the lock,
# and finally releases a permit to the `free_threads` semaphore.
#
# @param device The `Device` instance that spawned this execution.
# @param script The script object to be executed.
# @param location The data location (e.g., sensor ID) to operate on.
# @param neighbours A list of neighboring `Device` instances.
def execute(device, script, location, neighbours):

    # Block Logic: Acquires the location-specific lock from `device.locations`
    # to ensure exclusive access to the data at this `location` during script execution.
    with device.locations[location]:
        script_data = []

        # Block Logic: Gathers relevant sensor data from neighboring devices for script input.
        for dev in neighbours:
            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)

        # Block Logic: Includes the current device's own sensor data in the script input.
        data = device.get_data(location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script only if there is any data to process.
        if script_data != []:

            # Functional Utility: Executes the assigned script with the collected data.
            result = script.run(script_data)

            # Block Logic: Propagates the script's result back to neighboring devices.
            for dev in neighbours:
                dev.set_data(location, result)

            # Block Logic: Updates the current device's own sensor data with the script's result.
            device.set_data(location, result)
        # Functional Utility: Releases a permit to the `free_threads` semaphore,
        # indicating that a worker slot is now available.
        device.free_threads.release()

class DeviceThread(Thread):
    """
    @brief A dedicated thread for each Device, managing its simulation logic.

    This thread is responsible for initializing and synchronizing worker threads,
    discovering neighbors, waiting for script assignments, and orchestrating
    concurrent script execution.
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

        Waits for the device's initialization to complete, then continuously
        performs neighborhood discovery, waits for script assignments for the
        current timepoint, spawns worker threads (running `execute`) to
        process scripts concurrently (limited by a semaphore), joins all
        worker threads, clears the list, and finally synchronizes globally
        using the shared barrier before starting the next timepoint.
        """

        self.device.initialization_event.wait() # Wait until initial device setup is complete

        while True:

            # Functional Utility: The `script_received` event is used to signal readiness for script processing.
            # The next two lines seem to be a placeholder or incomplete logic. They were commented out in the original.
            # self.device.script_received.wait()
            # self.device.script_received.clear()


            # Block Logic: Discovers neighboring devices for the current timepoint.
            # Pre-condition: `self.device.supervisor` is available to provide neighborhood information.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Checks for a shutdown condition (None neighbors indicates termination).
            if neighbours is None:
                break

            # Block Logic: Waits for the supervisor to signal that script assignments for
            # the current timepoint are complete.
            self.device.timepoint_done.wait()

            # Block Logic: Iterates through all assigned scripts and spawns worker threads
            # (running the global `execute` function) for concurrent processing.
            # The number of active threads is limited by `self.device.free_threads` semaphore.
            for (script, location) in self.device.scripts:

                # Block Logic: Acquires a permit from the `free_threads` semaphore, potentially blocking
                # if the maximum number of active worker threads (8) has been reached.
                self.device.free_threads.acquire()
                # Functional Utility: Creates a new thread to execute the global `execute` function
                # with the current script and its context.
                device_thread = Thread(target=execute, \
                           args=(self.device, script, location, neighbours))


                device_thread.start()
                self.device.device_threads.append(device_thread)

            # Block Logic: Clears the `timepoint_done` event, resetting it for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been dispatched to worker threads.
            self.device.timepoint_done.clear()

            # Block Logic: Joins all dynamically spawned worker threads for the current timepoint
            # and clears the list. This ensures all scripts are processed before proceeding.
            self.device.clear_threads()

            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            self.device.barrier.wait()