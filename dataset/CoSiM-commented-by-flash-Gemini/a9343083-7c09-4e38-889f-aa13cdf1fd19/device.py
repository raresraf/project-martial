
"""
@file device.py
@brief Implements a simulated distributed device using an external reusable barrier and multi-threaded script execution with fine-grained locking.

This module defines the `Device`, `MyThread`, and `DeviceThread` classes,
which together simulate a node in a distributed sensing network. Each `Device`
manages its local sensor data and executes assigned scripts. A `DeviceThread`
per device coordinates the overall timepoint simulation, while `MyThread`s
are spawned to concurrently execute individual scripts. A shared external
`ReusableBarrier` ensures global timepoint synchronization, and a list of
`Lock` objects provides fine-grained control over access to sensor data locations.

Architecture:
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It holds references to the global barrier
  and the shared list of data locks.
- `DeviceThread`: A dedicated thread per `Device` that orchestrates the
  execution for each timepoint, including neighborhood discovery, spawning
  `MyThread`s, and global timepoint synchronization.
- `MyThread`: Worker threads spawned by `DeviceThread` to execute
  individual scripts. Each script thread acquires a lock for its target
  data location, gathers data from neighbors and itself, runs the script,
  updates data, and releases the lock.
- `ReusableBarrier`: An external reusable barrier used for global synchronization
  across all devices at each simulation step.

Patterns:
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `MyThread` instances (spawned by `DeviceThread`) act as consumers,
  processing them.
- Fine-grained Locking: A shared list of `Lock` objects (`self.lock`) ensures
  exclusive access to sensor data at specific locations during script execution
  across concurrent worker threads.
"""

from threading import Event, Thread, Lock
import ReusableBarrier


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
        self.barrier = None # Reference to the global ReusableBarrier
        self.lock = [] # List of Locks for fine-grained synchronization on data locations

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device synchronization mechanisms.

        This method initializes a shared `ReusableBarrier` and a shared list of `Lock`
        objects for all devices. These synchronization primitives are then propagated
        to all `Device` instances in the simulation. This setup is typically
        performed by a designated master device (e.g., the first device in the list).

        @param devices: A list of all Device instances in the simulation.
        """

        # Block Logic: Initializes the global barrier and the list of locks.
        # This setup is assumed to be handled by a master device or globally.
        barrier = ReusableBarrier.ReusableBarrier(len(devices)) # Create a single global barrier
        lock = []
        # Block Logic: Initializes a fixed number (100) of `Lock` objects for data locations.
        # This implies a maximum of 100 distinct data locations.
        for _ in range(0, 100):
            newlock = Lock()
            lock.append(newlock)

        # Block Logic: Propagates the initialized global barrier and list of locks to all devices.
        for dev in devices:
            dev.barrier = barrier
            dev.lock = lock


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

        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location: The identifier for the data location.
        @param data: The new data to set.
        """
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

    Each `MyThread` processes one assigned script, acquiring the necessary
    location-specific lock, gathering data from the local device and its neighbors,
    executing the script, and propagating the results back to the relevant devices.
    """

    def __init__(self, device, location, script, neighbours):
        """
        @brief Initializes a new MyThread instance.

        @param device: A reference to the parent `Device` instance.
        @param location: The data location (e.g., sensor ID) for the script.
        @param script: The script object to execute.
        @param neighbours: A list of neighboring `Device` instances.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours
    def run(self):
        """
        @brief The main execution logic for the `MyThread`.

        Acquires the location-specific lock, gathers data from neighbors and
        the local device, executes the script, updates sensor data on affected
        devices, and then releases the lock.
        """
        # Block Logic: Acquires the location-specific lock to ensure exclusive access
        # to the sensor data at `self.location` during script execution.
        self.device.lock[self.location].acquire()
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

            # Block Logic: Propagates the script's result back to neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)

        self.device.set_data(self.location, result) # Update the current device's own sensor data
        # Block Logic: Releases the location-specific lock, allowing other threads to access it.
        self.device.lock[self.location].release()

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
            # Block Logic: The purpose of calling set() again here is unclear and might be redundant
            # if `timepoint_done` is meant to be a one-shot signal per timepoint that is then cleared.
            self.device.timepoint_done.set() # Re-sets the event, potentially to prevent immediate re-wait?


            threads = []
            # Block Logic: Spawns a `MyThread` for each assigned script to execute them concurrently.
            # Pre-condition: `self.device.scripts` contains all scripts to be processed in this timepoint.
            for (script, location) in self.device.scripts:
                thread_aux = MyThread(self.device, location, script, neighbours) # Create a new MyThread

                threads.append(thread_aux)
            # Block Logic: Starts all `MyThread` instances.
            for auxiliar_thread in threads:
                auxiliar_thread.start()
            # Block Logic: Waits for all `MyThread` instances to complete their execution.
            for auxiliar_thread in threads:
                auxiliar_thread.join()

            # Block Logic: Clears the timepoint_done event, resetting it for the next timepoint.
            # Pre-condition: All scripts for the current timepoint have been processed.
            self.device.timepoint_done.clear()
            # Block Logic: Synchronizes all DeviceThread instances globally using the shared barrier.
            # Invariant: All device threads complete their script processing for the current timepoint
            # and reach this barrier before any proceed to the next timepoint.
            self.device.barrier.wait()


