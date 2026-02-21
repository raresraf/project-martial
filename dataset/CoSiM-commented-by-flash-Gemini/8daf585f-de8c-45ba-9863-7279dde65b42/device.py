"""
@file device.py
@brief Implements a simulated device and its threading model for distributed sensing and script execution.

This module defines the `Device` and `DeviceThread` classes, which together
simulate a distributed system of interconnected devices. Each `Device` manages
its local sensor data, communicates with a supervisor, and executes assigned scripts.
The `DeviceThread` class enables concurrent processing within each device,
handling neighborhood discovery, script execution, and synchronization.

Architecture:
- `Device`: Represents a single node in a distributed sensing network.
  Manages local state, inter-device communication mechanisms (via supervisor),
  and script assignments.
- `DeviceThread`: A worker thread within a `Device` responsible for
  concurrent execution of tasks, including neighborhood updates and script processing.
- `cond_barrier.ReusableBarrier`: Used for synchronization across multiple threads
  within a device and across multiple devices at timepoint boundaries.

Patterns:
- Producer-Consumer: The supervisor assigns scripts to devices, which are then
  consumed by `DeviceThread` instances.
- Barrier Synchronization: Ensures all threads/devices reach a specific point
  before proceeding, crucial for time-step simulations.
"""
import cond_barrier
from threading import Event, Thread, Lock


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensing network.

    Manages local sensor data, orchestrates script execution, and handles
    synchronization with other devices and a central supervisor.
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
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.threads = []

        self.neighbourhood = None
        self.map_locks = {}
        self.threads_barrier = None
        self.barrier = None
        self.counter = 0
        self.threads_lock = Lock()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device communication and synchronization mechanisms.

        If this is the master device (device_id 0), it initializes a global
        reusable barrier for all devices. All devices initialize their internal
        thread barriers and worker threads.

        @param devices: A list of all Device instances in the simulation.
        """
        # Block Logic: Initializes a global synchronization barrier across all devices.
        # This ensures that all devices complete a timepoint's processing before moving to the next.
        if self.device_id == 0:
            num_threads = len(devices)

            # Invariant: The barrier is initialized with a count equal to the total
            # number of device threads across all devices in the simulation.
            self.barrier = cond_barrier.ReusableBarrier(num_threads * 8)

            # Block Logic: Propagates the shared barrier and map locks to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.map_locks = self.map_locks

        # Block Logic: Initializes internal threads and their synchronization barrier
        # for concurrent operations within this specific device.
        self.threads_barrier = cond_barrier.ReusableBarrier(8)
        for i in range(8):
            self.threads.append(DeviceThread(self, i, self.threads_barrier))

        # Block Logic: Starts all worker threads associated with this device.
        for thread in self.threads:
            thread.start()


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script is provided, it's added to the device's script queue,
        and an event is set to signal script availability to worker threads.
        If no script is provided (None), it signals that the current timepoint
        processing is complete for script assignment.

        @param script: The script object to execute, or None to signal timepoint completion.
        @param location: The data location (e.g., sensor ID) where the script
                         should operate.
        """

        # Block Logic: Manages script assignment and signals to worker threads.
        # Pre-condition: 'script' can be an executable object or None.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

        # Block Logic: Initializes a lock for a given data location if one doesn't already exist.
        # This ensures exclusive access to sensor data at specific locations during script execution.
        if location not in self.map_locks:
            self.map_locks[location] = Lock()

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
        # Block Logic: Updates sensor data only if the location already exists.
        # This prevents accidental creation of new data locations.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining all its worker threads.

        Functional Utility: Ensures proper termination and cleanup of resources
        associated with the device's concurrent execution.
        """

        # Block Logic: Waits for all internal worker threads to complete their execution.
        # This is a critical step for graceful shutdown, preventing resource leaks or unexpected termination.
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    @brief A worker thread operating within a Device.

    Each DeviceThread is responsible for tasks such as discovering
    neighbors, waiting for scripts, and executing them on local and
    neighboring data.
    """

    def __init__(self, device, id, barrier):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: A reference to the parent Device.
        @param id: A unique identifier for this thread within the device.
        @param barrier: A reusable barrier for synchronizing threads within the device.
        """

        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id = id
        self.thread_barrier = barrier

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Continuously performs neighborhood discovery, waits for timepoint
        completion signals, and executes assigned scripts until a shutdown
        condition is met.
        """
        while True:
            # Block Logic: Performs neighborhood discovery if this is the first thread (id 0).
            # Pre-condition: This block is executed once per timepoint by a designated thread.
            if self.id == 0:
                self.device.neighbourhood = self.device.supervisor.get_neighbours()

            # Block Logic: Synchronizes all threads within the device before proceeding.
            # Invariant: All threads within this device have completed their neighborhood
            # update (or skipped it if not id 0) before moving past this barrier.
            self.thread_barrier.wait()

            # Block Logic: Checks for a shutdown condition (no neighborhood indicates termination).
            if self.device.neighbourhood is None:
                break

            # Block Logic: Waits for the supervisor to signal that script assignments for
            # the current timepoint are complete.
            # Invariant: No new scripts will be assigned for the current timepoint after this point.
            self.device.timepoint_done.wait()

            # Block Logic: Processes all assigned scripts for the current timepoint.
            # Each script is executed on relevant sensor data, with proper locking.
            while True:
                # Block Logic: Atomically assigns a script from the device's queue to the current thread.
                # Invariant: Ensures that each script is processed exactly once by one thread.
                with self.device.threads_lock:
                    if self.device.counter == len(self.device.scripts):
                        break
                    (script, location) = self.device.scripts[self.device.counter]
                    self.device.counter = self.device.counter + 1
                # Block Logic: Acquires a lock for the specific data location to prevent race conditions
                # during sensor data access and modification.
                self.device.map_locks[location].acquire()
                script_data = []

                # Block Logic: Gathers relevant sensor data from neighboring devices for script input.
                for device in self.device.neighbourhood:
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
                    for device in self.device.neighbourhood:
                        device.set_data(location, result)
                    # Block Logic: Updates the device's own sensor data with the script's result.
                    self.device.set_data(location, result)

                # Block Logic: Releases the lock for the data location, allowing other threads to access it.
                self.device.map_locks[location].release()

            # Block Logic: Synchronizes all threads across all devices after all scripts for the
            # current timepoint have been executed.
            # Invariant: All devices have completed their script execution for the current timepoint.
            self.device.barrier.wait()
            # Block Logic: Resets timepoint-specific state for the next iteration (only by thread 0).
            # Pre-condition: This ensures that these resets happen only after all threads/devices have synchronized.
            if self.id == 0:
                self.device.counter = 0
                self.device.timepoint_done.clear()
