"""
@file device.py
@brief Implements a simulated distributed device using a custom barrier and multi-threaded computation for script execution.

This module defines the `ReusableBarrierCond`, `Device`, `ComputationThread`,
and `DeviceThread` classes. The `Device` class represents a node in a simulated
sensing network, managing local sensor data and coordinating script execution.
A custom `ReusableBarrierCond` provides synchronization for all devices, while
`ComputationThread` instances perform the actual script processing concurrently.

Architecture:
- `ReusableBarrierCond`: A custom barrier implementation using `threading.Condition`
  for synchronizing multiple threads/devices. It ensures all participants reach
  a specific point before proceeding.
- `Device`: Represents a single node. Manages local state, assigned scripts,
  and its dedicated `DeviceThread`. It also holds a reference to the global barrier.
- `DeviceThread`: A dedicated thread per `Device` that orchestrates the execution
  for each timepoint, including neighborhood discovery and spawning `ComputationThread`s.
- `ComputationThread`: Worker threads spawned by `DeviceThread` to execute
  individual scripts, gather data from neighbors, and update sensor readings.

Patterns:
- Barrier Synchronization: Ensures all devices/threads reach a specific point
  before proceeding, crucial for time-step simulations.
- Producer-Consumer: `assign_script` acts as a producer, adding scripts.
  `ComputationThread` instances (spawned by `DeviceThread`) act as consumers,
  processing them.
- Concurrent Execution: Multiple `ComputationThread` instances run in parallel
  within each device to speed up script processing.
"""

"""
@9b2f2e53-b725-4b33-b9ce-27af79738cdd/device.py
@brief Implements a multi-threaded device simulation with a custom condition-variable-based reusable barrier.

This module defines a `Device` class that manages sensor data and orchestrates script execution
through a main `DeviceThread`. For each script, a dedicated `ComputationThread` is spawned
to process data. Synchronization across devices for timepoint progression is handled
by a custom `ReusableBarrierCond` implementation, ensuring coordinated advancement
of the simulation, alongside a lock for safe data modification.
"""

from threading import Event, Thread, Lock, Condition


class ReusableBarrierCond:
    """
    @brief A reusable synchronization barrier implemented using a Condition variable.

    This barrier allows a specified number of threads to wait until all have reached
    a specific point of execution before any are allowed to proceed. It is designed
    to be reusable for multiple synchronization points within a simulation.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrierCond instance.

        Sets the total number of threads that must reach the barrier and initializes
        the internal counter and condition variable.

        @param num_threads: The total number of threads expected to synchronize at this barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Current count of threads waiting
        self.cond = Condition() # Condition variable for synchronization

    def wait(self):
        """
        @brief Blocks the calling thread until all registered threads have reached the barrier.

        Acquires a condition lock, decrements the internal count of threads yet to reach the barrier.
        If this thread is the last to arrive, it notifies all waiting threads and
        resets the barrier for reuse. Otherwise, it waits until signaled by the last thread.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # Block Logic: If this is the last thread to arrive, notify all others and reset.
        if self.count_threads == 0:
            self.cond.notify_all() # Release all waiting threads
            self.count_threads = self.num_threads # Reset for next use
        # Block Logic: If not the last thread, wait until notified by the last thread.
        else:
            self.cond.wait() # Release the lock and wait; reacquires lock on wakeup
        self.cond.release() # Release the lock


class Device(object):
    """
    @brief Represents a simulated device managing sensor data and script execution.

    Each Device instance is responsible for its unique ID, sensor readings,
    and a reference to the supervisor. It processes scripts through its `DeviceThread`
    which, in turn, spawns `ComputationThread`s. Synchronization is managed
    globally via a `ReusableBarrierCond` and locally via a `Lock` for data modification.
    """
    
    barrier = None

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        Sets up device-specific attributes such as ID, sensor data, and supervisor reference.
        It also initializes internal state for script management, event signaling,
        and thread management. The global barrier (`barrier`) is initialized to None
        and is expected to be set up by the `setup_devices` method of a coordinating device.
        A `set_data_lock` is initialized to ensure thread-safe data modification.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's initial sensor readings.
        @param supervisor: A reference to the supervisor object managing the device network.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when scripts are assigned
        self.scripts = [] # List to store assigned scripts
        self.timepoint_done = Event() # Event to signal that script assignment for timepoint is complete
        self.set_data_lock = Lock() # Lock to protect concurrent writes to sensor_data
        self.thread = DeviceThread(self) # Dedicated thread for this device's operations
        self.thread.start() # Starts the device's main thread

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the global synchronization barrier.

        This method identifies a coordinating device (typically `device_id == 0`)
        which initializes a global `ReusableBarrierCond` to synchronize all devices
        in the simulation. This barrier is then propagated to all other devices
        to ensure consistent synchronization across the network.

        @param devices: A list of all Device instances participating in the simulation.
        """

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be processed or signals completion of a timepoint.

        If a `script` is provided, it is appended to the device's internal `scripts` list,
        and the `script_received` event is set.
        If `script` is None, it signals that all scripts for the current timepoint
        have been received by setting the `timepoint_done` event.

        @param script: The script object to be executed, or None to signal timepoint completion.
        @param location: The data location (e.g., sensor ID) the script operates on.
        """
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a script has been received
        else:
            self.timepoint_done.set() # Signal that all scripts for this timepoint are assigned

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The key identifying the sensor data to retrieve.
        @return: The sensor data at the specified location, or None if not found.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location, ensuring thread safety.

        Access to modify the sensor data is protected by `self.set_data_lock`
        to prevent race conditions during concurrent write operations from multiple threads.

        @param location: The key identifying the sensor data to update.
        @param data: The new data value to set for the specified location.
        """

    def shutdown(self):
        """
        @brief Shuts down the device's main processing thread.

        This method waits for the device's main `DeviceThread` to complete
        its execution, ensuring a clean and orderly shutdown.
        """
        
        self.thread.join()


class ComputationThread(Thread):
    """
    @brief Represents a worker thread that executes a single script for a Device.

    Each `ComputationThread` is spawned by the `DeviceThread` to process a specific script.
    Its responsibility is to gather relevant data from neighbors and the device itself,
    execute the script, and then disseminate the results.
    """
    

    def __init__(self, device_thread, neighbours, script_data):
        """
        @brief Initializes a new ComputationThread instance.

        Associates the thread with its parent `DeviceThread` instance,
        a snapshot of the current `neighbours` list, and the `script`
        and `location` data to execute.

        @param device_thread: The parent DeviceThread instance that spawned this computation thread.
        @param neighbours: A list of neighboring Device instances relevant for this script's execution.
        @param script_data: A tuple containing the script object and its associated location.
        """
        Thread.__init__(self, name="Worker %s" % device_thread.name)
        self.device_thread = device_thread
        self.neighbours = neighbours
        self.script = script_data[0] # The script to execute
        self.location = script_data[1] # The sensor data location for the script

    def run(self):
        """
        @brief Executes the assigned script, gathering data and disseminating results.

        This method performs the core logic of a script worker:
        1.  Collects relevant sensor data from all `neighbours` and the worker's
            `device` (accessed via `device_thread.device`) for the specified `location`.
        2.  If data is available, it executes the provided `script` with the
            collected data.
        3.  Disseminates the `result` of the script execution by updating the
            sensor data of all `neighbours` and the worker's own `device` at
            the specified `location`.
        """
        script_data = []
        # Block Logic: Gathers relevant sensor data from all specified neighbors for the current script's location.
        # Functional Utility: Collects necessary input for the script based on the current network state.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Block Logic: Includes the device's own sensor data for the current script's location.
        # Functional Utility: Ensures the script considers the device's local state.
        data = self.device_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Executes the script only if there is data to process.
        if script_data:
            # Block Logic: Executes the assigned script with the aggregated data.
            # Architectural Intent: Decouples computational logic from data management,
            #                      allowing dynamic script execution based on current data.
            result = self.script.run(script_data)

            # Block Logic: Disseminates the computed result to neighboring devices.
            # Functional Utility: Propagates state changes across the network as a result of script execution.
            for device in self.neighbours:
                device.set_data(self.location, result)

            # Block Logic: Updates the device's own sensor data with the computed result.
            # Functional Utility: Reflects local state changes due to script processing.
            self.device_thread.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    @brief Orchestrates the execution of `ComputationThread`s for a Device at each timepoint.

    This thread acts as the main control unit for a `Device`, responsible for
    fetching neighborhood information and spawning a `ComputationThread` for each
    assigned script. It ensures all computation threads complete their tasks and
    then participates in a global barrier synchronization before
    proceeding to the next timepoint.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        Sets up the thread with a descriptive name and associates it with
        the Device instance it will manage.

        @param device: The Device instance this thread is responsible for.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        This loop continuously performs the following steps for each simulation timepoint:
        1.  Fetches the current set of neighboring devices from the supervisor.
        2.  Terminates if no neighbors are found, signifying the end of the simulation for this device.
        3.  Waits for the supervisor to signal that a new timepoint has begun.
        4.  Spawns a `ComputationThread` for each assigned script to process them concurrently.
        5.  Waits for all `ComputationThread`s to complete their execution for the current timepoint.
        6.  Clears the timepoint completion event, preparing for the next timepoint.
        7.  Participates in a global `ReusableBarrierCond` synchronization, ensuring all devices
            are synchronized before advancing to the next timepoint.
        """
        while True:
            # Block Logic: Fetches the current set of active neighbors for data exchange.
            # Functional Utility: Dynamically updates the device's awareness of its network topology.
            neighbours = self.device.supervisor.get_neighbours()
            # Invariant: If no neighbors are returned, the simulation for this device is complete.
            if neighbours is None:
                break

            # Block Logic: Waits for the supervisor to signal the start of a new timepoint.
            # Functional Utility: Orchestrates the progression of simulation timepoints.
            self.device.timepoint_done.wait()

            # Block Logic: Spawns a ComputationThread for each assigned script to process them concurrently.
            # Architectural Intent: Leverages multi-threading to parallelize the execution of scripts
            #                      for the current timepoint.
            local_threads = []
            # Block Logic: Spawns a `ComputationThread` for each assigned script.
            # Invariant: Each script is processed by an independent worker thread.
            for script_data in self.device.scripts:
                worker = ComputationThread(self, neighbours, script_data)
                worker.start()
                local_threads.append(worker)

            # Block Logic: Waits for all spawned ComputationThread instances to complete their assigned tasks.
            # Functional Utility: Ensures all scripts for the current timepoint are fully processed
            #                      before signaling completion and moving to the next step.
            for worker in local_threads:
                worker.join()

            # Block Logic: Clears the timepoint completion event, preparing for the next timepoint.
            # Functional Utility: Resets the event for a new cycle of timepoint synchronization.
            self.device.timepoint_done.clear()

            # Block Logic: Global synchronization point for all devices across the simulation.
            # Functional Utility: Ensures all devices have completed their processing for the current
            #                      timepoint before advancing to the next.
            self.device.barrier.wait()
