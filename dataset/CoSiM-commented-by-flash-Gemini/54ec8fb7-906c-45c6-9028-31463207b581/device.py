"""
@file device.py
@brief Implements a simulated distributed device environment with thread synchronization.

This module defines classes for simulating a network of devices that can
execute scripts, share sensor data, and synchronize operations using
reusable barriers. It models concurrent execution through threads and
manages shared resources with locks and semaphores.

Classes:
- `ReusableBarrier`: A synchronization primitive allowing a fixed number
  of threads to wait for each other before proceeding in two phases.
- `Device`: Represents an individual device in the network, managing
  its sensor data, scripts, and interaction with a supervisor.
- `DeviceThread`: The operational thread for each `Device` instance,
  handling script assignment, execution, and barrier synchronization.
- `ScriptExecutorService`: Manages a pool of `ScriptExecutor` threads
  to execute assigned scripts concurrently.
- `ScriptExecutor`: A thread responsible for executing a single script
  on a device, gathering data, running the script, and updating data.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    @brief A reusable barrier synchronization primitive for a fixed number of threads.

    This barrier allows `num_threads` to synchronize in two distinct phases,
    ensuring all threads reach a certain point before any can proceed.
    It prevents deadlock in multi-phase synchronization scenarios.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier.

        @param num_threads: The total number of threads that must reach the barrier
                            before any can proceed.
        """
        self.num_threads = num_threads
        # Two counters for two phases of the barrier. Each initialized to the total number of threads.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()  # Lock to protect access to the counters
        # Two semaphores, one for each phase, initialized to 0. Threads acquire these.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this point
               for both phases of the barrier.

        This method coordinates the synchronization of threads through two distinct phases.
        """
        # First phase of synchronization
        self.phase(self.count_threads1, self.threads_sem1)
        # Second phase of synchronization
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single phase of the barrier synchronization.

        A thread decrements a shared counter. If it's the last thread to
        reach the barrier (counter becomes 0), it releases all waiting threads
        and resets the counter. All threads then acquire the semaphore to proceed.

        @param count_threads: A list (to allow mutable integer) representing the
                              counter for the current phase.
        @param threads_sem: The semaphore associated with the current phase.
        """

        with self.count_lock:
            # Block Logic: Decrement the counter and check if this is the last thread.
            # Invariant: `count_threads[0]` always reflects the number of threads
            #            yet to reach this barrier phase.
            count_threads[0] -= 1
            if count_threads[0] == 0:  # If this is the last thread
                # Block Logic: Release all waiting threads and reset the counter for reuse.
                for i in range(self.num_threads):
                    threads_sem.release() # Release all waiting threads
                count_threads[0] = self.num_threads  # Reset counter for the next use
        threads_sem.acquire()  # Acquire the semaphore; this blocks until released by the last thread


class Device(object):
    """
    @brief Represents a single device in the distributed simulation.

    Each device has a unique ID, sensor data, a supervisor to interact with,
    and a list of scripts to execute. It manages its own operational thread
    and participates in barrier synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.

        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing initial sensor data for various locations.
        @param supervisor: A reference to the Supervisor object that manages the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []  # List to store (script, location) tuples assigned to this device
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's script assignments


        self.thread = DeviceThread(self) # The dedicated thread for this device's operations
        self.thread.start()
        self.barrier = None  # ReusableBarrier instance, only for device 0
        self.barrier_is_up = Event() # Event to signal that the barrier has been initialized by device 0

        # Dictionary to store locks for protecting access to specific locations (sensor data)
        self.location_acces = {}
        # Reference to device with ID 0, needed for barrier synchronization
        self.device0 = None

        # Lock to control access to the script assignment mechanism
        self.can_receive_scripts = Lock()

    def __str__(self):
        """
        @brief Provides a string representation of the device.

        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up inter-device dependencies and initializes the barrier.

        If this is device 0, it creates the `ReusableBarrier` for all devices.
        All devices also obtain a reference to device 0.
        @param devices: A list of all Device instances in the simulation.
        """


        # Conditional Logic: Only device 0 is responsible for initializing the global barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices)) # Initialize barrier with total number of devices
            self.barrier_is_up.set() # Signal that the barrier has been set up

        # Block Logic: All devices get a reference to device 0 (the coordinator).
        for device in devices:
            if device.device_id == 0:
                self.device0 = device
    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific sensor data location on this device.

        If `script` is None, it signals that no more scripts are coming for the current timepoint.
        This operation is protected by a lock to ensure thread safety during script assignment.
        @param script: The script object to assign, or None to signal end of timepoint assignments.
        @param location: The sensor data location (key in `sensor_data`) where the script should operate.
        """



        # Block Logic: Acquire lock to safely add script or signal timepoint completion.
        self.can_receive_scripts.acquire()
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its target location
        else:
            self.timepoint_done.set() # Signal that all scripts for this timepoint have been assigned
        self.can_receive_scripts.release() # Release the lock

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location on this device.

        @param location: The key for the sensor data.
        @return: The data at the specified location, or None if the location does not exist.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location on this device.

        @param location: The key for the sensor data.
        @param data: The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Gracefully shuts down the device's operational thread.
        """
        self.thread.join() # Wait for the device's thread to complete its execution


class DeviceThread(Thread):
    """
    @brief The main operational thread for a Device.

    This thread continuously fetches neighbor information, processes assigned scripts,
    and synchronizes with other device threads using a shared barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.

        @param device: The `Device` instance that this thread will operate on.
        """


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.core_semaphore = Semaphore(8)  # Semaphore to limit concurrent script executions per device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        It continuously performs the following steps for each timepoint:
        1. Gets neighbors from the supervisor.
        2. Waits for all scripts for the current timepoint to be assigned.
        3. Clears the timepoint done signal for the next round.
        4. If it's the first timepoint, waits for the barrier to be set up.
        5. Submits assigned scripts to an executor service.
        6. Waits for all submitted scripts to finish.
        7. Synchronizes with other devices using the `ReusableBarrier`.
        The loop breaks if no neighbors are returned (indicating simulation end).
        """
        timepoint = 0
        executor_service = ScriptExecutorService()

        # Block Logic: Main simulation loop for processing timepoints.
        # Pre-condition: Device is active and connected to a supervisor.
        # Invariant: Continues processing timepoints until supervisor signals no more neighbors.
        while True:
            neighbours = self.device.supervisor.get_neighbours() # Fetch current neighbors from supervisor
            if neighbours is None: # If no neighbors, means simulation is over
                break

            # Block Logic: Wait until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Functional Utility: Acquire lock to prevent new scripts from being assigned
            # while the current set is being processed.
            self.device.can_receive_scripts.acquire()

            # Functional Utility: Reset the event for the next timepoint.
            self.device.timepoint_done.clear()

            # Conditional Logic: For the very first timepoint (timepoint 0),
            # wait for device 0 to set up the global barrier.
            if timepoint == 0:
                self.device.device0.barrier_is_up.wait() # Wait for barrier setup signal
                timepoint = timepoint + 1 # Increment timepoint after first barrier setup

            # Block Logic: Submit all assigned scripts for concurrent execution.
            for (script, location) in self.device.scripts:
                executor_service.submit_job(script, self.device, location, neighbours)

            # Block Logic: Wait for all scripts submitted in this timepoint to complete execution.
            executor_service.wait_finish()

            # Functional Utility: Release the lock, allowing new script assignments for the next timepoint.
            self.device.can_receive_scripts.release()


            # Block Logic: Synchronize with other devices using the global barrier.
            # All device threads wait here until every participating device thread
            # has reached this point.
            self.device.device0.barrier.wait() # Participate in the global barrier synchronization

class ScriptExecutorService(object):
    """
    @brief Manages the execution of scripts in parallel using a pool of threads.

    It limits the number of concurrently running scripts via a semaphore and
    provides methods to submit jobs and wait for their completion.
    """
    def __init__(self):
        """
        @brief Initializes the ScriptExecutorService.
        """

        self.core_semaphore = Semaphore(8)  # Semaphore to limit concurrent script executions
        self.executors = []  # List to keep track of active ScriptExecutor threads

    def submit_job(self, script, device, location, neighbours):
        """
        @brief Submits a script for execution as a new `ScriptExecutor` thread.

        It acquires a semaphore to control concurrency before starting the thread.
        @param script: The script to be executed.
        @param device: The `Device` instance on which the script is running.
        @param location: The sensor data location relevant to the script.
        @param neighbours: A list of neighboring `Device` instances.
        """


        # Functional Utility: Create a new ScriptExecutor thread for the given job.
        executor = ScriptExecutor(script, device, location, neighbours, self.core_semaphore)

        # Block Logic: Acquire semaphore to ensure that not more than 8 scripts run concurrently.
        self.core_semaphore.acquire() # Blocks if 8 executors are already running
        executor.start() # Start the script execution thread
        self.executors.append(executor) # Add to the list of active executors

    def wait_finish(self):
        """
        @brief Waits for all currently submitted script executor threads to complete.
        """
        # Block Logic: Join each executor thread, effectively waiting for its completion.
        for executor in self.executors:
            executor.join()
        self.executors = [] # Clear the list of executors after they have all finished

class ScriptExecutor(Thread):
    """
    @brief A thread dedicated to executing a single script on a device.

    It handles data acquisition from the device and its neighbors, runs the
    script with the collected data, and then updates the device's and
    neighbors' data with the script's result. It also manages access
    to shared data locations using locks.
    """
    def __init__(self, script, device, location, neighbours, core_semaphore):
        """
        @brief Initializes a ScriptExecutor thread.

        @param script: The script object to execute.
        @param device: The `Device` instance on which the script is running.
        @param location: The sensor data location (key in `sensor_data`) where the script operates.
        @param neighbours: A list of neighboring `Device` instances.
        @param core_semaphore: The semaphore from `ScriptExecutorService` to be released
                               upon completion of the script.
        """
        Thread.__init__(self, name="Script Executor pentru device-ul %d" % device.device_id)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script
        self.core_semaphore = core_semaphore

    def run(self):
        """
        @brief The main execution logic for a single script.

        It ensures exclusive access to the data location, gathers data from
        the local device and its neighbors, executes the script, disseminates
        the results back to the device and neighbors, and finally releases
        the data location lock and the core semaphore.
        """

        # Conditional Logic: Initialize a lock for the specific location if it doesn't exist yet
        # in device 0's central `location_acces` dictionary.
        if self.location not in self.device.device0.location_acces:
            self.device.device0.location_acces[self.location] = Lock()


        # Block Logic: Acquire a lock for the specific data location to ensure exclusive access
        # while reading and writing data for this script execution.
        self.device.device0.location_acces[self.location].acquire()
        script_data = [] # List to accumulate data for the script
        data = None

        # Block Logic: Gather data from neighboring devices for the given location.
        if self.neighbours is not None:
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data) # Add neighbor's data if available

        # Functional Utility: Gather data from the local device for the given location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data) # Add local device's data if available

        # Conditional Logic: Execute the script only if some data was collected.
        if script_data != []:
            # Functional Utility: Run the assigned script with the collected data.
            result = self.script.run(script_data)

            # Block Logic: Disseminate the script's result to neighboring devices.
            if self.neighbours is not None:
                for device in self.neighbours:
                    device.set_data(self.location, result)

            # Functional Utility: Update the local device's data with the script's result.
            self.device.set_data(self.location, result)

        # Functional Utility: Release the lock for the data location, allowing other
        # script executors to access it.
        self.device.device0.location_acces[self.location].release()

        # Functional Utility: Release the core semaphore, signaling that this script
        # execution slot is now free for another script.
        self.core_semaphore.release()