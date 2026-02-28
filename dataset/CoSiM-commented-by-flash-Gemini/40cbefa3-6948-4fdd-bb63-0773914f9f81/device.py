

"""
@file device.py
@brief Implements core components for a distributed device simulation.

This module defines the `ReusableBarrier`, `Device`, `ScriptThread`, and `DeviceThread` classes,
which together simulate a network of interconnected devices processing sensor data
and executing scripts. The architecture supports synchronization between devices
and their internal script execution threads, as well as interaction with a central supervisor.

Classes:
    - ReusableBarrier: A custom implementation of a barrier for synchronizing multiple threads.
    - Device: Represents an individual device in the simulation, managing its
              sensor data, script queue, and associated threads.
    - ScriptThread: A transient thread created by `DeviceThread` to execute a single script.
    - DeviceThread: The main execution thread for a Device, responsible for
                    synchronization, fetching neighbor information, and
                    orchestrating script execution via `ScriptThread`s.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    @class ReusableBarrier
    @brief Implements a reusable barrier for synchronizing multiple threads.

    This barrier allows a fixed number of threads (`num_threads`) to wait
    for each other at a synchronization point. Once all threads have arrived,
    they are all released simultaneously. It uses a two-phase approach
    (implemented via `phase` method) to ensure reusability without deadlocks.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes a ReusableBarrier instance.
        @param num_threads The total number of threads that must reach the barrier
                           before any of them can proceed.
        """
        self.num_threads = num_threads  # Total number of threads expected at the barrier.
        # Counter for the first phase of the barrier. Using a list to allow modification within `with Lock`.
        self.count_threads1 = [self.num_threads]
        # Counter for the second phase of the barrier, crucial for reusability.
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()  # Mutex to protect access to the thread counters.
        
        self.threads_sem1 = Semaphore(0)  # Semaphore for the first phase, blocks threads until all arrive.
        
        self.threads_sem2 = Semaphore(0)  # Semaphore for the second phase, blocks threads until all arrive for the second time.

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier.
        The thread will be blocked until all `num_threads` threads have called this method.
        """
        # Execute the first phase of the barrier.
        self.phase(self.count_threads1, self.threads_sem1)
        # Execute the second phase of the barrier for reusability.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements a single phase of the barrier synchronization.
        @param count_threads A list containing the current count of threads waiting for this phase.
        @param threads_sem The semaphore associated with this phase to block/release threads.
        
        Each thread decrements `count_threads`. The last thread to decrement
        (`count_threads` becomes 0) releases all waiting threads via `threads_sem`,
        and then resets the counter for the next use.
        """
        with self.count_lock: # Protect the shared counter.
            count_threads[0] -= 1
            
            if count_threads[0] == 0: # If this is the last thread to arrive:
                # Release all waiting threads (including itself).
                for _ in range(self.num_threads):
                    threads_sem.release()
                
                # Reset the counter for the next use of this barrier phase.
                count_threads[0] = self.num_threads
        # Acquire the semaphore; this blocks the thread until it is released by the last arriving thread.
        threads_sem.acquire()


class Device(object):
    """
    @class Device
    @brief Represents a single device within the distributed simulation network.

    Manages its unique identifier, local sensor data, interaction with a supervisor,
    and various synchronization primitives for coordinating with other devices
    and its internal `DeviceThread` for script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id Unique identifier for this device.
        @param sensor_data Initial sensor data pertinent to this device.
        @param supervisor A reference to the central supervisor managing the network.
        """
        self.device_id = device_id  # Unique identifier for the device.
        self.sensor_data = sensor_data  # Dictionary holding sensor data specific to this device.
        self.supervisor = supervisor  # Reference to the supervisor object for network coordination.
        self.scripts = []  # List to store scripts assigned to this device for execution.
        # Event flag to signal that all scripts for a timepoint are assigned.
        self.timepoint_done = Event()
        self.barrier = None  # Global synchronization barrier shared by all devices.
        # The main thread responsible for this device's high-level operations (e.g., script dispatch).
        self.thread = DeviceThread(self)
        self.thread.start() # Immediately start the main DeviceThread upon initialization.
        # List of Locks, where each lock protects sensor data at a specific location.
        self.location_locks = None

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the device for the simulation, especially handling global resources.
        @param devices A list of all Device instances in the simulation.

        This method is critical for initialization. If this is the master device (device_id == 0),
        it initializes the global `ReusableBarrier` and creates `Lock` objects for each
        unique sensor data location found across all devices. These global resources are
        then propagated to all other devices.
        """
        # Master device (ID 0) performs initial setup for all devices.
        if 0 == self.device_id:
            # Initialize a global reusable barrier for all devices, based on the total number of devices.
            self.barrier = ReusableBarrier(len(devices))
            
            locations = []
            # Identify all unique sensor data locations across all devices.
            for device in devices:
                for location in device.sensor_data:
                    if location not in locations:
                        locations.append(location)
            
            # Initialize a list of Locks, one for each unique location, to protect concurrent data access.
            self.location_locks = []
            for _ in range(len(locations)):
                self.location_locks.append(Lock())
            
            # Propagate the initialized global barrier and location-specific locks to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device's internal list of scripts.
        @param script The script object to be executed.
        @param location The sensor data location that the script primarily operates on.

        If `script` is None, it acts as a signal that script assignment for the
        current timepoint is complete, and sets the `timepoint_done` event.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If a None script is received, it signifies the end of script assignments for a timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        @param location The location identifier for which to retrieve data.
        @return The sensor data at the specified location, or None if not present.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.
        @param location The location identifier for which to set data.
        @param data The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its main `DeviceThread`.
        Ensures the thread completes its execution before the program exits cleanly.
        """
        self.thread.join()


class ScriptThread(Thread):
    """
    @class ScriptThread
    @brief Executes a single assigned script for a device.

    This thread is responsible for acquiring the necessary data locks,
    gathering relevant sensor data from its parent device and neighboring devices,
    executing the provided script with this data, and then updating
    the sensor data on both the parent device and its neighbors.
    """
    def __init__(self, device, script, location, neighbours):
        """
        @brief Initializes a ScriptThread instance.
        @param device The parent Device instance for which this script is being executed.
        @param script The script object to be run.
        @param location The specific sensor data location this script operates on.
        @param neighbours A list of neighboring Device instances.
        """
        Thread.__init__(self)
        self.device = device  # Reference to the parent device.
        self.script = script  # The script to execute.
        self.location = location  # The data location this script pertains to.
        self.neighbours = neighbours  # List of neighbor devices for data exchange.

    def run(self):
        """
        @brief The execution logic for the ScriptThread.

        This method acquires a lock for the specific data location to ensure
        exclusive access, collects data from itself and neighbors, executes
        the script, and then propagates the results back to itself and neighbors.
        Finally, it releases the data lock.
        """
        # Acquire a lock for the specific data location to prevent race conditions.
        with self.device.location_locks[self.location]:
            script_data = [] # List to accumulate data needed by the script.
            
            # Gather data from neighboring devices for the specified location.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from this device itself for the specified location.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
            # If any data was collected, run the script and update relevant devices.
            if script_data != []:
                # Execute the script with the collected data.
                result = self.script.run(script_data)
                
                # Propagate the result to neighboring devices.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                # Update this device's own data with the script result.
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief The primary control thread for a `Device` instance.

    This thread manages the overall execution flow for its associated device
    across simulation timepoints. It handles synchronization with the supervisor
    and other devices, fetches neighbor information, dispatches scripts to
    `ScriptThread`s for concurrent execution, and waits for their completion.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        This loop continuously processes simulation timepoints. It obtains
        neighbor information, waits for script assignments to be complete,
        launches `ScriptThread`s to execute assigned tasks concurrently,
        and then waits for all script threads to finish before synchronizing
        globally via the `ReusableBarrier`. The loop breaks if the supervisor
        signals the end of the simulation.
        """
        while True:
            # Retrieve updated neighbor information from the supervisor.
            vecini = self.device.supervisor.get_neighbours() # 'vecini' is Romanian for neighbors.
            if vecini is None:
                # If supervisor returns None, it signals the end of the simulation.
                break
            
            # Wait for all scripts for the current timepoint to be assigned and marked as done.
            self.device.timepoint_done.wait()
            threads = [] # List to hold references to active ScriptThread instances.
            
            # Check if there are neighbors (and thus potentially scripts to process or data to exchange).
            if len(vecini) != 0:
                # For each script assigned to this device:
                for (script, locatie) in self.device.scripts: # 'locatie' is Romanian for location.
                    # Create a new ScriptThread to execute this script.
                    thread = ScriptThread(self.device, script, locatie, vecini)
                    threads.append(thread)
                    thread.start() # Start the script execution in a new thread.
                
                # Wait for all launched ScriptThreads to complete their execution.
                for thread in threads:
                    thread.join()
            
            # Clear the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()
            
            # Clear the list of scripts for the next timepoint.
            self.device.scripts = [] # Reset scripts list after execution.
            
            # Wait at the global barrier for all devices to synchronize before proceeding to the next timepoint.
            self.device.barrier.wait()

