


"""
@906a4d6e-1304-4399-933e-f4ac251ed457/device.py
@brief Implements a multi-threaded simulation for distributed sensor devices with explicit worker thread management and condition-based barrier.

This module defines the core components for simulating a network of sensor devices,
each capable of executing scripts, managing local sensor data, and interacting
with a central supervisor. This version utilizes `Worker` threads for script
execution, dynamically managed by a `DeviceThread`, and a custom `ReusableBarrier`
implemented with `Condition` variables for timepoint synchronization.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate with neighbors under the
guidance of a supervisor.

Classes:
- Device: Represents a single simulated sensor device.
- DeviceThread: Manages the lifecycle and operation of a Device, including
                orchestrating `Worker` threads.
- Worker: A dedicated thread for executing individual scripts.
- ReusableBarrier: A custom barrier implementation using `Condition` variables
                   for thread synchronization.

Domain: Distributed Systems Simulation, Concurrent Programming, Condition Variables, Sensor Networks.
"""

from threading import Event, Thread, Condition, Lock


class Device(object):
    """
    @brief Represents a single simulated sensor device in a distributed network.

    Each device manages its own sensor data, interacts with a supervisor,
    and executes assigned scripts in a multi-threaded environment. It uses
    a dedicated `DeviceThread` which, in turn, spawns `Worker` threads for
    script execution. Synchronization across devices is handled by a
    `ReusableBarrier` based on condition variables.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        Sets up the device's unique identifier, its initial sensor data,
        a reference to the central supervisor, and initializes various
        synchronization primitives and state variables required for
        multi-threaded operation.

        @param device_id: A unique integer identifier for the device.
        @param sensor_data: A dictionary containing the device's initial sensor readings.
                            Keys are location IDs, values are sensor data.
        @param supervisor: A reference to the Supervisor object managing the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # Stores scripts assigned to this device for execution. Each script is (script_object, location_id).
        # Synchronization primitive: Event to signal that all scripts for the current timepoint have been assigned.
        self.scripts_done = Event()
        # Synchronization primitive: A per-device lock to protect access to its sensor data during read/write operations
        # by worker threads or other devices.
        self.my_lock = Lock()

        # Shared resource: A dictionary of locks, one for each location across all devices,
        # to ensure thread-safe access to specific sensor data locations.
        self.locations = None
        # Shared resource: A barrier to synchronize all devices at specific points in the simulation.
        self.barrier = None

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the device's awareness of all other devices in the simulation
               and initializes shared synchronization primitives.

        This method is called once at the beginning of the simulation.
        If this is the master device (device_id == 0), it initializes a shared
        `locations` dictionary of locks and a `ReusableBarrier`. Otherwise, it
        inherits these from the master device. It also starts its `DeviceThread`.

        @param devices: A list of all Device objects participating in the simulation.
        """
        # Block Logic: The master device (device_id == 0) initializes shared resources,
        # while other devices inherit them from the master.
        if self.device_id is 0:
            self.locations = {} # Initialize shared dictionary for location-specific locks.
            self.barrier = ReusableBarrier(len(devices)) # Initialize shared barrier for all devices.
            # Block Logic: Populate the shared 'locations' dictionary with locks for all known sensor data locations.
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass # Location already has a lock.
                else:
                    self.locations[loc] = Lock() # Assign a new lock for this location.
        else:
            # Block Logic: Non-master devices inherit shared resources from the master device.
            self.locations = devices[0].locations
            self.barrier = devices[0].get_barrier()
            # Block Logic: Populate the shared 'locations' dictionary with locks for all known sensor data locations.
            for loc in self.sensor_data:
                if loc in self.locations:
                    pass # Location already has a lock.
                else:
                    self.locations[loc] = Lock() # Assign a new lock for this location.

        # Spawns a dedicated thread to manage this device's operations.
        self.thread = DeviceThread(self, self.barrier, self.locations)
        self.thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device to be executed at a specific location
               or signals the completion of script assignments if no script is provided.

        @param script: The script object to be executed, or None if the timepoint is done.
        @param location: The location ID associated with the script, or irrelevant if script is None.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Block Logic: If no script is provided (script is None), it signifies that
            # all scripts for the current timepoint have been assigned.
            self.scripts_done.set() # Signals that script assignments are complete for the timepoint.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The location ID for which to retrieve data.
        @return The sensor data for the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.

        This method directly updates the `sensor_data` dictionary for the given location.

        @param location: The location ID for which to set data.
        @param data: The new sensor data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown process for the device by joining its associated thread.

        This ensures that the `DeviceThread` completes its execution before
        the program exits.
        """
        self.thread.join()

    def get_barrier(self):
        """
        @brief Returns the shared ReusableBarrier object.

        This method is primarily used by other devices to obtain a reference
        to the barrier initialized by the master device.

        @return The ReusableBarrier object.
        """
        return self.barrier



class DeviceThread(Thread):
    """
    @brief Manages the lifecycle and operation of a Device, including orchestrating `Worker` threads.

    This thread is responsible for handling timepoint progression, retrieving neighbor
    information from the supervisor, spawning and managing `Worker` threads for script
    execution, and ensuring global synchronization using the shared `ReusableBarrier`.
    """

    def __init__(self, device, barrier, locations):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: The Device object that this thread belongs to.
        @param barrier: The shared ReusableBarrier object for global synchronization.
        @param locations: The shared dictionary of locks for sensor data locations.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.locations = locations

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        It continuously retrieves neighbor information, waits for scripts to be
        assigned, dispatches scripts to `Worker` threads, waits for their completion,
        and then synchronizes with other devices using a shared barrier.
        """
        while True:
            # Block Logic: Retrieves information about neighboring devices from the supervisor.
            # This information is crucial for scripts that need to interact with or gather
            # data from adjacent devices in the simulated environment.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If there are no more neighbors (end of simulation), break the loop.
            if neighbours is None:
                break

            # Pre-condition: Waits until the `Device` signals that all scripts have been assigned
            # for the current timepoint.
            self.device.scripts_done.wait()
            self.device.scripts_done.clear() # Clears the event for the next timepoint.
            
            # Block Logic: Creates and starts `Worker` threads for each assigned script.
            # Each worker thread is responsible for executing one script.
            workers = []
            for (script, location) in self.device.scripts:
                w = Worker(self.device, neighbours, script, location, self.locations)
                workers.append(w)
                w.start()

            # Block Logic: Waits for all spawned `Worker` threads to complete their execution.
            for w in workers:
                w.join()

            # Synchronization point: All devices wait here until every other device
            # has completed its current timepoint processing. This ensures that
            # all devices are synchronized before moving to the next timepoint.
            self.barrier.wait()

class Worker(Thread):
    """
    @brief A dedicated worker thread (`Worker`) for executing individual scripts.

    Each `Worker` thread is responsible for acquiring a location-specific lock,
    gathering sensor data from its parent `Device` and its neighbors, executing
    a provided script, and then updating the sensor data based on the script's result.
    """
    def __init__(self, device, neighbours, script, location, locations):
        """
        @brief Initializes a new Worker instance.

        @param device: The parent Device object to which this worker thread belongs.
        @param neighbours: A list of neighboring Device objects from which to collect data.
        @param script: The script object to be executed.
        @param location: The location ID associated with the script.
        @param locations: The shared dictionary of locks for sensor data locations.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.locations = locations

    def run(self):
        """
        @brief The main execution logic for the Worker thread.

        It acquires a lock for its assigned location, collects sensor data
        from the device and its neighbors (with per-device locks), executes
        the script with this data, and then updates the sensor data in all
        relevant devices (also with per-device locks).
        """
        # Block Logic: Acquires a location-specific lock to ensure exclusive access
        # to the sensor data at this particular location during script execution and data update.
        # Invariant: Only one thread can operate on data at a given 'location' at any time across all devices.
        self.locations[self.location].acquire()
        script_data = [] 
        
        # Block Logic: Gathers sensor data from neighboring devices for the specified location.
        # Each neighbor's data access is protected by its individual `my_lock`.
        for device in self.neighbours:
            device.my_lock.acquire() # Pre-condition: Acquire lock for neighbor device's data.
            data = device.get_data(self.location)
            device.my_lock.release() # Post-condition: Release lock for neighbor device's data.
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Gathers sensor data from the current device for the specified location.
        # Access to the current device's data is protected by its `my_lock`.
        self.device.my_lock.acquire() # Pre-condition: Acquire lock for current device's data.
        data = self.device.get_data(self.location)
        self.device.my_lock.release() # Post-condition: Release lock for current device's data.
        if data is not None:
            script_data.append(data)

        # Pre-condition: If valid script data was collected (either from neighbors or self).
        if script_data != []:
            # Functional Utility: Executes the assigned script with the collected sensor data.
            # The script's `run` method presumably contains the core logic for data processing.
            result = self.script.run(script_data)

            # Block Logic: Updates the sensor data for all neighboring devices with the script's result.
            # Each neighbor's data update is protected by its individual `my_lock`.
            for device in self.neighbours:
                device.my_lock.acquire() # Pre-condition: Acquire lock for neighbor device's data.
                device.set_data(self.location, result)
                device.my_lock.release() # Post-condition: Release lock for neighbor device's data.

            # Block Logic: Updates the sensor data for the current device with the script's result.
            # Access to the current device's data is protected by its `my_lock`.
            self.device.my_lock.acquire() # Pre-condition: Acquire lock for current device's data.
            self.device.set_data(self.location, result)
            self.device.my_lock.release() # Post-condition: Release lock for current device's data.
            
        # Post-condition: Releases the location-specific lock, allowing other threads to access this location.
        self.locations[self.location].release()



class ReusableBarrier():
    """
    @brief Implements a reusable barrier using a Condition variable for synchronizing a fixed number of threads.

    This barrier allows a specified number of threads to wait at a synchronization
    point and then proceed together. It is designed to be reusable for multiple
    synchronization cycles. It ensures that all threads reach the barrier before
    any of them are allowed to proceed.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrier instance.

        @param num_threads: The total number of threads that must reach the barrier
                            before any of them can proceed.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads     # Counter for threads currently at the barrier.
        self.cond = Condition()                   # Condition variable for thread synchronization.
 
    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this barrier.

        The first `num_threads - 1` threads to call `wait` will block. The last thread
        will unblock all waiting threads and reset the barrier for reuse.
        """
        # Block Logic: Acquires the condition variable's intrinsic lock to protect shared state.
        self.cond.acquire()                      
        self.count_threads -= 1; # Decrement the counter for arriving threads.
        # Pre-condition: If this is the last thread to arrive at the barrier.
        if self.count_threads == 0:
            self.cond.notify_all()               # Functional Utility: Unblocks all threads currently waiting on this condition.
            self.count_threads = self.num_threads # Resets the counter for the next use of the barrier.
        else:
            self.cond.wait();                    # Functional Utility: Blocks the current thread until notified.
        self.cond.release();                     # Post-condition: Releases the condition variable's intrinsic lock.