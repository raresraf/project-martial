"""
@file device.py
@brief Implements a distributed device simulation framework where each device processes
       its assigned scripts using a local pool of worker threads. Global synchronization
       across all devices is managed by a conditional barrier, and data consistency
       for specific locations is maintained through a centralized dictionary of locks.

Algorithm:
- **Centralized Setup by Device 0:** The device with `device_id == 0` is responsible
  for initializing a single `ReusableBarrierCond` for all devices and a shared,
  empty dictionary `dict_location` that will dynamically hold `Lock` objects for
  each unique data location.
- **Dynamic Global Locks:** When a script is assigned to a device (via `assign_script`),
  if its `location` does not yet have an entry in the shared `dict_location`, a new
  `Lock` object is created and added for that location.
- **DeviceThread Lifecycle:**
    1. Each `DeviceThread` continuously fetches `neighbours` from a supervisor.
    2. It waits for `self.device.timepoint_done` (triggered by `assign_script(None, ...)`)
       to signal that scripts are ready for processing.
    3. It creates a fixed number (8) of `Worker` threads for this timepoint.
    4. It distributes its `self.device.scripts` among these `Worker` threads in a
       round-robin fashion.
    5. It starts all `Worker` threads and waits for them to complete using `join()`.
    6. It then clears `self.device.timepoint_done` and waits at the global `barrier`
       before starting the next time step.
- **Worker Thread Execution:**
    1. Each `Worker` processes the scripts it was assigned by its `DeviceThread`.
    2. For each script, it acquires the global lock for the specific data `location`
       from the shared `dict_location`.
    3. It gathers data from its `self_device` and `neighbours` for that `location`.
    4. It executes the `script.run()` method with the collected data.
    5. It updates the data on `self_device` and `neighbours` using the script's result.
    6. Finally, it releases the global lock for the `location`.

Time Complexity:
- The overall time complexity is highly dependent on:
    - The number of devices (N).
    - The number and complexity of scripts (S) per device.
    - The network topology (number of neighbors per device).
    - The number of worker threads in each device's pool (fixed at 8 here).
- Script execution is parallelized by the per-device worker pools.
- `ReusableBarrierCond.wait()` is a global synchronization point, which can be a bottleneck
  if any device's processing takes significantly longer.

Space Complexity:
- O(N_devices * (D + S)) for storing device-specific sensor data (D) and scripts (S).
- Additional overhead for:
    - `dict_location`: Stores `Lock` objects, growing with the number of unique data locations.
    - `ReusableBarrierCond` internal structures.
    - `DeviceThread`s and `Worker` threads' internal states and stacks.
"""

from threading import Event, Thread, Lock # Import necessary threading primitives.
import cond_barrier                       # Custom module for ReusableBarrierCond.


class Device(object):
    """
    @class Device
    @brief Represents a single simulated device in the distributed system.
           Manages its local sensor data, assigned scripts, and synchronization mechanisms.
           It interacts with a central supervisor and coordinates its operations through a dedicated thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id: Unique identifier for this device.
        @param sensor_data: A dictionary containing sensor readings/data specific to this device.
        @param supervisor: The central supervisor object responsible for managing all devices.
        """
        self.device_id = device_id     # Unique identifier for the device.
        self.sensor_data = sensor_data # Local sensor data (dict: {'location_id': data_value}).
        self.supervisor = supervisor   # Reference to the supervisor object.

        self.script_received = Event() # Event to signal when new scripts have been assigned.
        self.scripts = []              # List to store assigned scripts (script_obj, location_id).
        self.timepoint_done = Event()  # Event to signal the DeviceThread that scripts are ready for a timepoint.
        
        self.thread = DeviceThread(self) # Create and start the dedicated thread for this device.
        self.thread.start()
        
        self.dict_location = {}        # Reference to the shared dictionary of global locks per data location.
        self.barrier = None            # Reference to the shared ReusableBarrierCond for global synchronization.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives among all devices.
               Device 0 initializes the global barrier and `dict_location`,
               and propagates them to all other devices.
        @param devices: A list of all Device instances in the simulation.
        Functional Utility: Coordinates the initialization of global threading and
                            synchronization components to ensure consistent behavior across devices.
        """
        # Block Logic: Device 0 acts as the coordinator to initialize global shared resources.
        if self.device_id == 0:
            num_threads = len(devices) # Total number of devices participating in the barrier.
            
            self.barrier = cond_barrier.ReusableBarrierCond(num_threads) # Initialize the reusable conditional barrier.
            # Block Logic: Propagate the shared barrier and `dict_location` to all devices.
            # Invariant: All devices will share the same `barrier` and `dict_location` instances.
            for device in devices:
                device.barrier = self.barrier
                device.dict_location = self.dict_location # All devices reference the same global lock dictionary.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device to be executed at a specific data location.
               Dynamically creates a global lock for the location if it doesn't exist.
        @param script: The script object to be executed (expected to have a `run` method).
        @param location: The data location (key) the script operates on.
        Functional Utility: Queues scripts for processing and ensures that a global lock
                            exists for the data location to maintain consistency.
        """
        # Block Logic: If a lock for this location doesn't exist in the shared dictionary, create it.
        if location not in self.dict_location:
            self.dict_location[location] = Lock() # Create a new global lock for this data location.

        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
            self.script_received.set() # Signal the DeviceThread that scripts are received.
        else:
            # If `script` is None, it acts as a signal that all scripts for the current
            # timepoint have been assigned, allowing the DeviceThread to proceed.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
               Note: This implementation does not use a local lock (`self.slock` from previous versions)
               to protect `self.sensor_data`. Data consistency is expected to be managed by global
               locks in `dict_location` when accessed by Workers.
        @param location: The key corresponding to the data location.
        @return The data at the specified location, or None if the location does not exist.
        Functional Utility: Provides access to the device's local sensor data.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a specific location.
               Note: This implementation does not use a local lock to protect `self.sensor_data`.
               Data consistency is expected to be managed by global locks in `dict_location`.
        @param location: The key corresponding to the data location.
        @param data: The new data value to set.
        Functional Utility: Modifies the device's local sensor data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's thread.
        Functional Utility: Ensures proper termination and cleanup of the `DeviceThread`.
        """
        self.thread.join() # Waits for the DeviceThread to complete its execution.


class Worker(Thread):
    """
    @class Worker
    @brief A thread that executes a subset of scripts assigned to a `Device` for a given timepoint.
           It operates on specified data locations, gathers data from neighbors, runs scripts,
           and updates results, all while respecting global locks for data consistency.
    """

    def __init__(self, worker_id, neighbours, device, dict_location):
        """
        @brief Initializes a Worker thread.
        @param worker_id: A unique identifier for this worker thread.
        @param neighbours: A list of neighboring Device instances for data exchange.
        @param device: The parent `Device` instance this worker belongs to.
        @param dict_location: Shared dictionary of global `Lock` objects for data locations.
        """
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.worker_id = worker_id       # Unique identifier for the worker.
        self.neighbours = neighbours     # List of neighbor devices.
        self.device = device             # Reference to the parent Device.
        self.dict_location = dict_location # Shared global data locks.
        self.scripts = []                # List of scripts specifically assigned to this worker.
        self.location = []               # List of data locations corresponding to `self.scripts`.

    def addwork(self, script, location):
        """
        @brief Adds a script and its corresponding location to this worker's task list.
        @param script: The script object to be executed.
        @param location: The data location the script operates on.
        Functional Utility: Allows the `DeviceThread` to assign work to specific `Worker` instances.
        """
        self.scripts.append(script)     # Add the script to this worker's list.
        self.location.append(location) # Add the location corresponding to the script.

    def run(self):
        """
        @brief The main execution loop for the Worker thread.
        Functional Utility: Processes all scripts assigned to it, ensuring proper global
                            locking for data consistency during script execution and data updates.
        """
        i = 0 # Index to iterate through assigned scripts and locations.

        # Block Logic: Iterates through each script assigned to this worker.
        for script in self.scripts:
            # Block Logic: Acquires the global lock for the specific data `location`.
            #              This ensures exclusive access to data at this `location` across all devices/workers.
            self.dict_location[self.location[i]].acquire()

            script_data = [] # List to aggregate data for the current script execution.
            
            # Block Logic: Gathers data from all neighboring devices for the specified `location`.
            # Invariant: `script_data` will contain relevant data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location[i]) # Retrieves data using the neighbor's `get_data`.
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Gathers data from the current device (`self.device`) for the specified `location`.
            data = self.device.get_data(self.location[i]) # Retrieves data using the current device's `get_data`.
            if data is not None:
                script_data.append(data)

            # Block Logic: If any data was collected, execute the script and update device(s) data.
            if script_data != []:
                # Executes the assigned script with the collected data.
                result = script.run(script_data)

                # Block Logic: Updates the data on all neighboring devices with the script's result.
                for device in self.neighbours:
                    device.set_data(self.location[i], result)
                
                # Block Logic: Updates the data on the current device (`self.device`) with the script's result.
                self.device.set_data(self.location[i], result)
            
            # Releases the global lock for the current data `location`.
            self.dict_location[self.location[i]].release()
            i = i + 1 # Move to the next script/location.


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief Manages the execution lifecycle of scripts for a single device within the
           distributed simulation. It interacts with the supervisor, creates and
           manages its own pool of `Worker` threads, and synchronizes with the
           global conditional barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device: The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device # Reference to the associated Device instance.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        Functional Utility: Continuously synchronizes with other devices, fetches neighbors,
                            waits for scripts, distributes them to `Worker` threads,
                            waits for workers to complete, and prepares for the next timepoint.
        """
        # Block Logic: The main loop for the device's operational thread.
        # Invariant: The device continuously processes time steps until a shutdown signal is received.
        while True:
            # Block Logic: Fetches the current neighbors of the device from the supervisor.
            #              The supervisor determines the current set of active neighbors or signals shutdown.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Exit the main device loop if no neighbors (e.g., simulation ended).

            # Block Logic: Waits until the `timepoint_done` event is set.
            #              This event is typically set by `assign_script(None, ...)` or
            #              similar mechanism indicating all scripts are assigned for a timepoint.
            self.device.timepoint_done.wait()

            nr_worker = 0     # Counter for distributing scripts among workers.
            num_threads = 8   # Fixed number of worker threads for each DeviceThread.
            workers = []      # List to hold worker thread instances for this timepoint.

            # Block Logic: Creates and initializes `num_threads` Worker instances for the current timepoint.
            for i in range(num_threads): # Using range for Python 3 compatibility.
                lock_loc = self.device.dict_location # All workers share the same global lock dictionary.
                workers.append(Worker(i, neighbours, self.device, lock_loc))

            # Block Logic: Distributes scripts assigned to this device among the worker threads
            #              in a round-robin fashion.
            # Invariant: Each worker is assigned a subset of the device's scripts.
            for (script, location) in self.device.scripts:
                workers[nr_worker].addwork(script, location)
                nr_worker = nr_worker + 1
                if nr_worker == 8: # If 8 workers are assigned, reset to 0 for round-robin.
                    nr_worker = 0

            # Block Logic: Starts all created worker threads and waits for them to complete.
            # Functional Utility: Executes all assigned scripts in parallel within this device.
            for i in range(num_threads):
                workers[i].start() # Start each worker thread.
            for i in range(num_threads):
                workers[i].join()  # Wait for each worker thread to finish.

            # Resets the `timepoint_done` event for the next cycle.
            self.device.timepoint_done.clear()
            
            # Functional Utility: Synchronizes all devices at the reusable conditional barrier.
            #                     All devices must reach this point before any can proceed.
            self.device.barrier.wait()
