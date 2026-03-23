"""
@file device.py
@brief Implements a distributed device simulation framework where devices process data
       and interact with neighboring devices.
       This module defines the core components: `Device`, `DeviceThread`, and `ScriptThread`.

Algorithm:
- Devices are initialized with an ID, sensor data, and a supervisor.
- They set up synchronization barriers and per-location locks for concurrent access.
- Scripts are assigned to devices and processed by a pool of worker threads (`ScriptThread`).
- Each `ScriptThread` acquires a lock for a data location, gathers data from its device
  and neighbors, executes an assigned script, and updates the data.
- A `DeviceThread` manages the lifecycle of script execution and synchronization
  among `ScriptThread`s and with other devices via a barrier.

Time Complexity:
- Overall complexity is dependent on the number of devices (N), the number of scripts (S),
  and the complexity of individual scripts.
- Data gathering and distribution involve communication with neighbors, which can
  add to complexity based on network topology.

Space Complexity:
- O(N * (D + S)) where N is the number of devices, D is the size of sensor data
  per device, and S is the average size of scripts. Additional space is used
  for thread management and synchronization primitives.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem as Barrier

class Device(object):
    """
    @class Device
    @brief Represents a single simulated device in the distributed system.
           Manages its local sensor data, assigned scripts, and synchronization mechanisms.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id: Unique identifier for the device.
        @param sensor_data: A dictionary containing sensor readings/data specific to this device.
        @param supervisor: The central supervisor object managing all devices.
        """
        # Event to signal when new scripts have been assigned to the device.
        self.timepoint_done = None

        # Dictionary to hold locks, typically one per data location, to prevent race conditions.
        self.lock = None

        # List to temporarily store scripts that are ready to be processed by ScriptThreads.
        self.todo_scripts = []

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives (barrier and locks) among devices.
        @param devices: A list of all Device instances in the simulation.
        Functional Utility: Ensures all devices share the same barrier and lock
                            dictionaries for consistent global synchronization.
        """
        # Block Logic: Initializes the barrier if it hasn't been set up yet.
        #              The barrier ensures all devices synchronize at certain timepoints.
        if self.timepoint_done is None:
            self.timepoint_done = Barrier(len(devices))
            # Invariant: All devices in the list will point to the same shared barrier instance.
            for device in devices:
                if device.timepoint_done is None:
                    device.timepoint_done = self.timepoint_done

        # Block Logic: Initializes the shared lock dictionary if it hasn't been set up yet.
        #              This dictionary will store locks for different data locations.
        if self.lock is None:
            self.lock = {}
            # Invariant: All devices in the list will point to the same shared lock dictionary.
            for device in devices:
                if device.lock is None:
                    device.lock = self.lock

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device to be executed at a specific data location.
        @param script: The script object to be executed.
        @param location: The data location (key) the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            
            # Block Logic: Creates a new lock for the given location if one doesn't already exist.
            #              Ensures exclusive access to data at this location during script execution.
            if location not in self.lock:
                self.lock[location] = Lock()
        else:
            # If no script is provided (script is None), it signals that no more scripts are expected.
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        @param location: The key corresponding to the data location.
        @return The data at the specified location, or None if the location does not exist.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a specific location.
        @param location: The key corresponding to the data location.
        @param data: The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread.
        Functional Utility: Ensures proper termination and cleanup of the DeviceThread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief Manages the execution lifecycle of scripts for a single device.
           It spawns `ScriptThread`s and orchestrates synchronization.
    """
    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device: The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        Functional Utility: Continuously fetches neighbors, waits for script assignments,
                            distributes scripts to worker threads, and synchronizes.
        """
        while True:
            # Block Logic: Fetches the current neighbors of the device from the supervisor.
            #              If no neighbors are returned (e.g., simulation ended), the loop breaks.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits until the `script_received` event is set, indicating
            #              that new scripts have been assigned to the device.
            self.device.script_received.wait()

            # Block Logic: Transfers all newly assigned scripts from `device.scripts`
            #              to `device.todo_scripts` for worker threads to pick up.
            # Invariant: After this loop, `device.scripts` is empty and `device.todo_scripts`
            #            contains all scripts for the current timepoint.
            for script in self.device.scripts:
                self.device.todo_scripts.append(script)

            # Determines the number of worker (ScriptThread) subthreads to create.
            # Capped at 8 to manage concurrency.
            nr_subthreads = min(8, len(self.device.scripts))
            
            subthreads = [] # List to hold the worker thread instances.
            
            # Lock to protect access to `device.todo_scripts` when multiple worker threads
            # are trying to pop scripts from it.
            scripts_lock = Lock()

            # Block Logic: Spawns `nr_subthreads` ScriptThread instances.
            # Invariant: `subthreads` list contains initialized (but not yet started)
            #            worker threads.
            while len(subthreads) < nr_subthreads:
                # Creates a new ScriptThread, passing shared resources and its parent (this thread).
                subthread = ScriptThread(scripts_lock, self, neighbours)
                subthreads.append(subthread)

            # Block Logic: Starts all spawned ScriptThread worker threads concurrently.
            for subthread in subthreads:
                subthread.start()

            # Block Logic: Waits for all ScriptThread worker threads to complete their execution.
            # Functional Utility: Ensures all scripts for the current timepoint are processed.
            for subthread in subthreads:
                subthread.join()

            # Block Logic: Clears the `script_received` event, resetting it to wait for the next batch of scripts.
            self.device.script_received.clear()

            # Block Logic: Waits at a shared barrier until all devices have completed their
            #              processing for the current timepoint.
            # Functional Utility: Ensures global synchronization before proceeding to the next time step.
            self.device.timepoint_done.wait()

class ScriptThread(Thread):
    """
    @class ScriptThread
    @brief A worker thread responsible for executing a single script at a specific data location.
           Handles data gathering, script execution, and data updates, ensuring data consistency with locks.
    """
    def __init__(self, scripts_lock, parent, neighbours):
        """
        @brief Initializes a ScriptThread.
        @param scripts_lock: A lock to protect the shared `todo_scripts` list in the parent.
        @param parent: The parent `DeviceThread` instance.
        @param neighbours: A list of neighboring Device instances to interact with.
        """
        Thread.__init__(self)
        self.scripts_lock = scripts_lock
        self.parent = parent
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution loop for the ScriptThread.
        Functional Utility: Continuously picks up scripts from the `todo_scripts` list,
                            executes them on relevant data, and updates device data.
        """
        # Block Logic: Safely retrieves one script from the `todo_scripts` list using a lock.
        # Invariant: `current_script` will be a (script, location) tuple if available, else None.
        self.scripts_lock.acquire()
        length = len(self.parent.device.todo_scripts)
        if length > 0:
            current_script = self.parent.device.todo_scripts.pop()
        else:
            current_script = None
        self.scripts_lock.release()

        # Block Logic: Continues processing scripts as long as there are scripts available
        #              and the device has neighbors to interact with.
        while current_script is not None and self.neighbours is not None:
            
            (script, location) = current_script # Unpacks the script and its associated data location.
            
            script_data = [] # List to accumulate data for the current script execution.

            # Block Logic: Acquires a lock specific to the data `location` to ensure
            #              exclusive access during data gathering and updating.
            self.parent.device.lock[location].acquire()

            # Block Logic: Gathers data from all neighboring devices for the current `location`.
            # Invariant: `script_data` will contain relevant data from neighbors.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Gathers data from the current device itself for the current `location`.
            data = self.parent.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: If any data was collected, execute the script and update device(s) data.
            if script_data != []:
                # Executes the assigned script with the collected data.
                result = script.run(script_data)

                # Block Logic: Updates the data on all neighboring devices with the script's result.
                for device in self.neighbours:
                    device.set_data(location, result)
                
                # Block Logic: Updates the data on the current device with the script's result.
                self.parent.device.set_data(location, result)

            # Releases the lock for the current data `location`, allowing other threads/devices to access it.
            self.parent.device.lock[location].release()

            # Block Logic: Safely retrieves the next script from the `todo_scripts` list,
            #              ensuring thread-safe access.
            self.scripts_lock.acquire()
            length = len(self.parent.device.todo_scripts)
            if length > 0:
                current_script = self.parent.device.todo_scripts.pop()
            else:
                current_script = None
            self.scripts_lock.release()
