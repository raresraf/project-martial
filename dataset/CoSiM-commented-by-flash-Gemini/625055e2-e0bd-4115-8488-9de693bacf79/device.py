"""
@625055e2-e0bd-4115-8488-9de693bacf79/device.py
@brief Implements a distributed simulation or data processing system using a multi-threaded architecture.

This module defines three core classes:
- `Device`: Represents a computational node in the distributed system, managing its own sensor data,
  and orchestrating worker threads for script execution.
- `DeviceThread`: The main thread for a `Device` instance, responsible for managing timepoints,
  distributing scripts to worker threads, and synchronizing with a supervisor.
- `DeviceWorkerThread`: Executes assigned scripts on specific data locations, handling data
  access and synchronization using locks.

The system uses `threading.Event` for event signaling, `threading.Thread` for concurrency,
and `threading.Lock` for protecting shared data access, particularly during read/write
operations on sensor data and location-specific locks. A `ReusableBarrierCond` is employed
for timepoint synchronization across multiple devices.

Algorithm:
- Decentralized processing: Each `Device` operates semi-autonomously.
- Timepoint synchronization: Devices synchronize at discrete timepoints using a barrier.
- Concurrent script execution: `DeviceWorkerThread`s execute scripts in parallel.
- Distributed locking: Location-specific locks ensure data consistency across devices.
- Load balancing: Scripts are spread among worker threads.

Time Complexity:
- `__init__`: O(1)
- `setup_devices`: O(D * L) where D is number of devices and L is number of locations.
- `run` (DeviceThread): O(T * W * S * N * L) where T is timepoints, W is worker threads,
  S is scripts per worker, N is neighbors, L is locations.
- `run` (DeviceWorkerThread): O(S * N * L)
Space Complexity:
- `Device`: O(L) for sensor_data and locks_locations, O(W) for worker_threads.
- `DeviceThread`: O(W) for worker_threads.
- `DeviceWorkerThread`: O(S) for assigned_scripts.
"""

from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrierCond

class Device(object):
    """
    @brief Represents a single computational device (node) in the distributed system.

    Each device manages its own sensor data, orchestrates its processing via
    worker threads, and communicates with a supervisor and other devices
    for data exchange and synchronization. It maintains state related to
    assigned scripts, timepoint completion, and locks for data consistency.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: Unique identifier for the device.
        @param sensor_data: Dictionary containing initial sensor data for various locations.
        @param supervisor: Reference to the supervisor object for inter-device communication.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.lock_data = Lock()
        self.locks_locations = {}
        self.barrier = None
        self.worker_threads_no = 8
        self.worker_threads = []

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures shared resources (barrier and locks) across all devices.

        This method is designed to be called by a single designated device (device_id 0)
        to initialize global synchronization primitives and distribute them.

        @param devices: A list of all Device instances in the system.
        """
        

        
        # Block Logic: Only the device with ID 0 is responsible for setting up global resources.
        if self.device_id == 0:

            # Initializes a reusable barrier for all devices to synchronize timepoints.
            self.barrier = ReusableBarrierCond(len(devices))

            # Block Logic: Collects all unique data locations across all devices and sets up a lock for each.
            all_locations = []
            for device in devices:
                for location in device.sensor_data:
                    if location not in all_locations:
                        all_locations.append(location)
                        self.set_lock_on_location(location)

            # Block Logic: Distributes the initialized locks and barrier to all other devices.
            for device in devices:
                device.set_locks_locations(self.locks_locations)
                device.set_barrier(self.barrier)

    def set_barrier(self, pbarrier):
        """
        @brief Sets the shared synchronization barrier for this device.

        @param pbarrier: The ReusableBarrierCond instance to be used by this device.
        """
        
        self.barrier = pbarrier


    def set_lock_on_location(self, plocation):
        """
        @brief Creates and assigns a new lock for a specific data location.

        @param plocation: The identifier for the data location.
        """
        
        self.locks_locations[plocation] = Lock()

    def set_locks_locations(self, plocks):
        """
        @brief Sets the dictionary of shared location locks for this device.

        @param plocks: A dictionary where keys are locations and values are Lock objects.
        """
        
        self.locks_locations = plocks

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script is provided, it's added to the device's script queue, and
        the `script_received` event is set to signal the DeviceThread. If no
        script is provided (None), it signals that all scripts for the current
        timepoint are processed, setting `timepoint_done`.

        @param script: The script object to execute, or None to signal timepoint completion.
        @param location: The data location relevant to the script.
        """
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a new script has been received.
        else:
            self.timepoint_done.set() # Signal that all scripts for this timepoint have been assigned.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The identifier for the data location.
        @return The sensor data for the location, or None if the location is not found.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets (updates) sensor data for a given location.

        @param location: The identifier for the data location.
        @param data: The new data to set for the location.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its main thread.
        """
        
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the lifecycle and script execution for a single Device instance.

    This thread is responsible for advancing simulation timepoints, receiving
    new scripts, distributing them to a pool of worker threads, and synchronizing
    with other devices via a shared barrier.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread.

        @param device: The Device instance that this thread manages.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def spread_scripts_to_threads(self):
        """
        @brief Distributes assigned scripts among the DeviceWorkerThread instances.

        Each script from `self.device.scripts` is assigned to a worker thread
        in a round-robin fashion.
        """
        
        script_no = 0
        for (script, location) in self.device.scripts:
            thread_idx = script_no % self.device.worker_threads_no
            self.device.worker_threads[thread_idx].add_script(script, location)
            script_no += 1

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Continuously monitors for new timepoints, orchestrates script execution
        by worker threads, and synchronizes with other devices.
        """

        while True:
            # Block Logic: Get neighbors from the supervisor. If no neighbors (e.g., simulation ended), break the loop.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Wait until all scripts for the current timepoint have been assigned to this device.
            self.device.timepoint_done.wait()

            # Block Logic: Create and initialize worker threads for the current timepoint.
            for i in range(self.device.worker_threads_no):
                worker_thread = DeviceWorkerThread(self, neighbours)
                self.device.worker_threads.append(worker_thread)

            # Block Logic: Distribute the scripts received by this device to its worker threads.
            self.spread_scripts_to_threads()

            # Block Logic: Start all worker threads.
            for i in range(len(self.device.worker_threads)):
                self.device.worker_threads[i].start()

            # Block Logic: Wait for all worker threads to complete their assigned scripts.
            for i in range(len(self.device.worker_threads)):
                self.device.worker_threads[i].join()

            # Block Logic: Clear the list of worker threads for the next timepoint.
            del self.device.worker_threads[:]

            # Block Logic: Reset the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()

            # Block Logic: Synchronize with other devices at the barrier, marking the end of the current timepoint.
            self.device.barrier.wait()


class DeviceWorkerThread(Thread):
    """
    @brief Worker thread responsible for executing scripts on specific data locations.

    Each worker thread is assigned a subset of scripts for a given timepoint.
    It handles acquiring and releasing necessary locks to ensure data consistency
    during script execution, especially when accessing shared sensor data.
    """
    
    def __init__(self, device_thread, neighbours):
        """
        @brief Initializes a new DeviceWorkerThread.

        @param device_thread: The master DeviceThread that created this worker.
        @param neighbours: A list of neighboring Device instances from which to retrieve data.
        """
        super(DeviceWorkerThread, self).__init__()
        self.master_thread = device_thread
        self.device_neighbours = neighbours
        self.assigned_scripts = []

    def add_script(self, script, location):
        """
        @brief Assigns a script to this worker thread for execution.

        @param script: The script object to be executed.
        @param location: The data location pertinent to this script.
        """
        
        if script is not None:
            self.assigned_scripts.append((script, location))

    def run(self):
        """
        @brief The main execution loop for the DeviceWorkerThread.

        Iterates through assigned scripts, collects data from the local device
        and its neighbors, executes the script, and then propagates the results.
        Ensures thread-safe data access using locks.
        """
        
        for (script, location) in self.assigned_scripts:
            # Block Logic: Acquire the location-specific lock to ensure exclusive access to data at this location.
            self.master_thread.device.locks_locations[location].acquire()

            script_data = []
            
            # Block Logic: Collect data from neighboring devices for the specified location.
            # Each neighbor's data is acquired and released under its own data lock.
            for device in self.device_neighbours:
                
                device.lock_data.acquire() # Acquire lock for neighbor's data.
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
                device.lock_data.release() # Release lock for neighbor's data.

            
            # Block Logic: Collect data from the local device for the specified location.
            self.master_thread.device.lock_data.acquire() # Acquire lock for local device's data.
            data = self.master_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)
            self.master_thread.device.lock_data.release() # Release lock for local device's data.


            if script_data != []:
                # Block Logic: Execute the assigned script with the collected data.
                result = script.run(script_data)
                
                # Block Logic: Propagate the script's result back to neighboring devices.
                # Each neighbor's data is updated under its own data lock.
                for device in self.device_neighbours:
                    device.lock_data.acquire() # Acquire lock for neighbor's data.
                    device.set_data(location, result)
                    device.lock_data.release() # Release lock for neighbor's data.

                
                # Block Logic: Propagate the script's result back to the local device.
                self.master_thread.device.lock_data.acquire() # Acquire lock for local device's data.
                self.master_thread.device.set_data(location, result)
                self.master_thread.device.lock_data.release() # Release lock for local device's data.

            # Block Logic: Release the location-specific lock after script execution and data propagation.
            self.master_thread.device.locks_locations[location].release()
