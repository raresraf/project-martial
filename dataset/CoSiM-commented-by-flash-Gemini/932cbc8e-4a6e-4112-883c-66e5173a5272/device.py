


"""
@932cbc8e-4a6e-4112-883c-66e5173a5272/device.py
@brief Implements a multi-threaded simulation for distributed sensor devices using a producer-consumer pattern for scripts.

This module defines the core components for simulating a network of sensor devices,
each capable of executing scripts, managing local sensor data, and interacting
with a central supervisor. This version utilizes a dedicated pool of `Worker` threads
that consume script tasks from a shared `Queue` managed by `DeviceThread`.
Synchronization across devices is handled by a `ReusableBarrierSem`.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate with neighbors under the
guidance of a supervisor.

Classes:
- Device: Represents a single simulated sensor device.
- Worker: A dedicated thread that processes scripts from a shared queue.
- DeviceThread: Manages the lifecycle and operation of a Device, acting as a producer
                for script tasks to `Worker` threads.

Domain: Distributed Systems Simulation, Concurrent Programming, Producer-Consumer Pattern, Sensor Networks.
"""

from threading import Event, Thread, Lock
from Queue import Queue
from barrier import ReusableBarrierSem # Assuming ReusableBarrierSem is defined elsewhere or imported.

class Device(object):
    """
    @brief Represents a single simulated sensor device in a distributed network.

    Each device manages its own sensor data, interacts with a supervisor,
    and executes assigned scripts in a multi-threaded environment. This version
    uses a dedicated `DeviceThread` to feed scripts to a pool of `Worker` threads
    via a `Queue`. Synchronization across devices is handled by a `ReusableBarrierSem`.
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
        # Synchronization primitive: Event to signal that a new script has been assigned to the device.
        self.script_received = Event()
        self.scripts = [] # Stores scripts assigned to this device for execution. Each script is (script_object, location_id).
        self.devices = [] # List of all Device objects in the simulation. Populated in setup_devices.
        self.threads = [] # List of Worker threads associated with this device.
        self.barrier = None # Shared ReusableBarrierSem for inter-device synchronization.
        # Synchronization primitive: Event to signal that all scripts for the current timepoint have been assigned.
        self.timepoint_done = Event()
        # Shared resource: Queue used by DeviceThread to pass scripts to Worker threads.
        self.thread_queue = Queue()
        # Shared resource: A dictionary of locks, one for each location, to ensure thread-safe access to specific sensor data locations.
        self.locks = {}
        # Spawns a dedicated thread to manage this device's operations.
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
        @brief Sets up the device by initializing worker threads and configuring shared resources.

        This method is called once at the beginning of the simulation. It spawns
        a pool of `Worker` threads for the device and populates the list of all
        participating devices. It also initializes or inherits the shared
        `ReusableBarrierSem` and ensures location-specific locks are propagated.

        @param devices: A list of all Device objects participating in the simulation.
        """
        # Block Logic: Spawns a fixed number of `Worker` threads (8 in this case) for the device.
        for _ in range(8): # Invariant: 8 worker threads are created.
            thread = Worker(self)
            thread.start()
            self.threads.append(thread)
        
        # Block Logic: Populates the device's list of all other devices in the simulation.
        for device in devices:
            if device is not None:
                self.devices.append(device)
        
        # Block Logic: Initializes a shared `ReusableBarrierSem` if it's the master device (first device),
        # otherwise inherits it from an already initialized device.
        if self.barrier is None:
            self.barrier = ReusableBarrierSem(len(self.devices))
        
        # Block Logic: Propagates the shared barrier to all other devices.
        for device in self.devices:
            if device is not None:
                if device.barrier is None:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device to be executed at a specific location
               or signals the completion of script assignments if no script is provided.

        It also dynamically creates a lock for the specified location if one does not exist.

        @param script: The script object to be executed, or None if the timepoint is done.
        @param location: The location ID associated with the script, or irrelevant if script is None.
        """
        if script is not None:
            self.scripts.append((script, location))
            
            # Block Logic: Dynamically creates a lock for the given location if it doesn't already exist.
            if location is not None:
                if not self.locks.has_key(location):
                    self.locks[location] = Lock()
            self.script_received.set() # Signals that a new script has been assigned.
        else:
            # Block Logic: If no script is provided (script is None), it signifies that
            # all scripts for the current timepoint have been assigned.
            self.timepoint_done.set() # Signals that the timepoint processing is logically done (no more scripts to assign).

        # Block Logic: Propagates location locks to other devices that may not have them yet.
        for device in self.devices:
            if not device.locks.has_key(location):
                if self.locks.has_key(location):
                    device.locks[location] = self.locks[location]

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


class Worker(Thread):
    """
    @brief A dedicated worker thread that processes scripts from a shared queue.

    Each `Worker` thread continuously pulls script tasks from `device.thread_queue`.
    It then acquires a location-specific lock, gathers data from the device and its
    neighbors, executes the script, and updates sensor data.
    """
    def __init__(self, device):
        """
        @brief Initializes a new Worker instance.

        @param device: The parent Device object to which this worker thread belongs.
        """
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        @brief The main execution logic for the Worker thread.

        It continuously retrieves script tasks from the shared queue, acquires
        location-specific locks, collects data, executes the script, and updates
        sensor data. It also handles a sentinel value to gracefully terminate.
        """
        while True:
        	# Block Logic: Retrieves a script task from the shared queue.
            script_loc_neigh = self.device.thread_queue.get()
            # Pre-condition: Checks for a sentinel value (None, None, None) to signal termination.
            if script_loc_neigh[0] is None:
                if script_loc_neigh[1] is None:
                    if script_loc_neigh[2] is None:
                        self.device.thread_queue.task_done() # Signals that a task has been completed (the sentinel).
                        break # Terminate the worker thread.
            script_data = [] # List to accumulate sensor data for the script.
            
            # Block Logic: Acquires a location-specific lock to ensure exclusive access
            # to the sensor data at this particular location during script execution and data update.
            self.device.locks[script_loc_neigh[1]].acquire()
            
            # Block Logic: Gathers sensor data from neighboring devices for the specified location.
            for device in script_loc_neigh[2]:
                # Note: Assuming `get_data` handles its own locking or is read-only.
                data = device.get_data(script_loc_neigh[1])
                if data is not None:
                    script_data.append(data)

            # Block Logic: Gathers sensor data from the current device for the specified location.
            data = self.device.get_data(script_loc_neigh[1])
            if data is not None:
                script_data.append(data)
            
            # Pre-condition: If valid script data was collected.
            if script_data != []:
                # Functional Utility: Executes the assigned script with the collected sensor data.
                result = script_loc_neigh[0].run(script_data)
                
                # Block Logic: Updates the sensor data for all neighboring devices with the script's result.
                for device in script_loc_neigh[2]:
                    # Note: Assuming `set_data` handles its own locking or is atomic.
                    device.set_data(script_loc_neigh[1], result)
                
                # Block Logic: Updates the sensor data for the current device with the script's result.
                self.device.set_data(script_loc_neigh[1], result)

            self.device.locks[script_loc_neigh[1]].release() # Post-condition: Releases the location-specific lock.
            self.device.thread_queue.task_done() # Signals that the current script task has been completed.


class DeviceThread(Thread):
    """
    @brief Manages the lifecycle and operation of a Device, acting as a producer
           for script tasks to `Worker` threads.

    This thread is responsible for handling timepoint progression, retrieving neighbor
    information from the supervisor, queuing scripts for execution by `Worker` threads,
    and ensuring global synchronization using the shared `ReusableBarrierSem`.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: The Device object that this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        It continuously retrieves neighbor information, waits for scripts to be
        assigned, puts script tasks into a shared queue for `Worker` threads,
        waits for all tasks to be completed, and then synchronizes with other
        devices via the shared barrier. It also handles the shutdown of worker threads.
        """
        # Block Logic: Main simulation loop for processing timepoints.
        while True:
            # Block Logic: Retrieves information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If there are no more neighbors (end of simulation), break the loop.
            if neighbours is None:
                break

            # Pre-condition: Waits until the `Device` signals that all scripts for the current
            # timepoint have been assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Clears the event for the next timepoint.
            
            # Block Logic: Puts all assigned scripts as tasks into the shared queue for Worker threads.
            for (script, location) in self.device.scripts:
                self.device.thread_queue.put((script, location, neighbours))
            
            self.device.thread_queue.join() # Functional Utility: Blocks until all tasks in the queue are marked as done.
            
            # Synchronization point: All devices wait here until every other device
            # has completed its current timepoint processing. This ensures that
            # all devices are synchronized before moving to the next timepoint.
            self.device.barrier.wait()
        
        # Block Logic: Signals all worker threads to terminate by putting sentinel values into the queue.
        for _ in range(len(self.device.threads)):
            self.device.thread_queue.put((None, None, None))
        # Block Logic: Waits for all worker threads to gracefully terminate.
        for thread in self.device.threads:
            thread.join()
