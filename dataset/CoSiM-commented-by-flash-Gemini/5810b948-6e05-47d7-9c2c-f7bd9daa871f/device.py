

"""
This module implements a multi-threaded simulation of interconnected devices,
each capable of processing sensor data and interacting with a central supervisor.
It employs a reusable barrier for synchronization, work pools for script execution,
and fine-grained locking mechanisms for data consistency across shared resources.
"""

from threading import Event, Thread, Condition, Lock
from Queue import Queue

class ReusableBarrier(object):
    """
    A reusable synchronization barrier that allows a specified number of threads
    to wait until all have reached a common point before proceeding.
    """
    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        # Counter for threads currently waiting at the barrier.
        self.count_threads = self.num_threads
        # Condition variable for thread synchronization.
        self.cond = Condition()

    def wait(self):
        """
        Causes the calling thread to wait until all `num_threads` have reached this barrier.
        Once all threads arrive, they are all released simultaneously, and the barrier resets.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # Block Logic: Checks if the current thread is the last one to reach the barrier.
        # If true, it releases all waiting threads and resets the barrier for future use.
        # Otherwise, it waits for other threads to arrive.
        # Invariant: All threads either wait or are released together, and the barrier state is consistent.
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a simulated device in a distributed system, responsible for managing
    sensor data, executing scripts, and coordinating with a supervisor and other devices.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding sensor data, keyed by location.
            supervisor (Supervisor): A reference to the central supervisor managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.barrier = None # Reference to the shared ReusableBarrier for synchronization.
        self.locations_mutex = None # Mutex to protect access to the shared list of locations.
        self.can_begin = Event() # Event to signal when the device thread can begin its main loop.
        self.locks_computed = Event() # Event to signal when location-specific locks have been computed.
        self.timepoint_done = Event() # Event to signal when all scripts for a timepoint are processed.
        self.simulation_end = Event() # Event to signal the termination of the simulation.
        self.lock = Lock() # General purpose lock for the device.
        self.scripts_queue = Queue() # Queue for scripts waiting to be processed.
        self.scripts = [] # List of scripts assigned to the device.
        self.locations = [] # Shared list of all sensor data locations across devices.
        self.devices = [] # List of all devices in the simulation (set only for master device).
        self.thread = DeviceThread(self) # Dedicated thread for this device's operational logic.
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the shared resources and starts the device threads.
        This method is typically called by a designated master device (device_id == 0).

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        # Block Logic: Initializes shared synchronization primitives for all devices.
        # Invariant: Ensures that synchronization and work management infrastructure is set up once.
        if self.device_id == 0:
            # Initializes a reusable barrier for synchronizing all devices at specific timepoints.
            self.barrier = ReusableBarrier(len(devices))

            # Stores a reference to all devices for inter-device communication.
            self.devices = devices
            # Initializes a mutex to control access to the shared list of sensor data locations.
            self.locations_mutex = Lock()

            # Block Logic: Propagates shared resources and signals individual device threads to proceed.
            # Invariant: Each device is configured with necessary shared resources and signaled to begin.
            for device in devices:
                device.locations_mutex = self.locations_mutex
                device.locations = self.locations
                device.barrier = self.barrier
                device.can_begin.set()

            # Signals the current device's thread to begin operation.
            self.can_begin.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific sensor data location.
        If script is None, it signals the completion of script assignments for the current timepoint.

        Args:
            script (Script): The script object to be executed.
            location (int): The identifier for the sensor data location the script targets.
        """
        # Block Logic: Appends the script to the device's script list and queue if a script is provided.
        # Otherwise, it signals that script assignments for the current timepoint are complete.
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_queue.put((script, location))
        else:
            self.timepoint_done.set() # Signals that all scripts for the current timepoint have been assigned.

    def get_data(self, location):
        """
        Retrieves sensor data from a specified location on this device.

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or None if the location does not exist.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data at a specified location on this device.

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new data value to set.
        """
        # Block Logic: Updates the sensor data at the specified location if it exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the graceful shutdown of the device's operational thread.
        """
        # Block Logic: Waits for the device's thread to complete its execution before proceeding with shutdown.
        self.thread.join()

class DeviceThread(Thread):
    """
    Manages the lifecycle and operational logic of a single simulated device.
    It synchronizes with other devices, manages shared locations, and orchestrates
    the execution of scripts by delegating them to DeviceCore threads.
    """
    def __init__(self, device):
        """
        Initializes a new DeviceThread.

        Args:
            device (Device): The Device instance this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.current_neighbours = [] # List of neighboring devices for the current timepoint.
        # Creates a pool of DeviceCore threads, each acting as a processing unit for scripts.
        self.cores = [DeviceCore(self, i, self.device.simulation_end) for i in xrange(0, 8)]

    def run(self):
        """
        The main execution loop for the device thread.
        It manages the device's state, synchronizes with other devices,
        assigns scripts to its core workers, and handles simulation termination.
        """
        # Functional Utility: Waits for the 'can_begin' event to be set, allowing the device to start operations.
        self.device.can_begin.wait()

        # Block Logic: Acquires a mutex to safely add this device's sensor data locations to a shared list.
        # Invariant: The shared 'self.device.locations' list accurately reflects all unique locations across devices.
        self.device.locations_mutex.acquire()
        # Block Logic: Iterates through the sensor data keys of the current device.
        # If a location is not already in the shared list, it adds it.
        for location in self.device.sensor_data.keys():
            if location not in self.device.locations:
                self.device.locations.append(location)
        self.device.locations_mutex.release()

        # Functional Utility: Waits at the shared barrier to synchronize with other devices.
        self.device.barrier.wait()

        # Block Logic: If this is the master device (device_id == 0), it initializes locks for all shared locations
        # and propagates these locks to all other devices.
        # Invariant: All devices share the same set of location-specific locks for data consistency.
        if self.device.device_id == 0:
            # Initializes a list of locks, one for each unique sensor data location.
            self.device.locations_locks = [Lock() for _ in xrange(0, len(self.device.locations))]
            # Propagates the created location locks to all other devices.
            for device in self.device.devices:
                device.locations_locks = self.device.locations_locks

        # Functional Utility: Waits at the shared barrier again to ensure all devices have their locks set up.
        self.device.barrier.wait()

        # Functional Utility: Starts all DeviceCore threads, enabling them to begin processing.
        for core in self.cores:
            core.start()

        # Block Logic: Main loop for the device thread, continuously managing timepoints and script execution.
        # Invariant: The device processes timepoints, assigns tasks, and handles simulation termination.
        while True:

            # Block Logic: Clears any remaining scripts in the queue from a previous timepoint.
            # Invariant: The script queue is empty before processing new scripts for the current timepoint.
            while not self.device.scripts_queue.empty():
                self.device.scripts_queue.get()

            # Block Logic: Repopulates the script queue with the current timepoint's scripts.
            # Invariant: The script queue accurately reflects the tasks to be executed in the current timepoint.
            for script in self.device.scripts:
                self.device.scripts_queue.put(script)

            # Functional Utility: Retrieves the current set of neighboring devices from the supervisor.
            self.current_neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Checks if the simulation is meant to end (no more neighbors).
            # If so, it signals termination to all core workers and joins them before breaking the loop.
            if self.current_neighbours is None:
                # Signals all DeviceCore threads that the simulation has ended.
                self.device.simulation_end.set()
                # Ensures all core workers are aware of the simulation end.
                for core in self.cores:
                    core.got_script.set()
                # Waits for all core workers to finish their execution.
                for core in self.cores:
                    core.join()
                break

            # Block Logic: Continuously attempts to assign scripts from the queue to available DeviceCore workers.
            # It waits until all scripts for the current timepoint are assigned or processed.
            # Invariant: All scripts are either assigned to a core or the timepoint processing is complete.
            while not self.device.timepoint_done.isSet() or not self.device.scripts_queue.empty():
                # Checks if there are scripts in the queue to be processed.
                if not self.device.scripts_queue.empty():
                    script, location = self.device.scripts_queue.get()

                    core_found = False
                    # Block Logic: Searches for an available DeviceCore to assign the script.
                    # Invariant: A script is assigned to an available core, or the loop continues searching.
                    while not core_found:
                        for core in self.cores:
                            if core.running is False:
                                core_found = True
                                core.script = script
                                core.location = location
                                core.neighbours = self.current_neighbours
                                core.running = True
                                core.got_script.set() # Signals the core that a new script is available.
                                break

            # Functional Utility: Waits at the shared barrier to synchronize with other devices
            # after all scripts for the current timepoint have been assigned.
            self.device.barrier.wait()

            # Functional Utility: Resets the 'timepoint_done' event for the next timepoint.
            self.device.timepoint_done.clear()

class DeviceCore(Thread):
    """
    A core worker thread within a Device, responsible for executing a single script
    on sensor data and coordinating access to shared data locations.
    """
    def __init__(self, device_thread, core_id, simulation_end):
        """
        Initializes a new DeviceCore.

        Args:
            device_thread (DeviceThread): The parent DeviceThread managing this core.
            core_id (int): A unique identifier for this core within the device.
            simulation_end (Event): An Event to signal the termination of the simulation.
        """
        Thread.__init__(self, name="Device Core %d" % core_id)
        self.device_thread = device_thread
        self.core_id = core_id
        self.neighbours = [] # List of neighboring devices for script execution.
        self.got_script = Event() # Event to signal when a new script is assigned to this core.
        self.running = False # Flag indicating if the core is currently executing a script.
        self.simulation_end = simulation_end # Event to signal simulation termination.
        self.script = None # Placeholder for the script to be executed.
        self.location = None # Placeholder for the sensor data location.

    def run(self):
        """
        The main execution loop for the device core.
        It waits for scripts, executes them on sensor data, and manages data consistency.
        """
        # Block Logic: The core's main loop, continuously waiting for and processing scripts.
        # Invariant: The core remains active, processing assigned scripts until the simulation ends.
        while True:

            # Functional Utility: Waits until a new script is assigned to this core.
            self.got_script.wait()

            # Block Logic: Checks if the simulation has ended. If so, the core terminates its loop.
            if self.simulation_end.isSet():
                break

            # Acquires a lock specific to the sensor data location to prevent concurrent modifications.
            self.device_thread.device.locations_locks[self.location].acquire()

            script_data = []
            # Block Logic: Collects sensor data from neighboring devices for script execution.
            # Invariant: All available relevant sensor data from neighbors is gathered.
            for neighbour in self.neighbours:
                # Acquires a lock for the neighbor device to safely access its data.
                neighbour.lock.acquire()
                data = neighbour.get_data(self.location)
                neighbour.lock.release()
                if data is not None:
                    script_data.append(data)

            # Block Logic: Collects sensor data from the current device for script execution.
            # Invariant: The current device's relevant sensor data is gathered.
            self.device_thread.device.lock.acquire()
            data = self.device_thread.device.get_data(self.location)
            self.device_thread.device.lock.release()
            if data is not None:
                script_data.append(data)

            # Block Logic: Executes the script if data is available and propagates results.
            # Invariant: If a script runs, its output is used to update relevant sensor data.
            if script_data != []:
                # Executes the assigned script with the collected sensor data.
                result = self.script.run(script_data)

                # Block Logic: Updates the current device's sensor data with the script's result.
                self.device_thread.device.lock.acquire()
                self.device_thread.device.set_data(self.location, result)
                self.device_thread.device.lock.release()

                # Block Logic: Propagates the script's result to the relevant sensor locations on neighboring devices.
                # Invariant: Neighboring devices' sensor data is updated consistently with the script's output.
                for neighbour in self.neighbours:
                    neighbour.lock.acquire()
                    neighbour.set_data(self.location, result)
                    neighbour.lock.release()

            # Releases the lock on the sensor data location after updates are complete.
            self.device_thread.device.locations_locks[self.location].release()

            # Functional Utility: Marks the core as no longer running a script.
            self.running = False

            # Functional Utility: Resets the 'got_script' event, indicating readiness for a new script.
            self.got_script.clear()

            # Block Logic: Checks again if the simulation has ended, terminating the loop if true.
            if self.simulation_end.isSet():
                break
