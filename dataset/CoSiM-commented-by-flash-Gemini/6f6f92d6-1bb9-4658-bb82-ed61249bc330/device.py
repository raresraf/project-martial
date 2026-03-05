

"""
This module implements a device simulation framework with a master-worker
architecture using Python's `threading` and `Queue` modules. It defines
`Device` objects, a `DeviceThread` to manage script execution through
a pool of worker threads, and a `ReusableBarrierCond` for synchronization.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

import Queue

class Device(object):
    """
    Represents a simulated device in a distributed system. Each device manages
    its own sensor data, processes assigned scripts through a dedicated thread
    with a worker pool, and synchronizes its operations with other devices
    via a supervisor and shared synchronization primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor data, keyed by location.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        # Functional Utility: `is_available` acts as a lock to protect access to `sensor_data`,
        # ensuring thread-safe read/write operations.
        self.is_available = Lock()
        self.neighbours = [] # List of neighboring devices, updated by the master thread.
        self.supervisor = supervisor
        self.script_received = Event() # Signals that a script has been assigned.
        self.scripts = [] # List of (script, location) tuples assigned to this device.
        # Functional Utility: `script_queue` is a thread-safe queue used to distribute
        # script execution tasks to worker threads.
        self.script_queue = Queue.Queue()
        self.timepoint_done = Event() # Signals completion of script assignments for a timepoint.
        self.thread = DeviceThread(self) # The main thread for this device.
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the device's synchronization mechanisms.
        Device 0 initializes the shared barrier and per-location locks (if needed),
        which are then distributed to all other devices.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        if self.device_id == 0:
            # Functional Utility: Creates a `ReusableBarrierCond` instance for global synchronization
            # across all device threads.
            shared_barrier = ReusableBarrierCond(len(devices))
            # Functional Utility: Initializes an empty dictionary to store per-location locks.
            # Locks will be added as locations are encountered during script execution.
            location_lock = {}
            # Block Logic: Propagates the initialized barrier and shared `location_lock` dictionary
            # to all other devices.
            for device in devices:
                device.shared_barrier = shared_barrier
                device.location_lock = location_lock

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location for this device.
        If a script is provided, it is added to the list of scripts.
        If no script is provided (None), it signals that the timepoint is done.
        In either case, it signals that a script (or lack thereof) has been received.

        Args:
            script (Script or None): The script object to assign, or None if the timepoint is complete.
            location (int): The numerical identifier for the location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Functional Utility: Signals that all script assignments for the current timepoint are complete.
            self.timepoint_done.set()

        # Functional Utility: Signals to the device thread that a script (or end-of-timepoint)
        # has been received, allowing it to proceed with processing.
        self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location in a thread-safe manner.

        Args:
            location (int): The location for which to retrieve data.

        Returns:
            any: The sensor data for the specified location, or None if not found.
        """
        # Functional Utility: Acquires a lock to ensure exclusive read access to `sensor_data`.
        self.is_available.acquire()
        if location in self.sensor_data:
            data = self.sensor_data[location]
        else:
            data = None
        # Functional Utility: Releases the lock after data access.
        self.is_available.release()
        return data

    def set_data(self, location, data):
        """
        Sets sensor data for a given location in a thread-safe manner.

        Args:
            location (int): The location for which to set data.
            data (any): The new data to set.
        """
        # Functional Utility: Acquires a lock to ensure exclusive write access to `sensor_data`.
        self.is_available.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.is_available.release()

    def shutdown(self):
        """
        Shuts down the device thread, waiting for its completion.
        """
        self.thread.join()


class ScriptObject(object):
    """
    A simple data structure to encapsulate a script, its associated location,
    and a flag indicating whether its execution should halt the worker thread.
    """

    def __init__(self, script, location, stop_execution):
        """
        Initializes a new ScriptObject instance.

        Args:
            script (object): The script to be executed.
            location (int): The location pertaining to the script.
            stop_execution (bool): If True, signals the worker thread to stop after processing.
        """
        self.script = script
        self.location = location
        self.stop_execution = stop_execution

class DeviceThread(Thread):
    """
    The main thread for a Device. It is responsible for fetching neighbor
    information from the supervisor, distributing assigned scripts to a
    worker queue, coordinating script execution with a pool of worker threads,
    and handling synchronization at timepoints.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None # Stores the current list of neighboring devices.

        self.threads = [] # List to hold references to worker threads.
        # Block Logic: Spawns and starts a fixed number of generic worker threads
        # that will execute the `script_compute` method.
        for _ in range(8): # Creates 8 worker threads.
            worker_thread = Thread(target=self.script_compute)
            self.threads.append(worker_thread)
            worker_thread.start()

    def script_compute(self):
        """
        The target function for worker threads. It continuously retrieves
        `ScriptObject`s from the device's queue, executes the associated
        script, and updates sensor data while ensuring thread-safe access
        using per-location locks.
        """
        while True:
            # Block Logic: Retrieves a `ScriptObject` from the shared queue,
            # blocking until a task is available.
            script_object = self.device.script_queue.get()
            # Pre-condition: Checks if a stop signal has been received (via `stop_execution` flag).
            if script_object.stop_execution is True:
                # Functional Utility: Marks the task as done and breaks the loop,
                # terminating the worker thread.
                self.device.script_queue.task_done()
                break

            script = script_object.script
            location = script_object.location

            # Block Logic: Ensures a lock exists for the current location, creating one if necessary.
            # This handles dynamic creation of locks for new locations.
            if location not in self.device.location_lock:
                self.device.location_lock[location] = Lock()
            # Functional Utility: Acquires a lock for the specific location to ensure
            # exclusive access to the sensor data during script execution.
            self.device.location_lock[location].acquire()

            
            script_data = [] # Accumulator for all data relevant to the script.
            # Block Logic: Gathers sensor data from neighboring devices at the specified location.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Block Logic: Retrieves the local device's sensor data for the current location.
            data = self.device.get_data(location)

            if data is not None:
                script_data.append(data)

            # Pre-condition: Checks if there is any data collected for the script to run.
            if script_data != []:
                # Functional Utility: Executes the assigned script with the collected data,
                # simulating sensor data processing.
                result = script.run(script_data)
                
                # Functional Utility: Acquires a global lock before updating the local sensor data
                # to prevent race conditions during write operations.
                self.device.is_available.acquire()
                if location in self.device.sensor_data:
                    self.device.sensor_data[location] = result
                self.device.is_available.release()

                # Block Logic: Updates the sensor data of neighboring devices with the script's result.
                for device in self.neighbours:
                    device.set_data(location, result)

            # Functional Utility: Releases the lock for the current location after script execution.
            if self.device.location_lock[location].locked():
                self.device.location_lock[location].release()
            # Functional Utility: Marks the current task as done in the queue,
            # signaling completion to the main `DeviceThread`.
            self.device.script_queue.task_done()



    def run(self):
        """
        The main execution loop for the DeviceThread. It continuously
        fetches neighbor data, processes script assignments, pushes them
        to the task queue for workers, and synchronizes with other devices.
        """
        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: Checks if a shutdown signal has been received from the supervisor.
            if self.neighbours is None:
                # Block Logic: Puts a stop signal `ScriptObject` into the queue for each worker
                # thread, instructing them to terminate gracefully.
                for _ in range(8): # Sends 8 stop signals, one for each worker.
                    self.device.script_queue.put(ScriptObject(None, None, True))
                # Functional Utility: Waits for all worker threads to fully terminate.
                self.stop_all_threads()
                break

            # Functional Utility: Blocks until all scripts for the current timepoint have been
            # assigned to the device.
            self.device.timepoint_done.wait()

            # Block Logic: Creates `ScriptObject`s for each assigned script and adds them
            # to the queue for processing by worker threads.
            for (script, location) in self.device.scripts:
                new_scriptobject = ScriptObject(script, location, False)
                self.device.script_queue.put(new_scriptobject)

            # Block Logic: Waits for all script tasks submitted for the current timepoint
            # to be completed by the worker threads.
            self.device.script_queue.join()
            
            # Functional Utility: Synchronizes with all other DeviceThread instances
            # across devices using a shared barrier, ensuring all devices complete
            # their current timepoint processing before proceeding.
            self.device.shared_barrier.wait()
            # Functional Utility: Clears the `timepoint_done` event, preparing for the
            # next cycle of script assignment.
            self.device.timepoint_done.clear()

    def stop_all_threads(self):
        """
        Waits for all worker threads to finish their execution and then clears
        the list of active threads.
        """
        # Block Logic: Iterates through all worker threads and waits for each to finish.
        for thread in self.threads:
            thread.join()

        self.threads = []
