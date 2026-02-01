"""
/**
 * @file device.py
 * @brief This module simulates a network of distributed devices that execute scripts and synchronize their states.
 *
 * @details
 * The system is composed of `Device` objects, each running in its own `DeviceThread`.
 * These devices are managed by a central `Supervisor` (not defined in this file).
 * The simulation proceeds in synchronized timepoints. At each timepoint, each device
 * spawns multiple `ScriptThread` instances to execute assigned scripts in parallel.
 *
 * Core functionalities include:
 * - **Script Execution**: Each script is executed in its own thread (`ScriptThread`),
 *   which gathers data from neighboring devices before running the computation.
 * - **Data Synchronization**: The result of a script is written back to the device itself
 *   and all its neighbors.
 * - **Concurrency Control**:
 *   - `Lock`: Used to ensure thread-safe access to a device's sensor data (`lock_setter`, `lock_getter`)
 *     and to serialize script assignments (`lock_assign`).
 *   - `location_lock`: A shared dictionary of locks, one for each data location, to ensure that
 *     only one script across the entire system can operate on a specific location at a time.
 *   - `Semaphore`: Limits the number of concurrently running `ScriptThread` instances to 8,
 *     acting as a thread pool size limiter.
 * - **Global Synchronization**: A `ReusableBarrier` is employed to synchronize all devices at the end
 *   of each timepoint, ensuring a lock-step execution model for the entire simulation.
 *
 * The architecture demonstrates a distributed computation pattern where each device can process
 * its tasks in parallel, while global consistency is maintained through a combination of fine-grained
 * locks and a global barrier.
 */
"""

from threading import Event, Thread, Lock, Semaphore
from ReusableBarrier import ReusableBarrier


class Device(object):
    """
    Represents a single device (node) in the distributed system simulation.
    Each device has a unique ID, its own sensor data, and is managed by a supervisor.
    It uses threading primitives to handle concurrent script execution and data access.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding the device's local sensor data, keyed by location.
            supervisor (Supervisor): The central object that coordinates the devices in the system.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when all scripts for a timepoint have been assigned.
        self.script_received = Event()
        self.scripts = []

        # Individual locks for fine-grained control over data access and script assignment.
        self.lock_setter = Lock()
        self.lock_getter = Lock()
        self.lock_assign = Lock()

        # Shared synchronization objects, initialized by the setup_devices method.
        self.barrier = None
        self.location_lock = {}

        # Limits the number of concurrent script executions to 8.
        self.semaphore = Semaphore(8)


        self.thread = DeviceThread(self)

    def __str__(self):
        """
        Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects to all devices.

        This method is designed to be called once by a single device (device_id 0) to set up
        the global barrier and location-specific locks for the entire system.

        Args:
            devices (list): A list of all `Device` objects participating in the simulation.
        """
        # This setup is performed by a single designated device to create shared synchronization primitives.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))

            # Create a lock for each unique data location across all devices.
            for device in devices[:]:
                for loc in device.sensor_data.keys():
                    if loc not in self.location_lock:
                        self.location_lock[loc] = Lock()

            # Distribute the shared barrier and locks to all devices.
            for device in devices[:]:
                device.barrier = self.barrier
                device.location_lock = self.location_lock
                
                device.thread.start()


    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution.

        A `None` script is used as a sentinel to indicate that all scripts for the current
        timepoint have been assigned, which unblocks the main device thread.

        Args:
            script (Script): The script object to execute.
            location (any): The location context for the script execution.
        """
        with self.lock_assign:
            if script is not None:
                self.scripts.append((script, location))
            else:
                self.script_received.set()

    def get_data(self, location):
        """
        Retrieves data for a specific location, with thread-safe access.

        Args:
            location (any): The key for the data to retrieve.

        Returns:
            The data for the given location, or None if the location does not exist.
        """
        with self.lock_getter:
            return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates the data for a specific location, with thread-safe access.

        Args:
            location (any): The key for the data to update.
            data (any): The new value for the data.
        """
        with self.lock_setter:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by waiting for its main thread to complete.
        """
        self.thread.join()


class ScriptThread(Thread):
    """
    A worker thread responsible for executing a single script.
    It handles data gathering, execution, and result propagation for one script instance.
    """

    def __init__(self, device_thread, script, location, neighbours):
        """
        Initializes a ScriptThread.

        Args:
            device_thread (DeviceThread): The parent device thread that spawned this worker.
            script (Script): The script to be executed.
            location (any): The location context for the script.
            neighbours (list): A list of neighboring devices to interact with.
        """
        Thread.__init__(self)
        self.script = script
        self.device_thread = device_thread
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        The main execution logic for the script thread.
        It acquires locks, gathers data, runs the script, and updates data on all relevant devices.
        """
        # Functional Utility: The semaphore limits concurrency, while the location lock ensures data integrity for a specific location.
        self.device_thread.device.location_lock[self.location].acquire()
        self.device_thread.device.semaphore.acquire()

        script_data = []
        
        # Block Logic: Aggregate data from all neighbors for the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Include the parent device's own data.
        data = self.device_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            # Propagate the result to all neighbors and the parent device.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device_thread.device.set_data(self.location, result)

        # Release the semaphore and the location lock.
        self.device_thread.device.semaphore.release()
        self.device_thread.device.location_lock[self.location].release()




class DeviceThread(Thread):
    """
    The main control thread for a `Device`, managing its participation in the simulation.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device.
        This loop represents the device's lifecycle through synchronized timepoints.
        """
        # Block Logic: The main loop of the simulation for this device.
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: A None value for neighbors indicates a shutdown signal from the supervisor.
            if neighbours is None:
                break



            # Waits for the supervisor to signal that all scripts for the current timepoint have been assigned.
            self.device.script_received.wait()
            script_threads = []

            
            # Block Logic: Spawns a new ScriptThread for each assigned script.
            for (script, location) in self.device.scripts:
                
                thread = ScriptThread(self, script, location, neighbours)
                script_threads.append(thread)
                thread.start()

            # Wait for all spawned script threads to complete their execution.
            for thread in script_threads:
                thread.join()

            self.device.script_received.clear()

            
            # Functional Utility: Waits at the barrier, synchronizing with all other devices before proceeding to the next timepoint.
            self.device.barrier.wait()
