
"""
This module defines core components for a distributed device simulation framework.
It includes a worker thread (`MyThread`) for executing scripts, a `Device` class
to manage sensor data and coordinate worker threads, and a `DeviceThread`
for interacting with a supervisor and dispatching tasks. The framework relies
on threading primitives for concurrent execution and synchronization.
"""

from threading import Thread, Lock, Event

class MyThread(Thread):
    """
    A worker thread responsible for executing assigned scripts on sensor data.

    Each `MyThread` processes a list of scripts, collects relevant data from its
    associated device and its neighbors, executes the script, and updates data.
    Synchronization is managed using events and internal locks.
    """

    def __init__(self, device):
        """
        Initializes a MyThread instance.

        Args:
            device (Device): The `Device` instance this thread is associated with.
        """
        Thread.__init__(self)
        self.device = device
        self.scripts_list = [] # List of (script, location) tuples assigned to this thread.
        self.neighbours_list = [] # List of neighbor device lists for each script.
        # Event to signal this thread to start processing its assigned tasks.
        self.permission = Event()
        # Event to signal that this thread has finished processing its tasks for the current cycle.
        self.finish = Event()
        self.thread_killed = 0 # Flag to indicate if the thread should terminate.
        # Lock to protect `scripts_list` and `neighbours_list` during access and modification.
        self.lists = Lock()

    def run(self):
        """
        Main execution loop for the MyThread worker.

        Architectural Intent: Continuously waits for a signal to process tasks.
        Upon receiving permission, it iterates through its assigned scripts,
        collects data (from device and neighbors), executes the script, and
        updates the relevant sensor data, ensuring data consistency via locks.
        """
        while True:
            # Block Logic: Waits for the `permission` event to be set by the `DeviceThread`,
            # indicating that there are tasks to process or a shutdown signal.
            self.permission.wait()

            # Termination Condition: If `thread_killed` flag is set, the thread breaks its loop and terminates.
            if self.thread_killed == 1:
                break

            # Functional Utility: Clears events to prepare for the next cycle of processing.
            self.permission.clear()
            self.finish.clear()

            # Block Logic: Processes all assigned scripts and corresponding neighbor lists.
            # Invariant: Continues as long as there are scripts and neighbor lists to process.
            while len(self.scripts_list) > 0 and len(self.neighbours_list) > 0:
                
                # Functional Utility: Acquires a lock to safely access and remove elements
                # from the shared `scripts_list` and `neighbours_list`.
                self.lists.acquire()
                script, place = self.scripts_list[0]
                neighbours = self.neighbours_list[0]

                del self.scripts_list[0]
                del self.neighbours_list[0]
                self.lists.release()

                # Functional Utility: Acquires a global lock specific to the data location (`place`).
                # This ensures exclusive access to the sensor data during script execution and update.
                self.device.scripts_locks[place].acquire()

                # Block Logic: Collects data for the script execution.
                data_list = []
                # Collects data from its own device.
                data = self.device.get_data(place)
                if data is not None:
                    data_list.append(data)

                # Collects data from neighboring devices.
                for neighbour in neighbours:
                    data = neighbour.get_data(place)
                    if data is not None:
                        data_list.append(data)

                # Block Logic: If any data was collected, executes the script and updates data.
                if len(data_list) > 0:
                    
                    # Functional Utility: Executes the assigned script with the collected data.
                    result = script.run(data_list)

                    # Updates data on its own device.
                    self.device.set_data(place, result)

                    # Updates data on neighboring devices.
                    for neighbour in neighbours:
                        neighbour.set_data(place, result)

                # Functional Utility: Releases the global lock for the data location,
                # making it available for other threads/devices.
                self.device.scripts_locks[place].release()

            # Functional Utility: Sets the `finish` event to signal the `DeviceThread`
            # that this worker has completed its tasks for the current cycle.
            self.finish.set()


from threading import Event, Thread, Lock
from MyBarrier import MyBarrier # External dependency: Assumes MyBarrier class is defined in MyBarrier.py
# from MyThread import MyThread # This import is redundant if MyThread class is in the same file and already defined above

class Device(object):
    """
    Represents a simulated device within a distributed system.

    Each device manages its own sensor data, interacts with a central supervisor,
    and dispatches scripts to a pool of worker threads (`MyThread` instances)
    for concurrent execution. It participates in global synchronization through `MyBarrier`.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor readings
                                (e.g., {location_id: data_value}).
            supervisor (object): An object representing the central supervisor,
                                 used for coordination (e.g., getting neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that all scripts for the current timepoint have been assigned.
        self.script_received = Event()
        # List to temporarily store scripts assigned to this device before dispatching to worker threads.
        self.scripts = []

        # Event to signal that the device is ready to start its main processing loop.
        self.ready = Event()

        # The main thread for the device, responsible for supervisor interaction and dispatching scripts.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Pool of worker threads (`MyThread` instances) for concurrent script execution.
        self.threads = []
        self.nr_threads = 8 # Invariant: A fixed number of 8 worker threads per device.

        # Block Logic: Creates and starts the specified number of worker threads.
        for _ in range(0, self.nr_threads):
            thread = MyThread(self) # Each worker thread is associated with this Device instance.
            self.threads.append(thread)
            thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global synchronization primitives (barrier and data locks)
        and distributes them among all devices.

        This method is typically called once during the initialization phase of the simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0: # Block Logic: Only device with ID 0 acts as the coordinator for setup.
            # Functional Utility: Creates a global barrier that all devices will use for synchronization.
            nr_devices = len(devices)
            barrier = MyBarrier(nr_devices)

            # Functional Utility: Initializes a global pool of locks for sensor data locations.
            # This design implies that `places` and `locks` are globally managed and shared
            # across all devices to ensure consistent access control to sensor data.
            places = [] # Not explicitly used after creation, but shows intent.
            locks = []
            # Invariant: Iterates through all devices to determine the total number of data locations
            # and create a corresponding lock for each.
            for device in devices:
                places.extend(device.sensor_data.keys())
                data = len(device.sensor_data.keys())
                for _ in range(data):
                    locks.append(Lock()) # Creates a Lock for each sensor data point.
                
                # Assigns the globally created barrier and locks to each device.
                device.barrier = barrier
                device.scripts_locks = locks
                # Signals that the device is ready to proceed.
                device.ready.set()

    def assign_script(self, script, location):
        """
        Assigns a script for execution at a specific data location.

        These scripts are queued for processing by the device's worker threads.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the sensor data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If script is None, it signifies the end of script assignments for the current timepoint.
            # Functional Utility: Signals that all scripts for the current timepoint have been assigned.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Note: Locks are managed by the `MyThread` worker prior to calling this method.

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location.

        Note: Locks are managed by the `MyThread` worker prior to calling this method.

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new sensor data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Performs a graceful shutdown of all worker threads and the main device thread.
        """
        # Waits for all worker threads (`MyThread` instances) to complete their execution.
        for thread in self.threads:
            thread.join()
        # Waits for the main device thread (`DeviceThread`) to complete its execution.
        self.thread.join()

class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for managing its interaction
    with the supervisor and coordinating the execution of scripts by its
    pool of worker threads (`MyThread` instances).
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The `Device` instance this thread is associated with.
        """
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        Main execution loop for the DeviceThread.

        Architectural Intent: Continuously fetches neighbor information from the supervisor.
        It orchestrates the assignment of scripts to `MyThread` workers using a round-robin
        approach and manages synchronization points using events and a global barrier.
        It also handles the graceful termination of worker threads.
        """
        # Block Logic: Waits until the device is marked as ready, ensuring all setup is complete.
        self.device.ready.wait()

        while True:
            # Block Logic: Fetches the current list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Termination Condition: If no neighbors are returned (None), it signifies
            # that the simulation is ending for this device. It then initiates the
            # shutdown of its worker threads.
            if neighbours is None:
                # Signals all worker threads to terminate.
                for thread in self.device.threads:
                    thread.thread_killed = 1
                    thread.permission.set() # Sets permission to allow threads to check the kill flag.
                break # Exits the main loop.

            # Block Logic: Waits for the `script_received` event to be set,
            # indicating that all scripts for the current timepoint have been assigned to the device.
            self.device.script_received.wait()

            
			# Block Logic: Dispatches scripts to the worker threads in a round-robin fashion.
            # This balances the workload across the `MyThread` pool.
            scr = len(self.device.scripts)
            for i in range(0, scr):
                crt = i % self.device.nr_threads # Determines which worker thread to assign the script to.
                # Appends the script and its location to the worker thread's internal list.
                self.device.threads[crt].scripts_list.append(self.device.scripts[i])
                # Appends the list of neighbors to the worker thread's internal list.
                self.device.threads[crt].neighbours_list.append(neighbours)

            # Functional Utility: Clears the device's script list as they have been dispatched.
            self.device.scripts = []

            # Block Logic: Signals all worker threads to start processing their assigned tasks
            # by setting their `permission` event.
            for thread in self.device.threads:
                thread.permission.set()

            # Functional Utility: Clears the `script_received` event to prepare for the next timepoint.
            self.device.script_received.clear()

            # Block Logic: Waits for all worker threads to finish their current tasks
            # by waiting on their individual `finish` events.
            for thread in self.device.threads:
                thread.finish.wait()

            # Block Logic: Waits at the global barrier for all devices to complete their
            # current timepoint processing before proceeding to the next simulation step.
            self.device.barrier.wait()
