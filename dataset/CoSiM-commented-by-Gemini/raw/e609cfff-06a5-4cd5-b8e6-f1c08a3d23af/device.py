"""
This module defines the classes for a simulated distributed device network.

It provides a framework for simulating multiple devices that process sensor data
concurrently. The architecture is built around three main classes:
- Device: Represents a single node in the network with its own data and worker threads.
- DeviceThread: The main control thread for a Device, managing its lifecycle.
- DeviceOwnThread: A worker thread within a Device that executes computational scripts.

The system uses threading primitives like Locks, Events, and Barriers to synchronize
the devices and their operations in discrete, coordinated cycles.
"""
from threading import Event, Thread, Lock
from barrier import Barrier
from device_thread import DeviceOwnThread

class Device(object):
    """Represents a single device in the simulated network.

    Each device manages its own sensor data, a set of scripts to execute, and a
    pool of worker threads. It communicates with other devices and a central
    supervisor to perform synchronized, location-based computations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor values.
            supervisor (Supervisor): An external object that manages the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

        self.new_sensor_data = {}
        self.new_sensor_data_lock = Lock()

        self.other_devices = []

        # Event to signal that the device is ready to receive scripts.
        self.ready_to_get_scripts = Event()

        # Event to signal that all scripts for the current cycle have been received.
        self.got_all_scripts = Event()

        # Lock to ensure thread-safe assignment of scripts to worker threads.
        self.assign_script_lock = Lock()

        # Lock to protect access to the device's sensor_data.
        self.set_data_lock = Lock()

        # Barrier to synchronize the start of each processing loop across all devices.
        self.start_loop_barrier = Barrier()

        # Barrier to synchronize all devices after they have received their scripts.
        self.got_scripts_barrier = Barrier()

        # Barrier to ensure all devices have completed their computation for the cycle.
        self.everyone_done = Barrier()

        # A list of mutexes, one for each location, to protect location-based data.
        self.location_mutex = []

        # Lock to synchronize access to neighbour discovery.
        self.get_neighbours_lock = Lock()

        # Event to signal when sensor data is stable and ready to be read.
        self.data_ready = Event()
        self.data_ready.set()

        # A pool of worker threads to execute scripts concurrently.
        self.own_threads = []
        self.power = 20  # The number of worker threads in the pool.
        for _ in range(0, self.power): 
            new_thread = DeviceOwnThread(self)
            self.own_threads.append(new_thread)
            new_thread.start()

        # Index for round-robin assignment of scripts to worker threads.
        self.own_threads_rr = 0

        # Event to signal that the device has been fully initialized.
        self.initialized = Event()

        # The main control thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def get_main_device(self):
        """Finds and returns the main device, which acts as a coordinator.

        The main device is determined by the lowest device_id in the network.
        It is responsible for managing shared synchronization primitives.

        Returns:
            Device: The main device instance.
        """
        min_device = self
        min_id = self.device_id
        for device in self.other_devices:
            if device.device_id < min_id:
                min_device = device
                min_id = device.device_id
        return min_device

    def get_start_loop_barrier(self):
        """Gets the shared start loop barrier from the main device."""
        return self.get_main_device().start_loop_barrier

    def get_got_scripts_barrier(self):
        """Gets the shared 'got scripts' barrier from the main device."""
        return self.get_main_device().got_scripts_barrier

    def get_get_neighbours_lock(self):
        """Gets the shared 'get neighbours' lock from the main device."""
        return self.get_main_device().get_neighbours_lock

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the network of devices and initializes shared resources.

        This method is called to provide each device with a list of all other
        devices in the network. The main device initializes shared barriers
        and location-based mutexes.
        """
        for device in devices:
            self.other_devices.append(device)

        # The main device is responsible for initializing shared synchronization objects.
        if self.get_main_device() == self:
            # Set the number of parties for each barrier to the total number of devices.
            self.start_loop_barrier.set_n(len(devices))
            self.got_scripts_barrier.set_n(len(devices))
            self.everyone_done.set_n(len(devices))

            # Determine the total number of unique locations from all devices' sensor data.
            number_of_locations = 0
            for device in devices:
                current_number = max(device.sensor_data) if device.sensor_data else -1
                if current_number > number_of_locations:
                    number_of_locations = current_number

            # Create a mutex for each location.
            for _ in range(0, number_of_locations + 1):
                self.location_mutex.append(Lock())

        # Signal that the device is ready to start receiving scripts.
        self.ready_to_get_scripts.set()

        # Signal that the device initialization process is complete.
        self.initialized.set()

    def assign_script(self, script, location):
        """Assigns a script to a worker thread in a round-robin fashion.

        If the script is None, it's treated as a signal that no more scripts
        will be assigned for the current cycle.

        Args:
            script (Script): The computational script to be executed.
            location (int): The location on which the script will operate.
        """
        self.ready_to_get_scripts.wait()
        with self.assign_script_lock:
            if script is not None:
                # Assign script to the next worker thread.
                self.own_threads[self.own_threads_rr].assign_script(script, location)
                self.own_threads_rr = (self.own_threads_rr + 1) % len(self.own_threads)
            else:
                # A None script marks the end of script assignment for this cycle.
                self.ready_to_get_scripts.clear()
                self.data_ready.clear()
                self.got_all_scripts.set()

    def get_data(self, location):
        """Gets sensor data for a specific location in a thread-safe manner."""
        self.data_ready.wait()
        with self.set_data_lock:
            result = self.sensor_data.get(location)
        return result

    def get_temp_data(self, location):
        """Gets sensor data without waiting or locking. Used for internal script execution."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data for a specific location in a thread-safe manner."""
        with self.set_data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its threads."""
        for dot in self.own_threads:
            dot.join()
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a Device.

    This thread orchestrates the device's participation in the synchronized
    computation cycles of the network.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop for the device."""
        self.device.initialized.wait()
        self.device.ready_to_get_scripts.set()
        self.device.data_ready.clear()
        
        while True:
            # === Start of Cycle: Synchronize all devices ===
            self.device.get_start_loop_barrier().wait()

            # Get the list of neighbors for this cycle from the supervisor.
            with self.device.get_get_neighbours_lock():
                neighbours = self.device.supervisor.get_neighbours()

            # Pass neighbour list to worker threads and notify them to start.
            for dot in self.device.own_threads:
                dot.waiting_for_permission.wait()
            for dot in self.device.own_threads:
                dot.neighbours = neighbours
                dot.waiting_for_permission.clear()
                with dot.start_loop_condition:
                    dot.start_loop_condition.notify_all()

            # A None for neighbours signals the end of the simulation.
            if neighbours is None:
                break

            # === Script Assignment Phase ===
            # Wait until the supervisor signals that all scripts for this cycle are assigned.
            self.device.got_all_scripts.wait()
            self.device.got_all_scripts.clear()
            self.device.data_ready.clear()

            # Synchronize to ensure all devices have received their scripts before execution.
            self.device.get_got_scripts_barrier().wait()

            # === Script Execution Phase ===
            # Signal all worker threads to start executing their assigned scripts.
            for dot in self.device.own_threads:
                dot.execute_scripts_event.set()

            # Wait for all local worker threads to complete their execution.
            for dot in self.device.own_threads:
                dot.done.wait()

            # === End of Cycle: Synchronize all devices ===
            # Wait for all devices in the network to confirm completion.
            self.device.get_main_device().everyone_done.wait()

            self.device.data_ready.set()
            
            # Reset events for the next cycle.
            for dot in self.device.own_threads:
                dot.done.clear()
                dot.execute_scripts_event.clear()

            # Signal that the device is ready to accept scripts for the next cycle.
            self.device.ready_to_get_scripts.set()



from threading import Event, Thread, Condition

class DeviceOwnThread(Thread):
    """A worker thread owned by a Device.

    This thread is responsible for executing one or more computational scripts
    on data gathered from its parent device and its neighbors.
    """

    def __init__(self, device):
        """Initializes the worker thread."""
        Thread.__init__(self)
        self.device = device

        # Event to signal that this thread has completed its work for the cycle.
        self.done = Event()

        # List of (script, location) tuples to be executed.
        self.scripts = []

        # Event to trigger the execution of scripts.
        self.execute_scripts_event = Event()

        # Condition variable to wait for the start of a new cycle.
        self.start_loop_condition = Condition()

        # Event to signal that the thread is ready to receive neighbour data.
        self.waiting_for_permission = Event()

        self.neighbours = []

    def assign_script(self, script, location):
        """Receives a script and its target location for execution."""
        self.scripts.append((script, location))

    def execute_scripts(self):
        """Executes all assigned scripts for the current cycle."""
        for (script, location) in self.scripts:
            # Acquire a lock for the specific location to ensure data consistency.
            with self.device.get_main_device().location_mutex[location]:
                script_data = []
                
                # Gather data for the location from all neighboring devices.
                for device in self.neighbours:
                    data = device.get_temp_data(location)
                    if data is not None:
                        script_data.append(data)

                # Gather data from the parent device.
                data = self.device.get_temp_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Run the script on the collected data.
                    result = script.run(script_data)

                    # Broadcast the result back to all neighbors and the parent device.
                    for device in self.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

    def run(self):
        """The main loop of the worker thread."""
        while True:
            # === Wait for Start of Cycle ===
            with self.start_loop_condition:
                self.waiting_for_permission.set()
                self.start_loop_condition.wait()

            # Check for simulation end signal.
            if self.neighbours is None:
                break

            # === Wait for Execution Trigger ===
            self.execute_scripts_event.wait()

            self.execute_scripts()

            # Signal that execution for this cycle is complete.
            self.done.set()
