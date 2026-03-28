"""
Defines the core components for a simulated concurrent sensor network.

This module contains the `Device` and `DeviceThread` classes, which model the behavior
of individual sensor nodes and their internal processing units in a distributed,
time-synchronized environment. The simulation relies on threading primitives to manage
concurrent data processing and inter-device communication.
"""

from threading import Event, Thread, Semaphore
from reusable_barrier import TimePointsBarrier, ClassicBarrier

class Device(object):
    """
    Represents a single device (node) in the sensor network simulation.

    Each device manages its own sensor data, a pool of worker threads, and its
    relationship with neighboring devices. It is responsible for coordinating its
    internal threads and synchronizing with other devices at discrete time points.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor data, keyed by location.
        supervisor: A reference to a supervisor object managing the simulation.
        scripts (list): A list of (script, location) tuples to be processed.
        script_received (Event): An event set when a new script is assigned.
        timepoint_done (Event): An event set to signal the end of a processing timepoint.
        threads (list): A list of `DeviceThread` objects managed by this device.
        neighbours (list): A list of other `Device` objects that are its neighbors.
        locations_semaphore (list): A list of semaphores for mutual exclusion on sensor locations.
        devices_barrier (ClassicBarrier): A barrier to synchronize all devices in the simulation.
        neighbours_barrier (TimePointsBarrier): A barrier to synchronize the device's own threads.
        all_devices (list): A reference to all devices in the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor: The simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()


        self.threads = []
        self.neighbours = []
        self.num_threads = 8

        self.locations_semaphore = None
        self.devices_barrier = None
        self.neighbours_barrier = None
        self.all_devices = None

    def set_neighbours(self, new_neighbours):
        """Assigns a list of neighboring devices."""
        self.neighbours = new_neighbours


    def set_devices_barrier(self, barrier):
        """Sets the global barrier used for synchronizing all devices."""
        self.devices_barrier = barrier


    def set_locations_semaphore(self, locations_semaphore):
        """Sets the list of semaphores for protecting access to sensor locations."""
        self.locations_semaphore = locations_semaphore

    def get_locations(self, location_list):
        """
        Appends this device's sensor data locations to a provided list.

        This method is used during the initial setup phase to aggregate all unique
        sensor locations across all devices.
        """
        for location in self.sensor_data:
            location_list.append(location)


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device and initializes the simulation environment.

        If this is the master device (device_id == 0), it orchestrates the creation
        of shared resources like the global device barrier and location semaphores.
        This method also creates and starts the device's internal worker threads.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        # The device with ID 0 acts as the master for one-time setup.
        if self.device_id == 0:
            # Create a barrier to synchronize all devices.
            barrier = ClassicBarrier(len(devices))
            
            # Aggregate all unique sensor locations from all devices.
            locations = []
            for device in devices:
                device.get_locations(locations)
            locations = sorted(list(set(locations)))

            # Create a semaphore for each unique location to ensure mutual exclusion.
            locations_semaphore = [Semaphore(value=1) for _ in range(len(locations))]

            # Distribute the shared barrier and semaphores to all devices.
            for device in devices:
                device.set_devices_barrier(barrier)
                device.set_locations_semaphore(locations_semaphore)

        self.all_devices = devices
        # A per-device barrier for its own internal threads.
        self.neighbours_barrier = TimePointsBarrier(self.num_threads, self)

        # Create and start the worker threads for this device.
        for i in range(self.num_threads):
            current_thread = DeviceThread(self, i)
            current_thread.start()
            self.threads.append(current_thread)

    def assign_script(self, script, location):
        """
        Assigns a script to be executed or signals the end of a timepoint.

        If a script is provided, it is added to the queue for processing.
        If the script is None, it signals to the worker threads that all scripts
        for the current timepoint have been assigned.

        Args:
            script: The script object to execute.
            location (int): The data location the script applies to.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script is the signal that the timepoint's script assignment is complete.
            self.timepoint_done.set()

    def has_data(self, location):
        """Checks if the device has data for a given location."""
        return location in self.sensor_data

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Returns:
            The data for the location, or None if not present.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates the sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all of its worker threads."""
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread that executes data processing scripts for a Device.

    Each thread processes a subset of the scripts assigned to its parent device.
    It synchronizes with other threads and devices, gathers data from neighbors,
    executes scripts, and updates data based on the results.
    """

    def __init__(self, device, thread_id):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent device.
            thread_id (int): The unique ID for this thread within the device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        The main execution loop for the worker thread.
        
        The thread enters an infinite loop where it waits for synchronization
        signals, processes its assigned scripts, and collaborates with neighbors.
        The loop terminates when the parent device's neighbors are set to None.
        """
        while True:
            # Wait for all threads within this device to reach this point.
            self.device.neighbours_barrier.wait()

            # A shutdown signal from the supervisor.
            if self.device.neighbours is None:
                break

            # Wait for the signal that all scripts for the current timepoint are assigned.
            self.device.timepoint_done.wait()

            if len(self.device.neighbours) != 0:
                devices_with_date = []
                
                # Process a statically-assigned subset of the scripts.
                # This implements a form of parallel work distribution.
                for index in range(
                        self.thread_id,
                        len(self.device.scripts),
                        self.device.num_threads):
                    (script, location) = self.device.scripts[index]
                    
                    script_data = []
                    # Acquire a lock for the specific data location to prevent race conditions.
                    self.device.locations_semaphore[location].acquire()

                    # Gather data for the current location from all neighboring devices.
                    for device in self.device.neighbours:
                        if device.has_data(location):
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                                devices_with_date.append(device)

                    # Gather data from the parent device as well.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                        devices_with_date.append(self.device)

                    # Run the script only if there is data to process.
                    if script_data:
                        # Execute the data fusion/processing script.
                        result = script.run(script_data)
                        # Propagate the result back to all devices that contributed data.
                        for device in devices_with_date:
                            device.set_data(location, result)
                        devices_with_date = []

                    # Release the lock for the data location.
                    self.device.locations_semaphore[location].release()
