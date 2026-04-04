


"""
@93719e14-d9a8-4619-882c-1738ececfb23/device.py
@brief Implements a multi-threaded simulation for distributed sensor devices with specialized barrier and semaphore management.

This module defines the core components for simulating a network of sensor devices,
each capable of executing scripts, managing local sensor data, and interacting
with a central supervisor. This version utilizes custom `TimePointsBarrier` and
`ClassicBarrier` classes for different levels of synchronization, and a list of
`Semaphore` objects for location-specific data access. Script execution is
partitioned among multiple `DeviceThread` instances within each device.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate with neighbors under the
guidance of a supervisor.

Classes:
- Device: Represents a single simulated sensor device.
- DeviceThread: Manages the lifecycle and operation of a Device, including partitioning
                and executing scripts.
- TimePointsBarrier: A custom barrier for synchronizing DeviceThreads within a single device.
- ClassicBarrier: A custom barrier for synchronizing all devices.

Domain: Distributed Systems Simulation, Concurrent Programming, Custom Barrier Synchronization, Sensor Networks.
"""

from threading import Event, Thread, Semaphore
from reusable_barrier import TimePointsBarrier, ClassicBarrier

class Device(object):
    """
    @brief Represents a single simulated sensor device in a distributed network.

    Each device manages its own sensor data, interacts with a supervisor,
    and executes assigned scripts in a multi-threaded environment. This version
    uses multiple `DeviceThread` instances per `Device`, which coordinate script
    execution. Synchronization involves a `TimePointsBarrier` for intra-device
    thread coordination and a `ClassicBarrier` for inter-device synchronization.
    Location-specific data access is protected by `Semaphore` objects.
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
        # Synchronization primitive: Event to signal that all scripts for the current timepoint have been assigned.
        self.timepoint_done = Event()
        self.threads = [] # List of DeviceThread instances associated with this device.
        self.neighbours = [] # List of neighboring devices for the current timepoint.
        self.num_threads = 8 # Configuration: Number of DeviceThread instances per device.

        self.locations_semaphore = None # List of Semaphores, one for each location, for thread-safe access.
        self.devices_barrier = None     # Shared ClassicBarrier for inter-device synchronization.
        self.neighbours_barrier = None  # TimePointsBarrier for intra-device (DeviceThread) synchronization.
        self.all_devices = None         # List of all Device objects in the simulation.

    def set_neighbours(self, new_neighbours):
        """
        @brief Sets the list of neighboring devices for the current timepoint.

        @param new_neighbours: A list of neighboring Device objects.
        """
        self.neighbours = new_neighbours

    def set_devices_barrier(self, barrier):
        """
        @brief Sets the shared ClassicBarrier for inter-device synchronization.

        @param barrier: The ClassicBarrier object.
        """
        self.devices_barrier = barrier

    def set_locations_semaphore(self, locations_semaphore):
        """
        @brief Sets the list of shared Semaphores for location-specific data access.

        @param locations_semaphore: A list of Semaphore objects, one per location.
        """
        self.locations_semaphore = locations_semaphore

    def get_locations(self, location_list):
        """
        @brief Populates a list with all sensor data locations known to this device.

        @param location_list: A list to which the device's locations will be appended.
        """
        for location in self.sensor_data:
            location_list.append(location)

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the device by initializing worker threads and configuring shared resources.

        This method is called once at the beginning of the simulation. If it's the master
        device (device_id == 0), it initializes shared barriers and semaphores. It also
        spawns multiple `DeviceThread` instances for parallel script execution.

        @param devices: A list of all Device objects participating in the simulation.
        """
        # Block Logic: The master device (device_id == 0) initializes shared resources.
        if self.device_id == 0:
            barrier = ClassicBarrier(len(devices)) # Initialize a shared ClassicBarrier for all devices.
            locations = [] # Temporary list to collect all unique locations across all devices.
            locations_semaphore = [] # List to hold semaphores for each unique location.
            
            # Block Logic: Collects all unique locations from all devices.
            for device in devices:
                device.get_locations(locations)

            locations = sorted(list(set(locations))) # Ensures unique and sorted locations.

            # Block Logic: Creates a Semaphore for each unique location.
            for i in range(0, len(locations)):
                locations_semaphore.append(Semaphore(value=1)) # Initialize semaphore with value 1 (binary semaphore).

            # Block Logic: Propagates the shared barrier and location semaphores to all devices.
            for device in devices:
                device.set_devices_barrier(barrier)
                device.set_locations_semaphore(locations_semaphore)

        self.all_devices = devices # Stores the reference to all devices.
        # Functional Utility: Initializes a TimePointsBarrier for intra-device thread synchronization.
        self.neighbours_barrier = TimePointsBarrier(self.num_threads, self)


        # Block Logic: Spawns multiple `DeviceThread` instances for parallel script execution within this device.
        for i in range(self.num_threads):
            current_thread = DeviceThread(self, i)
            current_thread.start()
            self.threads.append(current_thread)

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device to be executed at a specific location
               or signals the completion of script assignments if no script is provided.

        @param script: The script object to be executed, or None if the timepoint is done.
        @param location: The location ID associated with the script, or irrelevant if script is None.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signals that a new script has been assigned.
        else:
            # Block Logic: If no script is provided (script is None), it signifies that
            # all scripts for the current timepoint have been assigned.
            self.timepoint_done.set() # Signals that the timepoint processing is logically done (no more scripts to assign).

    def has_data(self, location):
        """
        @brief Checks if the device has sensor data for a specific location.

        @param location: The location ID to check.
        @return True if data exists for the location, False otherwise.
        """
        if location in self.sensor_data:
            return True
        return False
        
    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The location ID for which to retrieve data.
        @return The sensor data for the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data \
            else None

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
        @brief Initiates the shutdown process for the device by joining its associated threads.

        This ensures that all `DeviceThread` instances associated with this device
        complete their execution before the program exits.
        """
        for i in range(self.num_threads): # Invariant: Joins each DeviceThread.
            self.threads[i].join()


class DeviceThread(Thread):
    """
    @brief Represents a worker thread for a Device, responsible for executing a subset of assigned scripts.

    Each `Device` spawns multiple `DeviceThread` instances. These threads coordinate
    to partition and execute scripts, gather neighbor data, and update sensor data
    while ensuring thread-safe access to shared resources through semaphores.
    """

    def __init__(self, device, thread_id):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: The Device object that this thread belongs to.
        @param thread_id: A unique integer identifier for this specific `DeviceThread` within its `Device`.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id # Unique ID for this thread within its device.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        It continuously attempts to execute its assigned portion of scripts.
        This involves synchronizing with other threads within the same device to get neighbor
        information, waiting for new scripts, processing them, and then synchronizing across
        all devices at the end of a timepoint. The loop terminates when a global exit signal is received.
        """
        while True:
            # Synchronization point: All `DeviceThread` instances within this device wait here.
            # The master thread (thread_id=0) will set `self.device.neighbours` before all threads proceed.
            self.device.neighbours_barrier.wait()

            # Pre-condition: If `self.device.neighbours` is None, it signals the end of the simulation.
            if self.device.neighbours is None:
                break

            # Pre-condition: Waits until the Device signals that all scripts for the current
            # timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: This thread executes a subset of scripts assigned to its parent device.
            # Scripts are partitioned among `num_threads` based on `thread_id`.
            if len(self.device.neighbours) != 0: # Check if there are active neighbors.
                devices_with_date = [] # List to track devices whose data is included in script_data.
                
                # Block Logic: Iterates through its assigned portion of scripts.
                for index in range(
                        self.thread_id, # Start index for this thread's partition.
                        len(self.device.scripts), # End of the scripts list.
                        self.device.num_threads): # Step size, ensuring each thread processes its own subset.
                    (script, location) = self.device.scripts[index]
                    
                    script_data = [] # Collects data for the current script.
                    # Pre-condition: Acquires a semaphore for the specific location to ensure exclusive access.
                    self.device.locations_semaphore[location].acquire()

                    # Block Logic: Gathers sensor data from neighboring devices for the specified location.
                    for device in self.device.neighbours:
                        if device.has_data(location):
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                                devices_with_date.append(device)

                    # Block Logic: Gathers sensor data from the current device for the specified location.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                        devices_with_date.append(self.device)

                    # Pre-condition: If valid script data was collected.
                    if script_data != []:
                        # Functional Utility: Executes the assigned script with the collected sensor data.
                        result = script.run(script_data)
                        # Block Logic: Updates the sensor data for all involved devices with the script's result.
                        for device in devices_with_date:
                            device.set_data(location, result)
                        devices_with_date = [] # Reset for next script.

                    self.device.locations_semaphore[location].release() # Post-condition: Releases the semaphore for the location.


