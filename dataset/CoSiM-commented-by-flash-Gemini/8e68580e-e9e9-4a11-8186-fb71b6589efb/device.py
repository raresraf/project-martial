

"""
@8e68580e-e9e9-4a11-8186-fb71b6589efb/device.py
@brief Implements a multi-threaded simulation for distributed sensor devices using a different synchronization model.

This module defines the core components for simulating a network of sensor devices,
each capable of executing scripts, managing local sensor data, and interacting
with a central supervisor. This version utilizes a different set of threading
primitives, specifically `Event`, `Thread`, `Lock`, and a custom `ReusableBarrier`
for synchronization.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate with neighbors under the
guidance of a supervisor.

Classes:
- Device: Represents a single simulated sensor device.
- DeviceThread: Manages the lifecycle and operation of a Device instance in its own thread,
                responsible for script execution and inter-device communication.

Domain: Distributed Systems Simulation, Concurrent Programming, Sensor Networks.
"""

from threading import Event, Thread, Lock
import reusable_barrier_semaphore


class Device(object):
    """
    @brief Represents a single simulated sensor device in a distributed network.

    Each device manages its own sensor data, interacts with a supervisor,
    and executes assigned scripts in a multi-threaded environment. This implementation
    uses multiple `DeviceThread` instances per `Device` to process scripts concurrently
    and relies on a `ReusableBarrier` for timepoint synchronization.
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
        self.scripts = [] # Stores scripts assigned to this device for execution. Each script is (script_object, location_id).

        # Configuration: Defines the number of `DeviceThread` instances launched per device.
        self.number_threads_per_device = 8
        # Configuration: Defines the total number of distinct locations for sensor data.
        self.number_locations = 100

        # Synchronization primitive: A barrier to synchronize all `DeviceThread` instances
        # across all devices at the end of each timepoint. This is set by the master device.
        self.barrier_timepoint = None
        
        # Synchronization primitive: A reusable barrier to synchronize the `DeviceThread` instances
        # within this specific device before proceeding to get neighbors.
        self.barrier_get_neighbours = \
            reusable_barrier_semaphore.\
            ReusableBarrier(self.number_threads_per_device)

        # Synchronization primitive: A reusable barrier to synchronize the `DeviceThread` instances
        # within this specific device before resetting counters for the next timepoint.
        self.barrier_reset_counters = \
            reusable_barrier_semaphore.\
            ReusableBarrier(self.number_threads_per_device)

        # Synchronization primitive: A lock to protect access to the `scripts` list and
        # `script_access_index` when multiple `DeviceThread` instances try to fetch scripts.
        self.script_access = Lock()

        # State variable: Keeps track of the index of the next script to be accessed by a `DeviceThread`.
        self.script_access_index = -1

        # State variable: Flag to indicate if all assigned scripts for the current timepoint have been processed.
        self.finished_scripts = 0

        # State variable: Flag to signal that the simulation should exit.
        self.exit_simulation = 0

        self.neighbours = [] # Stores a list of neighboring devices for the current timepoint.
        self.all_devices = [] # A list of all Device objects in the simulation, set during setup.

        # Synchronization primitive: A list of locks, one for each location, to protect concurrent
        # access to sensor data at specific locations when multiple threads update it.
        self.locks_location_update_data = []
        for _ in xrange(self.number_locations):
            self.locks_location_update_data.append(Lock())

        # Synchronization primitive: An event used to signal `DeviceThread` instances when
        # new scripts have been assigned or when the `scripts` list is updated.
        self.event_access_data = Event()

        self.thread_id = 0
        self.thread_list = []

        # Block Logic: Initializes and starts multiple `DeviceThread` instances for this device.
        # These threads will concurrently process scripts assigned to this device.
        for self.thread_id in xrange(self.number_threads_per_device):
            self.thread_list.append(DeviceThread(self, self.thread_id))

        for thread in self.thread_list:
            thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the device's awareness of all other devices in the simulation
               and initializes shared synchronization primitives if it's the master device.

        This method is called once at the beginning of the simulation.
        If this is the master device (device_id == 0), it initializes a global
        `barrier_timepoint` and propagates it, along with location locks, to all other devices.

        @param devices: A list of all Device objects participating in the simulation.
        """
        self.all_devices = devices
        
        # Block Logic: Master device (device_id == 0) initializes and propagates shared synchronization objects.
        # This ensures that all devices share the same global timepoint barrier and location-specific locks.
        if (self.device_id is 0):
            # Functional Utility: Initializes a global barrier for synchronizing all `DeviceThread` instances
            # across all devices at the end of each timepoint.
            self.barrier_timepoint = \
                reusable_barrier_semaphore. \
                ReusableBarrier(len(self.all_devices) * \
                    self.number_threads_per_device)
            # Block Logic: Propagates the initialized global barrier and location locks to all devices.
            i = 0
            for i in xrange(len(self.all_devices)):
                devices[i].barrier_timepoint = self.barrier_timepoint

                devices[i].locks_location_update_data = \
                self.locks_location_update_data
                

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device to be executed at a specific location.

        This method appends the script and its location to the device's script list
        and signals the `DeviceThread` instances that new scripts are available.

        @param script: The script object to be executed.
        @param location: The location ID associated with the script.
        """
        self.scripts.append((script, location))

        # Functional Utility: Signals to any waiting `DeviceThread` instances that there is new data
        # (scripts) available to be processed.
        self.event_access_data.set()


    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The location ID for which to retrieve data.
        @return The sensor data for the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if  \
            location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.

        This method directly updates the `sensor_data` dictionary for the given location.

        @param location: The location ID for which to set data.
        @param data: The new sensor data value.
        """
        if location in self.sensor_data:
            # Post-condition: Updates the sensor data at the specified location.
            self.sensor_data[location] = data
            

    def shutdown(self):
        """
        @brief Initiates the shutdown process for the device by joining its associated threads.

        This ensures that all `DeviceThread` instances belonging to this device
        complete their execution before the program exits.
        """
        thread = None
        for thread in self.thread_list:
            thread.join()


class DeviceThread(Thread):
    """
    @brief Represents a worker thread for a Device, responsible for executing assigned scripts.

    Each `Device` spawns multiple `DeviceThread` instances to process scripts concurrently.
    These threads coordinate to fetch scripts, retrieve neighbor data, execute scripts,
    and update sensor data while ensuring thread-safe access to shared resources.
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

        # State variable: Keeps track of the index of the script currently being accessed by this thread.
        self.script_to_access = 0
        


    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        It continuously attempts to fetch and execute scripts assigned to its parent `Device`.
        This involves synchronizing with other threads within the same device to get neighbor
        information, fetching scripts, processing them, and then synchronizing across all
        devices at the end of a timepoint. The loop terminates when a global exit signal is received.
        """
        while True:

            # Block Logic: Only the first thread (thread_id == 0) in each device retrieves neighbor information
            # from the supervisor to avoid redundant calls and coordinate.
            if (self.thread_id == 0):
                self.device.neighbours = \
                self.device.supervisor.get_neighbours()
                # Pre-condition: If no neighbors are returned (indicating end of simulation), set the exit flag.
                if self.device.neighbours is None:
                    self.device.exit_simulation = 1
 
            # Synchronization point: All `DeviceThread` instances within this device wait here
            # until the master thread has retrieved neighbor information or set the exit flag.
            self.device.barrier_get_neighbours.wait()

            # Pre-condition: If the simulation exit flag is set, break out of the main loop.
            if (self.device.exit_simulation == 1):
                break
            
            # Block Logic: Loop to process scripts assigned to the device.
            # Invariant: Continues until all scripts for the current timepoint are marked as finished.
            while (self.device.finished_scripts == 0):

                # Block Logic: Acquires a lock to safely get the next script index to process.
                self.device.script_access.acquire()

                self.device.script_access_index += 1
                self.script_to_access = self.device.script_access_index

                # Pre-condition: If the current script index exceeds the number of available scripts,
                # it means this thread has processed all current scripts or is waiting for more.
                # It then waits on `event_access_data` if no more scripts are available.
                if (self.device.script_access_index >= \
                    len(self.device.scripts)):
                    # Functional Utility: Blocks until `event_access_data` is set (e.g., by `assign_script` or cleanup).
                    self.device.event_access_data.wait()                        
                
                # Invariant: If `finished_scripts` is still 0 (meaning scripts are still being processed),
                # clear the event for the next cycle.
                if (self.device.finished_scripts == 0):
                    self.device.event_access_data.clear()       
                
                self.device.script_access.release()

                # Pre-condition: If the `finished_scripts` flag is set by another thread, break from this inner loop.
                if (self.device.finished_scripts == 1):
                    break
      
                # Functional Utility: Retrieves the script and its associated location.
                (script, location) = self.device.scripts[self.script_to_access]
       
                # Pre-condition: If a `None` script is encountered (a sentinel value), it signifies
                # that all scripts for the current timepoint have been processed.
                if (script is None):
              
                    self.device.finished_scripts = 1 # Post-condition: Marks all scripts as finished for this timepoint.

                    # Functional Utility: Signals to other threads waiting on `event_access_data` that
                    # scripts are now finished, allowing them to break their loops.
                    self.device.event_access_data.set()
                    break
                    
                
                # Block Logic: Acquires a location-specific lock to ensure exclusive access
                # to the sensor data at this particular location during script execution and data update.
                self.device.locks_location_update_data[location].acquire()
                script_data = []

                # Block Logic: Gathers sensor data from neighboring devices for the specified location.
                if (self.device.neighbours != []):
                    for device in self.device.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                # Pre-condition: If valid script data was collected (either from neighbors or self).
                if (script_data != []):

                    # Block Logic: Gathers sensor data from the current device for the specified location.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    # Functional Utility: Executes the assigned script with the collected sensor data.
                    # The script's `run` method presumably contains the core logic for data processing.
                    result = script.run(script_data)
             
                    # Block Logic: Updates the sensor data for all neighboring devices with the script's result.
                    for device in self.device.neighbours:

                        device.set_data(location, result)
            
                    # Block Logic: Updates the sensor data for the current device with the script's result.
                    self.device.set_data(location, result)

                # Post-condition: Releases the location-specific lock.
                self.device.locks_location_update_data[location].release()

            # Post-condition: Resets the script access index for the next timepoint.
            self.script_to_access = 0

            # Synchronization point: All `DeviceThread` instances within this device wait here
            # until all scripts have been processed and counters are ready to be reset.
            self.device.barrier_reset_counters.wait()
            # Block Logic: Only the first thread (thread_id == 0) in each device performs cleanup
            # and resets state for the next timepoint.
            if (self.thread_id == 0):
                # Functional Utility: Removes the processed scripts and resets internal state.
                # Invariant: The last script entry might be a sentinel `None` and is removed.
                self.device.scripts.pop() # Removes the `None` sentinel script from the list.
                self.device.event_access_data.clear() # Resets the event for the next timepoint.
                self.device.script_access_index = -1 # Resets the script index.
                self.device.finished_scripts = 0 # Resets the finished scripts flag.

            # Synchronization point: All `DeviceThread` instances across ALL devices wait here
            # until every other device's threads have completed their current timepoint processing.
            # This ensures global synchronization before moving to the next timepoint.
            self.device.barrier_timepoint.wait()
