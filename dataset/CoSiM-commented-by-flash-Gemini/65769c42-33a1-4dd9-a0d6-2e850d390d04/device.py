
"""

@65769c42-33a1-4dd9-a0d6-2e850d390d04/device.py

@brief Implements a simulated device for a distributed sensor network, with concurrent script execution and detailed locking mechanisms.

This module defines a `Device` that processes sensor data and executes scripts.

It features a `DeviceThread` that dispatches scripts to `ScriptThread` instances for

concurrent processing. Synchronization is handled by a `ReusableBarrier` for global

time-step coordination, and a combination of shared and location-specific `Lock` objects

to ensure thread-safe data access and modification across devices.

"""



from threading import Event, Thread, Lock

from reusable_barrier import ReusableBarrier # Assumed to contain a ReusableBarrier class.





class Device(object):

    """

    @brief Represents a single device in the distributed system simulation.

    Manages its local sensor data, assigned scripts, and coordinates its operation

    through a dedicated thread, a shared barrier, and multiple layers of locking

    for data consistency.

    """

    

    # Class-level attributes to be shared across all instances of Device.

    timepoint_barrier = None # Shared ReusableBarrier for global time step synchronization.

    script_lock = None # Shared Lock for critical sections during script processing.

    data_lock = None # General shared Lock for data access, potentially superseded by `data_locks`.

    data_locks = {} # Dictionary to hold location-specific Locks for fine-grained data protection.



    def __init__(self, device_id, sensor_data, supervisor):

        """

        @brief Initializes a Device instance.

        @param device_id: A unique identifier for this device.

        @param sensor_data: A dictionary containing the device's local sensor readings.

        @param supervisor: The supervisor object responsible for managing the overall simulation.

        """

        self.device_id = device_id

        self.sensor_data = sensor_data

        self.supervisor = supervisor

        self.script_received = Event() # Event to signal that a script has been assigned.

        self.scripts = [] # List to store assigned scripts.

        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.

        self.thread = DeviceThread(self)

        self.thread.start()

        self.devices = [] # Will store a reference to all devices in the simulation.



    def __str__(self):

        """

        @brief Provides a string representation of the device.

        @return A string in the format "Device <device_id>".

        """

        return "Device %d" % self.device_id



    def setup_devices(self, devices):

        """

        @brief Sets up shared resources (global barrier and various locks) among all devices.

        Only the device with `device_id == 0` is responsible for initializing these resources,

        which are then distributed to all other devices.

        @param devices: A list of all Device instances in the simulation.

        Precondition: This method is called once during system setup.

        """

        self.devices = devices # Stores a reference to all devices.

        num_devices = len(devices)

        # Block Logic: The device with `device_id == 0` initializes the shared barrier and locks.

        # Invariant: A single `ReusableBarrier`, `script_lock`, and `data_lock` instance are

        # created and shared across all devices.

        if self.device_id == 0:

            self.timepoint_barrier = ReusableBarrier(num_devices)

            self.script_lock = Lock()

            self.data_lock = Lock()

            # Block Logic: Distributes the initialized shared resources to all other devices.

            for i in range(1, len(devices)):

                devices[i].data_lock = self.data_lock

                devices[i].script_lock = self.script_lock

                devices[i].timepoint_barrier = self.timepoint_barrier



    def assign_script(self, script, location):

        """

        @brief Assigns a script to the device for execution at a specific data `location`.

        If a script is provided, it's added to the queue and `script_received` is set.

        It also ensures a location-specific `Lock` is created and shared if one doesn't exist.

        If no script (i.e., `None`) is provided, it signals that the timepoint is done.

        @param script: The script object to assign.

        @param location: The data location relevant to the script.

        """

        if script is not None:

            self.scripts.append((script, location))

            self.script_received.set()

            # Block Logic: Ensures a location-specific lock exists and is shared across all devices.

            if not location in self.data_locks:

                lock = Lock()

                for dev in self.devices:

                    dev.data_locks[location] = lock

        else:

            # Block Logic: Signals completion of the timepoint if no script is assigned.

            self.timepoint_done.set()



    def get_data(self, location):

        """

        @brief Retrieves sensor data for a given location.

        Note: This method does not acquire any locks directly. It is expected that the calling

        `ScriptThread` will acquire the appropriate `data_locks[location]` before calling this method.

        @param location: The key identifying the sensor data.

        @return The data associated with the location, or `None` if the location is not found.

        """

        return self.sensor_data[location] if location in self.sensor_data \

            else None



    def set_data(self, location, data):

        """

        @brief Sets or updates sensor data for a specified location.

        Note: This method does not acquire any locks directly. It is expected that the calling

        `ScriptThread` will acquire the appropriate `data_locks[location]` before calling this method.

        @param location: The key for the sensor data to be modified.

        @param data: The new data value to store.

        Precondition: `location` must be a valid key in `self.sensor_data`.

        """

        if location in self.sensor_data:

            self.sensor_data[location] = data



    def shutdown(self):

        """

        @brief Shuts down the device's operational thread, waiting for its graceful completion.

        """

        self.thread.join()





class DeviceThread(Thread):

    """

    @brief The dedicated thread of execution for a `Device` instance.

    This thread manages the device's operational cycle, including fetching neighbor data,

    executing scripts concurrently via `ScriptThread` instances, and coordinating with

    other device threads using a shared `ReusableBarrier`.

    Time Complexity: O(T * S_total * (N * D_access + D_script_run)) where T is the number of timepoints,

    S_total is the total number of scripts executed by the device, N is the number of neighbors,

    D_access is data access time, and D_script_run is script execution time.

    """

    



    def __init__(self, device):

        """

        @brief Initializes a `DeviceThread` instance.

        @param device: The `Device` instance that this thread is responsible for.

        """

        Thread.__init__(self, name="Device Thread %d" % device.device_id)

        self.device = device

        self.script_threads = [] # List to keep track of active `ScriptThread` instances.



    def run(self):

        """

        @brief The main loop for the device's operational thread.

        Block Logic:

        1. Continuously fetches neighbor information from the supervisor.

           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.

        2. Waits for the `timepoint_done` event to be set, indicating that scripts are ready to be processed.

        3. Creates and starts a `ScriptThread` for each assigned script, allowing concurrent execution.

           Invariant: All scripts for the current timepoint are executed in parallel.

        4. Clears the `timepoint_done` event for the next cycle.

        5. Waits for all `ScriptThread` instances to complete.

        6. Synchronizes with all other `DeviceThread` instances using a shared `ReusableBarrier`.

           Invariant: All active `DeviceThread` instances must reach this barrier before any can

           progress to the next timepoint, ensuring synchronized advancement of the simulation.

        """

        while True:

            

            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:

                break



            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).

            self.device.timepoint_done.wait()



            # Block Logic: Creates and starts `ScriptThread` instances for each assigned script.

            # Invariant: Each script is executed concurrently in its own `ScriptThread`.

            for (script, location) in self.device.scripts:

                script_thread = ScriptThread(self.device, script, location, \

                    neighbours)

                script_thread.start()

                self.script_threads.append(script_thread)



            # Block Logic: Clears the `timepoint_done` event for the next timepoint cycle.

            self.device.timepoint_done.clear()

            # Block Logic: Waits for all initiated `ScriptThread` instances to complete their execution.

            for script_thread in self.script_threads:

                script_thread.join()

            self.script_threads = [] # Clears the list of threads.

            # Block Logic: Synchronizes with other device threads using a shared barrier,

            # ensuring all devices complete their processing before proceeding.

            self.device.timepoint_barrier.wait()



class ScriptThread(Thread):

    """

    @brief A dedicated thread for executing a single script for a specific data location.

    This thread is responsible for gathering data, running the script, and then

    propagating the results to relevant devices, ensuring thread-safe access to data

    through location-specific `Lock` objects.

    """

    



    def __init__(self, device, script, location, neighbours):

        """

        @brief Initializes a `ScriptThread` instance.

        @param device: The parent `Device` instance for which the script is being run.

        @param script: The script object to execute.

        @param location: The data location that the script operates on.

        @param neighbours: A list of neighboring `Device` instances.

        """

        Thread.__init__(self, name="Script Thread %d" % device.device_id)





        self.device = device

        self.script = script

        self.location = location

        self.neighbours = neighbours



    def run(self):

        """

        @brief The main execution logic for `ScriptThread`.

        Block Logic:

        1. Acquires the location-specific lock (`data_locks[self.location]`) to ensure exclusive access to that data.

        2. Collects data from neighboring devices and its own device for the specified `location`.

        3. Executes the assigned `script` if any data was collected.

        4. Acquires the shared `script_lock` to protect the data update phase.

        5. Propagates the script's `result` to neighboring devices and its own device.

        6. Releases the shared `script_lock`.

        7. Releases the location-specific lock.

        Invariant: All data access and modification for a given `location` are protected by a shared `Lock`.

        """

        # Block Logic: Acquires the location-specific lock to ensure exclusive access to data at this `location`.

        with self.device.data_locks[self.location]:

            script_data = []

            

            # Block Logic: Collects data from neighboring devices for the specified location.

            for device in self.neighbours:

                data = device.get_data(self.location)

                if data is not None:

                    script_data.append(data)

            

            # Block Logic: Collects data from its own device for the specified location.

            data = self.device.get_data(self.location)

            if data is not None:

                script_data.append(data)



            # Block Logic: Executes the script if any data was collected and propagates the result.

            if script_data != []:

                

                result = self.script.run(script_data)



                # Block Logic: Acquires the shared `script_lock` to protect the subsequent data updates.

                with self.device.script_lock:

                    

                    # Block Logic: Updates neighboring devices with the script's result.

                    for device in self.neighbours:

                        device.set_data(self.location, result)

                    

                    # Block Logic: Updates its own device's data with the script's result.

                    self.device.set_data(self.location, result)
