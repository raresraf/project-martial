"""
@file device.py
@brief Implements core components for a distributed device simulation.

This module defines the `Barrier`, `Device`, and `DeviceThread` classes,
which together simulate a network of interconnected devices processing sensor data
and executing scripts. The architecture employs class-level shared synchronization
primitives for global coordination across devices and threads.

Classes:
    - Barrier: A basic, non-reusable thread synchronization barrier.
    - Device: Represents an individual device in the simulation, managing its
              sensor data, script queues, and a pool of `DeviceThread`s. It utilizes
              global (class-level) barriers and locks for multi-device coordination.
    - DeviceThread: Worker threads associated with a `Device`. One is designated as
                    a "first" thread for specific coordination tasks, while others
                    act as regular script executors.
"""

from threading import Event, Thread, Condition, Lock


class Barrier(object):
    """
    @class Barrier
    @brief Implements a basic, non-reusable barrier for thread synchronization.

    This barrier allows a specified number of threads (`num_threads`) to wait
    for each other. Once all threads have arrived, they are all released
    simultaneously. It uses a `threading.Condition` object for blocking and
    releasing threads.
    """
    def __init__(self, num_threads=0):
        """
        @brief Initializes a Barrier instance.
        @param num_threads The total number of threads that must reach the barrier
                           before any of them can proceed. Defaults to 0.
        """
        self.num_threads = num_threads # Total number of threads expected at the barrier.
        self.count_threads = self.num_threads # Current count of threads waiting at the barrier.
        
        self.cond = Condition() # Condition variable for thread waiting and notification.

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier.

        The thread will be blocked until `num_threads` threads have called this method.
        The `count_threads` is reset after all threads are released, but this specific
        implementation does not explicitly handle reusability across multiple `wait` calls
        for the same `Barrier` instance within a continuous loop without external reset logic.
        """
        # Acquire the condition variable's lock.
        self.cond.acquire()
        self.count_threads -= 1 # Decrement the count of threads yet to arrive.
        if self.count_threads == 0: # If this is the last thread to arrive:
            self.cond.notify_all() # Release all waiting threads.
            self.count_threads = self.num_threads # Reset the count for the next use.
        else:
            self.cond.wait() # Wait for the last thread to arrive and notify.
        
        self.cond.release() # Release the condition variable's lock.

class Device(object):
    """
    @class Device
    @brief Represents a single device within the distributed simulation network.

    Manages its unique identifier, local sensor data, interaction with a supervisor,
    and a pool of `DeviceThread` instances for parallel processing. This class
    utilizes globally shared (class-level) synchronization primitives (`bariera_devices`
    and `locks`) for coordinating across multiple `Device` instances and protecting
    shared data.
    """
    
    # @var bariera_devices A class-level Barrier for synchronizing all Device instances globally.
    bariera_devices = Barrier()
    # @var locks A class-level list of Lock objects, where each lock protects sensor data
    #            at a specific location ID across all devices. This ensures mutual exclusion
    #            when multiple threads/devices access the same location's data.
    locks = []

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id Unique identifier for this device.
        @param sensor_data Initial sensor data pertinent to this device.
        @param supervisor A reference to the central supervisor managing the network.
        """
        self.device_id = device_id  # Unique identifier for the device.
        self.sensor_data = sensor_data  # Dictionary holding sensor data specific to this device.
        self.supervisor = supervisor  # Reference to the supervisor object for network coordination.

        # Instance attributes for script management.
        self.scripts = []  # List to store script objects assigned to this device.
        self.locations = []  # List to store location IDs corresponding to each script.
        
        self.nr_scripturi = 0  # Count of scripts assigned to this device for the current timepoint.
        # Index of the current script being processed by worker threads. Protected by `lock_script`.
        self.script_crt = 0

        # Event flag to signal that all scripts for a timepoint have been assigned.
        self.timepoint_done = Event()

        self.neighbours = []  # List of neighboring devices, updated by the supervisor.
        # Event flag to signal that neighbor information has been updated for the current timepoint.
        self.event_neighbours = Event()
        # Lock to protect `script_crt` and ensure atomic access to the script queue.
        self.lock_script = Lock()
        # Internal barrier for synchronizing the worker threads within this specific device.
        self.bar_thr = Barrier(8) # There are 8 worker threads per device.

        # Thread management: One "first" DeviceThread and 7 additional worker DeviceThreads.
        # The 'first' thread (first=1) has special responsibilities like updating neighbors
        # and clearing events.
        self.thread = DeviceThread(self, 1)
        self.thread.start()
        self.threads = []  # List to hold references to the additional worker DeviceThreads.
        for _ in range(7):
            tthread = DeviceThread(self, 0) # Worker threads have first=0.
            self.threads.append(tthread)
            tthread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures global (class-level) synchronization primitives for all devices.
        @param devices A list of all Device instances in the simulation.

        This method is called during the initial setup phase. It initializes the
        class-level `bariera_devices` with the total number of devices.
        It also initializes `Device.locks` (a list of `Lock` objects) if not already
        done, creating one lock per possible location to protect shared sensor data.
        """
        # Initialize the global barrier for all devices using the total number of devices.
        Device.bariera_devices = Barrier(len(devices))
        
        # Initialize the global list of locks if it's empty.
        if Device.locks == []:
            # Create one Lock object for each possible location, as determined by the supervisor's testcase.
            for _ in range(self.supervisor.supervisor.testcase.num_locations):
                Device.locks.append(Lock())

    def assign_script(self, script, location):
        """
        @brief Assigns a script and its corresponding location to the device's internal lists.
        @param script The script object to be executed.
        @param location The sensor data location that the script primarily operates on.

        If `script` is None, it acts as a signal that script assignment for the
        current timepoint is complete, and sets the `timepoint_done` event.
        """
        if script is not None:
            self.scripts.append(script)  # Add the script to the list of scripts to be executed.
            self.locations.append(location) # Add the associated location.
            
            self.nr_scripturi += 1 # Increment the count of assigned scripts.
        else:
            # If a None script is received, it signifies the end of script assignments for a timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        @param location The location identifier for which to retrieve data.
        @return The sensor data at the specified location, or None if not present.
        """
        return self.sensor_data[location] if location in \
        self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.
        @param location The location identifier for which to set data.
        @param data The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its main and all worker threads.
        Ensures all threads complete their execution before the program exits cleanly.
        """
        self.thread.join() # Join the "first" DeviceThread.
        for tthread in self.threads: # Join all other worker DeviceThreads.
            tthread.join()


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief Worker thread responsible for executing scripts and synchronizing within a Device.

    Each `Device` creates multiple `DeviceThread` instances. One is designated as
    the "first" thread (first=1), which handles specific coordination tasks
    like fetching neighbor information and clearing events. Other threads (first=0)
    focus on picking up and executing scripts from the shared queue.
    """
    

    def __init__(self, device, first):
        """
        @brief Initializes a DeviceThread instance.
        @param device The Device instance this thread belongs to.
        @param first A flag (1 for "first" thread, 0 for worker thread) indicating
                     special responsibilities for this thread.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device
        self.first = first

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        This loop continuously performs synchronization, script execution, and data
        management for its associated `Device`. The "first" thread handles global
        neighbor updates and event signaling, while all threads collectively
        process scripts and update sensor data, ensuring proper locking for shared resources.
        """
        while True:
            # Logic specific to the "first" thread of a device.
            # This thread handles fetching neighbors and resetting script counter for the timepoint.
            if self.first == 1:
                # Fetch updated neighbor information from the supervisor.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                # Reset the script counter for the new timepoint.
                self.device.script_crt = 0
                # Signal that neighbor information is updated and script counter is reset.
                self.device.event_neighbours.set()

            # Wait for the "first" thread to update neighbors and reset script counter.
            self.device.event_neighbours.wait()

            # If neighbors are None, it signals the end of the simulation.
            if self.device.neighbours is None:
                break

            # Wait for all scripts for the current timepoint to be assigned by the Device.
            self.device.timepoint_done.wait()

            # Loop to process scripts assigned to this device.
            while True:
                # Acquire a lock to safely get the next script index.
                self.device.lock_script.acquire()
                index = self.device.script_crt # Get the current script index.
                self.device.script_crt += 1 # Increment for the next worker.
                self.device.lock_script.release()

                # If all scripts for this timepoint have been assigned and picked up, break.
                if index >= self.device.nr_scripturi:
                    break

                # Get the script and its associated location.
                location = self.device.locations[index]
                script = self.device.scripts[index]

                # Acquire the global lock for this specific location to protect sensor data.
                Device.locks[location].acquire()

                script_data = [] # List to accumulate data needed by the script.
                    
                # Gather data from neighboring devices for the specified location.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from this device itself for the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # If any data was collected, run the script and update relevant devices.
                if script_data != []:
                    # Execute the script with the collected data.
                    result = script.run(script_data)

                    # Propagate the result to neighboring devices.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    # Update this device's own data with the script result.
                    self.device.set_data(location, result)

                # Release the global lock for this specific location.
                Device.locks[location].release()

            # Internal barrier for worker threads within this device to synchronize after script execution.
            self.device.bar_thr.wait()
            
            # Logic specific to the "first" thread of a device for cleaning up events.
            if self.first == 1:
                # Clear events for the next timepoint.
                self.device.event_neighbours.clear()
                self.device.timepoint_done.clear()
            # Second internal barrier for worker threads within this device.
            self.device.bar_thr.wait()
            
            # The "first" thread waits at the global device barrier to synchronize with all other devices.
            if self.first == 1:
                Device.bariera_devices.wait()

