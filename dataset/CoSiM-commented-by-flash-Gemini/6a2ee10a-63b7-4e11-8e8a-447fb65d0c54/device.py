"""
@file device.py
@brief Implements core components for a distributed system, likely a simulation or sensor network.
This module defines Device objects that can communicate and process data concurrently
using multiple threads per device, employing synchronization primitives like locks,
events, semaphores, and reusable barriers for coordinated operation and data consistency.
"""

from threading import *


class ReusableBarrier(object):
    """
    @brief Implements a reusable barrier for thread synchronization.
    This barrier ensures that a specified number of threads all reach a certain point
    before any of them are allowed to proceed. It uses semaphores and a lock to
    manage the waiting and releasing of threads across two phases for reusability.
    """

    def __init__(self, numOfTh):
        """
        @brief Initializes the reusable barrier.

        @param numOfTh (int): The total number of threads that must reach the barrier.
        """
        self.numOfTh = numOfTh
        # Stores counts and semaphores for two phases, allowing the barrier to be reused.
        self.threads = [{}, {}]
        self.threads[0]['count'] = numOfTh  # Counter for threads in phase 0.
        self.threads[1]['count'] = numOfTh  # Counter for threads in phase 1.
        self.threads[0]['sem'] = Semaphore(0)  # Semaphore for threads waiting in phase 0.
        self.threads[1]['sem'] = Semaphore(0)  # Semaphore for threads waiting in phase 1.
        self.lock = Lock()  # Lock to protect access to the counters.

    def wait(self):
        """
        @brief Blocks the calling thread until all 'numOfTh' threads have reached the barrier
        and then allows them to proceed. This method executes both phases of the barrier.
        """
        # Block Logic: Iterates through the two phases of the barrier.
        for i in range(0, 2):
            with self.lock:  # Ensures exclusive access to the counter.
                self.threads[i]['count'] -= 1  # Decrements the thread count for the current phase.
                # Conditional Logic: If this is the last thread to reach the barrier in this phase.
                if self.threads[i]['count'] == 0:
                    # Releases all waiting threads from the semaphore.
                    for _ in range(self.numOfTh):
                        self.threads[i]['sem'].release()
                    # Resets the counter for the next use of this phase.
                    self.threads[i]['count'] = self.numOfTh
            self.threads[i]['sem'].acquire()  # Threads wait here until released by the last thread.


class Device(object):
    """
    @brief Represents a single device in the distributed system.
    Each device has a unique ID, manages its sensor data, and interacts with a supervisor.
    It utilizes a pool of `DeviceThread` instances for concurrent script processing,
    employing various synchronization mechanisms.
    """

    # These class variables are designed to be shared across all Device instances.
    tBariera = None  # A global barrier for all devices (initialized by device_id 0).
    locks = []  # A global list of RLock objects, typically one per data location.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id (int): A unique identifier for the device.
        @param sensor_data (dict): A dictionary holding sensor readings for different locations.
        @param supervisor (Supervisor): A reference to the central supervisor managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.scripts = []  # List to hold (script, location) tuples assigned to this device.
        self.timepoint_done = Event()  # Event to signal when script assignment for a timepoint is complete.

        self.threads = []  # List to hold DeviceThread instances for this device.
        self.no_threads = 8  # Fixed number of DeviceThread instances per device.
        
        self.sLock = Lock()  # Lock to protect access to self.scripts.
        self.iBariera = ReusableBarrier(8)  # A local barrier for the 8 DeviceThread instances within this Device.

        self.etLock = Lock()  # Lock to protect access to `timepoint_done` event (used with acquire/release instead of .wait()/.set()).
        self.lastScripts = []  # List to store scripts that were processed in the previous iteration (for retry/re-processing).

        # Conditional Logic: Device 0 has an additional initialization event for global setup coordination.
        if device_id == 0:
            self.init_event = Event()  # Event to signal completion of Device 0's initialization.

        # Block Logic: Creates and initializes 8 DeviceThread instances for this Device.
        for tid in range(self.no_threads):
            thread = DeviceThread(self, tid)
            self.threads.append(thread)

    def __str__(self):
        """
        @brief Provides a string representation of the Device.

        @return str: A formatted string indicating the device ID.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs global setup tasks, including initializing the global barrier and locks.
        Device 0 coordinates this setup.

        @param devices (list): A list of all Device instances in the system.
        """
        # Conditional Logic: Auxiliary devices wait for Device 0 to complete global setup.
        if self.device_id != 0:
            i = 0
            while (i < len(devices) and devices[i].device_id != 0):
                i += 1
            if i < len(devices):
                # Waits for Device 0's initialization event to be set.
                devices[i].init_event.wait()
                # Copies the globally initialized barrier and locks from Device 0.
                Device.tBariera = devices[i].tBariera
                Device.locks = devices[i].locks
        else:
            aux = 0
            # Device 0 initializes the global barrier with the total number of devices.
            Device.tBariera = ReusableBarrier(len(devices))

            # Calculates the total number of sensor data locations across all devices (potentially a simplification).
            for d in devices:
                aux += len(d.sensor_data)
            # Initializes a global list of RLock objects, one for each data location.
            Device.locks = [RLock() for _ in range(aux)]
            # Signals that Device 0's initialization is complete.
            self.init_event.set()

        # Block Logic: Starts all DeviceThread instances for this Device after setup is complete.
        for thread in self.threads:
            thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        @param script (callable): The script (function or object with a run method) to execute.
                                  If None, it signals that script assignment for the timepoint is done.
        @param location (int): The identifier of the data location the script operates on.
        """
        # Conditional Logic: If a script is provided, it's added to the scripts list.
        if script is None:
            self.etLock.acquire() # Acquires a lock before setting the timepoint_done event.
            self.timepoint_done.set() # Signals that script assignment for the timepoint is complete.
        else:
            self.sLock.acquire() # Acquires a lock to protect the scripts list.
            self.scripts.append((script, location)) # Appends the script and its location.
            self.sLock.release() # Releases the lock.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location, acquiring its associated lock.

        @param location (int): The identifier of the data location.
        @return any: The sensor data at the specified location, or None if not found.
        """
        # Conditional Logic: Acquires the location-specific lock if the location exists in sensor_data.
        if location in self.sensor_data:
            Device.locks[location].acquire() # Note: RLock allows multiple acquisitions by the same thread.

        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location, releasing its associated lock.

        @param location (int): The identifier of the data location to update.
        @param data (any): The new data value.
        """
        # Conditional Logic: Updates data and releases the location-specific lock if the location exists in sensor_data.
        if location in self.sensor_data:
            self.sensor_data[location] = data
            Device.locks[location].release() # Releases the lock acquired in get_data.

    def shutdown(self):
        """
        @brief Shuts down all DeviceThread instances associated with this device.
        """
        # Block Logic: Joins all DeviceThread instances to ensure their completion.
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    @brief A thread of execution dedicated to a Device. Multiple instances of this
    thread run concurrently for a single Device, cooperatively processing scripts.
    It handles fetching neighbor information and participating in synchronization.
    """

    def __init__(self, device, thread_id):
        """
        @brief Initializes a new DeviceThread instance.

        @param device (Device): The Device object this thread is responsible for.
        @param thread_id (int): A unique identifier for this thread instance within its Device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        @brief The main execution loop for a DeviceThread.
        This thread participates in fetching neighbors (only if thread_id is 0),
        cooperatively processes scripts by taking them from a shared list,
        and synchronizes with other threads within the same Device and with other Devices.
        """
        # Conditional Logic: Only the thread with thread_id 0 is responsible for fetching neighbors.
        if self.thread_id == 0:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            # Conditional Logic: Removes the device itself from its list of neighbors if present.
            if self.device in self.device.neighbours:
                self.device.neighbours.remove(self.device)

        # Main processing loop for timepoints.
        while True:
            # Synchronization: All 8 DeviceThreads within a single Device synchronize.
            self.device.iBariera.wait()
            
            # Conditional Logic: Checks if the supervisor has signaled a shutdown.
            neighbours = self.device.neighbours
            if neighbours is None:
                break # Terminates the thread if supervisor signals shutdown.

            # Block Logic: Cooperatively processes scripts from the device's shared list.
            while len(self.device.scripts) != 0:
                script = None

                self.device.sLock.acquire() # Acquires lock to safely access shared scripts list.
                # Conditional Logic: Retrieves a script if the list is not empty.
                if len(self.device.scripts) != 0:
                    script, location = self.device.scripts.pop(0) # Atomically pops a script.
                    self.device.lastScripts.append((script, location)) # Stores for potential re-processing.
                self.device.sLock.release() # Releases the lock.

                # Conditional Logic: If a script was successfully retrieved.
                if script:
                    script_data = []

                    # Block Logic: Collects data from neighboring devices.
                    for device in neighbours:
                        # Calls get_data, which acquires the location-specific lock.
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                    # Collects data from its own device.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    # Conditional Logic: If any data was collected, executes the script and propagates results.
                    if script_data != []:
                        result = script.run(script_data)

                        # Block Logic: Propagates the new data to all neighboring devices.
                        for device in neighbours:
                            # Calls set_data, which updates data and releases the location-specific lock.
                            device.set_data(location, result)

                        # Updates data on its own device.
                        self.device.set_data(location, result)
            
            # Synchronization: Waits for scripts assignment for the current timepoint to be done.
            self.device.timepoint_done.wait()

            # The block of code below this line appears to be a duplicate of the script processing logic above.
            # This might indicate a re-processing phase or a bug in the original code structure.
            # It seems to be processing the scripts list again after timepoint_done is set.
            while len(self.device.scripts) != 0: # Duplicate logic.
                script = None

                self.device.sLock.acquire()
                if len(self.device.scripts) != 0:

                    script, location = self.device.scripts.pop(0)
                    self.device.lastScripts.append((script, location))
                self.device.sLock.release()

                if script:
                    script_data = []

                    for device in neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        result = script.run(script_data)



                        for device in neighbours:
                            device.set_data(location, result)

                        self.device.set_data(location, result)

            # Synchronization: All 8 DeviceThreads within this Device wait at the local barrier again.
            self.device.iBariera.wait()

            # Conditional Logic: Only the thread with thread_id 0 coordinates global barrier and resets.
            if self.thread_id == 0:
                Device.tBariera.wait() # Global synchronization among all Devices.

                # Fetches neighbors again for the next timepoint.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                if self.device.neighbours and self.device in self.device.neighbours:
                    self.device.neighbours.remove(self.device)

                self.device.timepoint_done.clear() # Clears event for next timepoint.
                # Releases a lock acquired in assign_script when script is None.
                self.device.etLock.release()
