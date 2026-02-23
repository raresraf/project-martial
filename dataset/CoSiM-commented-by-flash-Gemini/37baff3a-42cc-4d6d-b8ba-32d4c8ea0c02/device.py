"""
@37baff3a-42cc-4d6d-b8ba-32d4c8ea0c02/device.py
@brief Implements a distributed device simulation framework where scripts are processed by worker threads
using a work-stealing approach, synchronized by a reusable barrier and location-specific semaphores.
This variant introduces a dynamic work distribution for script execution to enhance parallelism.
* Algorithm: Master-Worker pattern for script execution with a shared queue (implicit via `getWork`),
             and barrier synchronization for timepoint progression.
* Concurrency: Uses `threading.Thread`, `threading.Lock`, `threading.Semaphore`, and
               an external `ReusableBarrierCond` for synchronization and task management.
"""

from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierCond # Assuming ReusableBarrierCond is a standard condition-variable based barrier.


class Device(object):
    """
    @brief Represents a simulated device, responsible for managing sensor data,
    assigning and coordinating script execution across multiple worker threads,
    and synchronizing with other devices.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor readings relevant to this device.
        @param supervisor: A reference to the supervisor object managing the devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data # Stores the device's sensor data.
        self.supervisor = supervisor
        self.script_received = Event() # Signals when a script has been assigned.
        self.scripts = [] # List to store assigned (script, location) tuples.
        self.thread = DeviceThread(self) # The main thread for this device.
        self.loopBarrier = None # Shared barrier for device-wide synchronization (initialized by supervisor).
        self.locationSemaphores = None # Dictionary to store semaphores for specific data locations.
        self.thread.start() # Start the device's main thread immediately.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization mechanisms (barrier and location semaphores)
        among all devices. This method is typically called by a supervisor or coordinator.
        @param devices: A list of all Device objects in the simulation.
        """
        # Block Logic: Initialize a shared loop barrier and a dictionary for location semaphores.
        # Invariant: These shared objects are then distributed to all participating devices.
        loopBarrier = ReusableBarrierCond(len(devices)) # Create a new reusable barrier for all devices.
        locationSemaphores = {} # Initialize a shared dictionary for location-specific semaphores.
        for device in devices :
            device.loopBarrier = loopBarrier # Assign the shared barrier.
            device.locationSemaphores = locationSemaphores # Assign the shared semaphore dictionary.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific data location.
        Initializes a semaphore for the location if one does not already exist.
        @param script: The script object to be executed.
        @param location: The data location relevant to the script.
        """
        # Pre-condition: A script is provided; `None` is used as a signal in some implementations.
        if script is not None:
            self.scripts.append((script, location))
            # Block Logic: Ensure a semaphore exists for this data location for mutual exclusion.
            if self.locationSemaphores.get(location) is None:
                self.locationSemaphores[location] = Semaphore() # Create a new semaphore if not present.
        else:
            # Invariant: If script is None, it typically signals that script assignment for the current timepoint is complete.
            self.script_received.set() # Signal that all scripts have been received for this timepoint.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key for the sensor data.
        @return: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.
        @param location: The key for the sensor data.
        @param data: The new data to be set.
        """
        # Pre-condition: `location` must exist in `sensor_data` to be updated.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's associated thread.
        """
        self.thread.join() # Wait for the device's main thread to complete its execution.


class DeviceThread(Thread):
    """
    @brief The main thread for a Device, responsible for managing timepoint progression,
    fetching neighbors, and coordinating the execution of `ScriptThread` workers.
    It implements a simple work distribution mechanism for assigned scripts.
    """
    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device: The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workLock = Lock() # Lock to protect access to `lastScriptGiven`.
        self.lastScriptGiven = 0; # Index to track the next script to be distributed.

    def run(self):
        """
        @brief The main execution loop for the device thread.
        Handles fetching neighbors, distributing scripts to worker threads,
        waiting for workers to complete, and synchronizing with other devices.
        """
        # Invariant: Loop continuously until the simulation signals termination.
        while True:
            # Retrieve neighbors from the supervisor for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If `neighbours` is None, it signals the end of the simulation.
            if neighbours is None:
                break # Exit the loop and terminate the thread.

            # Reset the index for script distribution for the new timepoint.
            self.lastScriptGiven = 0;

            # Wait until all scripts for the current timepoint have been assigned to this device.
            self.device.script_received.wait()
            
            # Block Logic: Create and start a pool of ScriptThread workers.
            workers = []
            workLock = Lock() # A local lock for worker thread coordination (possibly redundant if workLock is intended to be shared).

            # Spawn up to 8 worker threads, or fewer if there are fewer scripts.
            for i in range(0, min(8, len(self.device.scripts))):
                worker = ScriptThread(self, neighbours, workLock) # Pass this DeviceThread as master.
                workers.append(worker)

            # Invariant: Start all worker threads concurrently.
            for worker in workers:
                worker.start()

            # Invariant: Wait for all worker threads to complete their execution.
            for worker in workers:
                worker.join()

            # Reset the script_received event for the next timepoint.
            self.device.script_received.clear()

            # Synchronize all device threads at the loop barrier before proceeding.
            self.device.loopBarrier.wait()

    def getWork(self):
        """
        @brief Distributes the next available script to a worker thread.
        This method ensures that each script is handed out exactly once.
        @return: A (script, location) tuple if available, otherwise None.
        """
        script = None
        # Pre-condition: Acquire workLock to safely access `lastScriptGiven`.
        with self.workLock:
            # Invariant: If there are still scripts to give out, return the next one.
            if (self.lastScriptGiven < len(self.device.scripts)):
                script = self.device.scripts[self.lastScriptGiven]
                self.lastScriptGiven += 1

        return script

class ScriptThread(Thread) :
    """
    @brief A worker thread that fetches a script from its master (DeviceThread),
    acquires a location-specific semaphore, collects data, executes the script,
    updates data, and releases the semaphore. It then attempts to get more work.
    """
    def __init__(self, master, neighbours, workLock):
        """
        @brief Initializes a ScriptThread.
        @param master: The DeviceThread instance that created this worker.
        @param neighbours: A list of neighboring devices for data collection.
        @param workLock: A lock used for coordinating work distribution with the master.
        """
        Thread.__init__(self)
        self.master = master
        self.neighbours = neighbours
        self.workLock = workLock # This lock is passed from DeviceThread, likely intended for getWork.

    def run(self) :
        """
        @brief The main execution loop for the script worker thread.
        Continuously gets scripts, processes them, and attempts to fetch new work.
        """
        # Pre-condition: Acquire workLock to safely get a script from the master.
        self.workLock.acquire()
        scriptLocation = self.master.getWork() # Get the initial script task.
        self.workLock.release()

        # Invariant: Continue processing scripts until `getWork` returns None (no more scripts).
        while scriptLocation is not None:
            (script, location) = scriptLocation
            script_data = []
            
            # Pre-condition: Acquire the semaphore for the specific data `location` to ensure exclusive access.
            self.master.device.locationSemaphores.get(location).acquire()
            # Block Logic: Collect data from neighboring devices for the current `location`.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Collect data from the current device itself for the current `location`.
            data = self.master.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Pre-condition: Execute the script only if relevant data was collected.
            if script_data != []:
                # Execute the script with the collected data.
                result = script.run(script_data)

                # Disseminate the script result to neighboring devices.
                for device in self.neighbours:
                    device.set_data(location, result)
                
                # Update the current device's data with the script result.
                self.master.device.set_data(location, result)

            # Release the semaphore for the current data `location`.
            self.master.device.locationSemaphores.get(location).release()
            
            # Pre-condition: Acquire workLock again to safely get the next script.
            self.workLock.acquire()
            scriptLocation = self.master.getWork() # Attempt to get more work.
            self.workLock.release()