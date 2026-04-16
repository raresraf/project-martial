


"""
@file device.py
@brief Implements a simulated distributed device with concurrent processing capabilities.

This module defines the `Device` class, representing a node in a distributed system or sensor network.
Each device manages its sensor data, interacts with a supervisor, and executes assigned scripts
in parallel using a pool of worker threads. The module leverages various Python threading
primitives such as `Event`, `Lock`, `Semaphore`, and a custom `ReusableBarrierSem` for
synchronization and coordination among threads and devices.

Domain: Distributed Systems, Concurrency, Multithreading, Simulation, Sensor Networks, Synchronization.
"""

from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierSem


class Device(object):
    """
    @brief Represents a single device (node) in a simulated distributed system.

    Each Device instance manages its own sensor data, interacts with a supervisor,
    and can execute scripts in a parallel fashion. It also handles synchronization
    with other devices for coordinated actions.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.

        Parameters:
          - device_id: A unique identifier for this device.
          - sensor_data: A dictionary containing initial sensor data for various locations.
          - supervisor: A reference to the central supervisor responsible for coordinating devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when a script has been assigned.
        self.scripts = []              # List to store assigned scripts (script, location).
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's processing.

        # Initialize and start the dedicated thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()
        
        self.time_bar = None    # Reusable barrier for synchronizing device timepoints.
        self.script_bar = None  # Reusable barrier for synchronizing script assignment across devices.
        self.devloc = []        # List of locks, one for each sensor data location, for granular access control.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization primitives across all devices.

        This method is intended to be called by a coordinating entity (e.g., the supervisor
        or device_id 0 itself) to initialize global synchronization barriers and
        per-location locks that will be shared among all devices.

        Parameters:
          - devices: A list of all Device instances in the system.
        """
        
        # Block Logic: Only device with ID 0 is responsible for initializing shared barriers.
        if self.device_id == 0:
            
            # Initialize reusable barriers for timepoint and script synchronization.
            self.time_bar = ReusableBarrierSem(len(devices))
            self.script_bar = ReusableBarrierSem(len(devices))

            # Assign the newly created barriers to all devices in the system.
            for device in devices:
                device.time_bar = self.time_bar
                device.script_bar = self.script_bar

            # Block Logic: Determine the maximum location index to initialize enough locks.
            maxim = 0
            for device in devices:
                loc_list = list(device.sensor_data.keys()) # Convert dict_keys to list for sorting.
                if loc_list: # Ensure loc_list is not empty before sorting.
                    loc_list.sort()
                    if loc_list[-1] > maxim:
                        maxim = loc_list[-1]

            # Initialize a list of Locks, one for each possible sensor data location.
            # This ensures granular thread-safe access to sensor data at different locations.
            while maxim >= 0:
                self.devloc.append(Lock())
                maxim -= 1

            # Assign the newly created list of locks to all devices.
            for device in devices:
                device.devloc = self.devloc


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific sensor data location.

        If a script is provided, it is added to the device's script queue.
        If `script` is None, it acts as a signal for the device to proceed,
        signaling `script_received` and waiting on `script_bar`.

        Parameters:
          - script: The script object to execute, or None to signal completion.
          - location: The sensor data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If no script, signal that script assignment is complete for this timepoint.
            self.script_received.set()
            # Wait for all other devices to also finish their script assignments
            # before proceeding to the execution phase.
            self.script_bar.wait()


    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.

        Parameters:
          - location: The identifier for the sensor data location.

        Returns:
          - The sensor data at the given location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets (updates) sensor data for a specified location.

        Parameters:
          - location: The identifier for the sensor data location.
          - data: The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device thread.

        Joins the device's dedicated thread, ensuring all its operations are
        completed before the main program exits.
        """
        self.thread.join()



class ParallelScript(Thread):
    """
    @brief A worker thread responsible for executing a single script within a device's context.

    Instances of this class are part of a thread pool managed by `DeviceThread`.
    They acquire tasks (scripts) from a shared queue, execute them, and update
    sensor data, ensuring thread-safe access to shared resources.
    """
    
    def __init__(self, device_thread):
        """
        @brief Initializes a ParallelScript worker thread.

        Parameters:
          - device_thread: A reference to the parent `DeviceThread` instance that manages this worker.
        """
        Thread.__init__(self)
        self.device_thread = device_thread
    def run(self):
        """
        @brief The main execution loop for the ParallelScript worker.

        This method continuously attempts to acquire a script execution task from
        the `DeviceThread`'s queue. If a task is available (not None), it
        executes the script, reads and updates sensor data for the relevant location,
        and ensures that shared data access is synchronized using per-location locks.
        The loop terminates when a `None` task is received, signaling shutdown.
        """
        while True:
            # Pre-condition: Wait until a script task is available in the queue.
            self.device_thread.sem_scripts.acquire()
            
            # Block Logic: Retrieves the next script task from the queue.
            # Invariant: `to_procces` is a list-like structure from which tasks are dequeued.
            nod = self.device_thread.to_procces[0]
            
            # Inline: Removes the processed task from the queue.
            del self.device_thread.to_procces[0]
            # Termination condition: If `None` is received, it's a signal to shut down.
            if nod is None:
                break
            
            # Unpack the task components: neighboring devices, script object, and location.
            neighbours, script, location = nod[0], nod[1], nod[2]


            
            # Pre-condition: Acquire the lock for the specific sensor data location
            # to ensure exclusive access during read/write operations for this location.
            self.device_thread.device.devloc[location].acquire()

            script_data = [] # Stores data collected for the script's execution.

            # Block Logic: Collects sensor data from neighboring devices for the given location.
            # Invariant: `script_data` will contain relevant sensor readings from neighbors.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Collect sensor data from the current device itself for the given location.
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Pre-condition: Only execute the script if there is data to process.
            if script_data != []:
                
                # Execute the script with the collected data.
                result = script.run(script_data)

                # Block Logic: Update sensor data on neighboring devices with the script's result.
                # Invariant: Neighbors' sensor data at `location` is updated.
                for device in neighbours:
                    device.set_data(location, result)
                
                # Update sensor data on the current device with the script's result.
                self.device_thread.device.set_data(location, result)

            # Post-condition: Release the lock for the sensor data location.
            self.device_thread.device.devloc[location].release()



class DeviceThread(Thread):
    """
    @brief The main thread for a Device, managing its script execution and synchronization.

    This thread is responsible for fetching neighbor information, receiving scripts
    assigned to its parent `Device` instance, queuing these scripts for parallel
    execution by a pool of `ParallelScript` worker threads, and coordinating
    its progress with other `DeviceThread` instances using synchronization barriers.
    """
    
    def create_pool(self, device_thread):
        """
        @brief Creates and starts a pool of `ParallelScript` worker threads.

        These worker threads will execute the assigned scripts concurrently.

        Parameters:
          - device_thread: A reference to the current `DeviceThread` instance,
                           passed to the worker threads so they can access
                           shared resources and synchronization primitives.

        Returns:
          - A list of `ParallelScript` worker thread instances.
        """
        pool = []
        # Block Logic: Initializes a fixed number of `ParallelScript` worker threads.
        for _ in xrange(self.numar_procesoare):
            aux_t = ParallelScript(device_thread)
            pool.append(aux_t)
            
            # Inline: Starts the worker thread.
            aux_t.start()
        return pool


    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.

        Parameters:
          - device: A reference to the parent `Device` instance this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Semaphore to control access to script tasks; initially 0 as no tasks are available.
        self.sem_scripts = Semaphore(0)
        self.numar_procesoare = 8 # Number of worker threads in the pool.
        self.pool = self.create_pool(self) # Initialize the pool of worker threads.
        
        self.to_procces = [] # Queue for script tasks to be processed by worker threads.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        This method continuously:
        1. Fetches information about neighboring devices from the supervisor.
        2. Waits for scripts to be assigned by the supervisor.
        3. Queues the assigned scripts for parallel execution by its worker pool.
        4. Coordinates with other device threads using synchronization barriers
           for script assignment and timepoint completion.
        5. Shuts down its worker pool when a termination signal is received from the supervisor.
        """
        while True:
            # Get the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If `neighbours` is None, it signals the shutdown of the simulation.
            if neighbours is None:
                
                # Block Logic: Signals all worker threads in the pool to terminate.
                # Invariant: Each worker receives a `None` task and terminates gracefully.
                for _ in range(self.numar_procesoare):
                    self.to_procces.append(None) # Enqueue a None task for each worker.
                    self.sem_scripts.release()  # Release the semaphore to unblock workers.
                # Wait for all worker threads to finish their execution.
                for item in self.pool:
                    item.join()
                break # Exit the main loop, terminating the DeviceThread.
            
            # Wait until the supervisor has assigned scripts for the current timepoint.
            self.device.script_received.wait()
            
            # Block Logic: Enqueues assigned scripts for processing by worker threads.
            # Invariant: Each (script, location) pair is added to `to_procces` queue
            # and the semaphore is released, making the task available to a worker.
            for (script, location) in self.device.scripts:
                
                # Inline: Create a task node containing neighbours, script, and location.
                nod = (neighbours, script, location)
                
                self.to_procces.append(nod) # Add the task to the queue.
                
                # Inline: Release the semaphore to allow a worker thread to pick up the task.
                self.sem_scripts.release()

            # Wait at the script barrier until all device threads have processed their assigned scripts.
            # This synchronizes the completion of script assignment across all devices.
            self.device.script_bar.wait()

            # Wait at the time barrier until all device threads have completed processing
            # for the current timepoint. This synchronizes overall time progression.
            self.device.time_bar.wait()
            
            # Reset the script received event for the next timepoint.
            self.device.script_received.clear()

