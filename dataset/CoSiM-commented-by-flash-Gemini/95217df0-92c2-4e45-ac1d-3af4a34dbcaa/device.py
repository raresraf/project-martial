

"""
@95217df0-92c2-4e45-ac1d-3af4a34dbcaa/device.py
@brief Implements a multi-threaded simulation for distributed sensor devices with a thread pool for script execution.

This module defines the core components for simulating a network of sensor devices,
each capable of executing scripts, managing local sensor data, and interacting
with a central supervisor. This version employs two `ReusableBarrierSem` instances
for distinct synchronization phases and a thread pool managed by `ParallelScript`
instances for concurrent script execution.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate with neighbors under the
guidance of a supervisor.

Classes:
- Device: Represents a single simulated sensor device.
- ParallelScript: A worker thread responsible for executing individual scripts from a queue.
- DeviceThread: Manages the lifecycle and operation of a Device, including creating
                and managing the `ParallelScript` thread pool.

Domain: Distributed Systems Simulation, Concurrent Programming, Parallel Processing, Thread Pooling, Two-Phase Synchronization.
"""

from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierSem


class Device(object):
    """
    @brief Represents a single simulated sensor device in a distributed network.

    Each device manages its own sensor data, interacts with a supervisor,
    and executes assigned scripts in a multi-threaded environment. This version
    uses two global barriers for synchronization and an array of locks for
    location-specific data protection.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data, keyed by location.
        @param supervisor: A reference to the central supervisor managing all devices.
        """
        # Unique identifier for this device.
        self.device_id = device_id
        # Dictionary storing sensor data, keyed by location.
        self.sensor_data = sensor_data
        # Reference to the central supervisor.
        self.supervisor = supervisor
        # Event to signal that a script has been received for the current timepoint.
        self.script_received = Event()
        # List to store assigned scripts, each being a tuple of (script, location).
        self.scripts = []
        # Event to signal that all scripts for the current timepoint have been processed.
        self.timepoint_done = Event() # This event appears unused in the provided code.
        # The main thread responsible for the device's lifecycle.
        self.thread = DeviceThread(self)
        self.thread.start()
        
        # Global barrier for synchronizing timepoint progression across all devices.
        self.time_bar = None
        # Global barrier for synchronizing script reception and readiness across all devices.
        self.script_bar = None
        # Array of Locks, where each index corresponds to a location, protecting concurrent access to data for that location.
        self.devloc = []

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up global synchronization resources for all devices.

        This method initializes two `ReusableBarrierSem` instances (`time_bar` and `script_bar`)
        and an array of `Lock` objects (`devloc`) on device 0 (master) and distributes
        them to all other devices. The `devloc` array's size is determined by the
        maximum location index across all devices.

        @param devices: A list of all Device instances in the simulation.
        """
        # Block Logic: Device 0 acts as the master to initialize shared synchronization primitives.
        if self.device_id == 0:
            # Initializes the timepoint synchronization barrier for all devices.
            self.time_bar = ReusableBarrierSem(len(devices))
            # Initializes the script readiness synchronization barrier for all devices.
            self.script_bar = ReusableBarrierSem(len(devices))

            # Block Logic: Distributes the initialized barriers to all other devices.
            for device in devices:
                device.time_bar = self.time_bar
                device.script_bar = self.script_bar

            # Block Logic: Determines the maximum location index across all devices to size the `devloc` array.
            maxim = 0
            for device in devices:
                loc_list = device.sensor_data.keys()
                # Pre-condition: `loc_list` contains integer keys.
                # Invariant: `maxim` stores the highest location index encountered.
                if loc_list: # Ensure loc_list is not empty before accessing last element
                    loc_list.sort()
                    if loc_list[-1] > maxim:
                        maxim = loc_list[-1]

            # Block Logic: Populates the `devloc` array with Lock objects, one for each location up to `maxim`.
            while maxim >= 0:
                self.devloc.append(Lock())
                maxim -= 1

            # Block Logic: Distributes the initialized `devloc` array to all other devices.
            for device in devices:
                device.devloc = self.devloc


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed for a specific location on this device.

        If `script` is None, it signals that no more scripts are being assigned for
        the current timepoint, and triggers the `script_bar` wait.

        @param script: The script object to be executed, or None to signal completion.
        @param location: The location pertinent to the script execution.
        """
        if script is not None:
            # Pre-condition: `script` is not None.
            # Invariant: The script is added to the device's list of pending scripts.
            self.scripts.append((script, location))
        else:
            # Pre-condition: `script` is None, signaling end of script assignment for this timepoint.
            # Invariant: The `script_received` event is set, and the device waits on the `script_bar`
            # to synchronize with other devices that have also received their scripts.
            self.script_received.set()
            self.script_bar.wait()


    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The location for which to retrieve data.
        @return: The sensor data for the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location: The location for which to set data.
        @param data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's main thread.

        Waits for the device's main thread to complete its execution.
        """
        self.thread.join()


class ParallelScript(Thread):
    """
    @brief A worker thread in the thread pool, responsible for executing individual scripts.

    These threads acquire scripts from a shared queue (`to_procces`) and execute them,
    ensuring thread-safe access to location-specific data using `devloc` locks.
    """
    
    def __init__(self, device_thread):
        """
        @brief Initializes a ParallelScript worker thread.

        @param device_thread: A reference to the parent DeviceThread managing this pool.
        """
        Thread.__init__(self)
        self.device_thread = device_thread
    
    def run(self):
        """
        @brief The main execution loop for the ParallelScript thread.

        Pre-condition: `device_thread.sem_scripts` is released, indicating a script is available.
        Invariant: The thread continuously processes scripts from the queue until a `None`
                   node is encountered, signaling termination.
        """
        while True:
            # Block Logic: Waits on `sem_scripts` until a script is available in `to_procces`.
            self.device_thread.sem_scripts.acquire()
            
            # Retrieves the script execution node from the queue.
            nod = self.device_thread.to_procces[0]
            
            # Inline: Removes the processed node from the queue.
            del self.device_thread.to_procces[0]
            if nod is None:
                # Pre-condition: `nod` is None, indicating a shutdown signal for this worker thread.
                # Invariant: The loop breaks, terminating the worker thread.
                break
            
            # Unpacks the script execution details: neighbours, script, and location.
            neighbours, script, location = nod[0], nod[1], nod[2]

            # Critical Section: Acquires a lock for the specific location to ensure exclusive
            # access to the data associated with this location during script execution and data updates.
            self.device_thread.device.devloc[location].acquire()

            script_data = []

            # Block Logic: Gathers data from neighboring devices for the current location.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Gathers data from its own sensor_data for the current location.
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Executes the script with the collected data.
                result = script.run(script_data)

                # Block Logic: Updates data on neighboring devices with the script's result.
                for device in neighbours:
                    device.set_data(location, result)
                
                # Updates its own data with the script's result.
                self.device_thread.device.set_data(location, result)

            # Releases the lock for the specific location, allowing other threads to access it.
            self.device_thread.device.devloc[location].release()


class DeviceThread(Thread):
    """
    @brief Manages the lifecycle for a Device, orchestrating script execution through a thread pool.

    This thread is responsible for initializing and managing a pool of `ParallelScript`
    worker threads, queuing scripts for execution, and coordinating synchronization
    across timepoints using global barriers.
    """
    
    def create_pool(self, device_thread):
        """
        @brief Creates and starts a pool of `ParallelScript` worker threads.

        @param device_thread: A reference to the current DeviceThread instance.
        @return: A list of `ParallelScript` thread instances that constitute the thread pool.
        """
        pool = []
        for _ in xrange(self.numar_procesoare):
            aux_t = ParallelScript(device_thread)
            pool.append(aux_t)
            # Starts each worker thread in the pool.
            aux_t.start()
        return pool


    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.

        @param device: The Device instance that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Semaphore used to signal the availability of scripts in `to_procces` queue to worker threads.
        self.sem_scripts = Semaphore(0)
        # Defines the size of the thread pool (number of concurrent worker threads).
        self.numar_procesoare = 8 
        # Initializes the thread pool.
        self.pool = self.create_pool(self)
        
        # A list acting as a queue for scripts to be processed by `ParallelScript` threads.
        self.to_procces = []


    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Pre-condition: The device's synchronization mechanisms are properly set up.
        Invariant: The device continuously processes timepoints, queues assigned scripts
                   for execution by the thread pool, and synchronizes with other devices
                   until a shutdown signal is received from the supervisor.
        """
        while True:
            # Block Logic: Fetches the current neighbors of this device from the supervisor.
            # This allows for dynamic network topology changes between timepoints.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Pre-condition: `neighbours` is None, indicating a termination signal from the supervisor.
                # Invariant: The thread pool workers are signaled to terminate, and then this thread exits.
                # Signals each worker thread in the pool to terminate by appending `None` to the queue.
                for _ in range(self.numar_procesoare):
                    self.to_procces.append(None)
                    self.sem_scripts.release()
                # Waits for all worker threads in the pool to complete their execution.
                for item in self.pool:
                    item.join()
                break # Exits the main loop, terminating DeviceThread.
            
            # Block Logic: Waits for the supervisor to signal that scripts for the current
            # timepoint have been received and assigned to this device.
            self.device.script_received.wait()
            
            # Block Logic: Enqueues each assigned script into the `to_procces` list and
            # signals the `sem_scripts` semaphore, making them available to worker threads.
            for (script, location) in self.device.scripts:
                # Constructs a node containing execution context for the script.
                nod = (neighbours, script, location)
                self.to_procces.append(nod)
                # Releases the semaphore, allowing a worker thread to pick up this script.
                self.sem_scripts.release()

            # Block Logic: Waits on the `script_bar` to ensure all devices have finished
            # queuing their scripts for the current timepoint.
            self.device.script_bar.wait()

            # Block Logic: Waits on the `time_bar` to ensure all devices have completed
            # processing their scripts for the current timepoint before proceeding.
            self.device.time_bar.wait()
            
            # Resets the `script_received` event, preparing for the next timepoint.
            self.device.script_received.clear()
