

"""
This module implements a device simulation framework with a focus on
distributed script execution and synchronization. It defines `Device` objects,
`DeviceThread` for managing device operations, and `Worker` threads
for executing scripts on sensor data.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
from worker import Worker


class Device(object):
    """
    Represents a simulated device in a distributed system. Each device
    manages its own sensor data, executes scripts, and synchronizes
    with other devices through a supervisor and shared barrier/locks.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor data, keyed by location.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.devices = []
        self.cores = 8 # Represents the number of worker threads to be used by this device
        self.barrier = None
        self.shared_locks = []
        self.timepoint_done = Event() # Event to signal completion of timepoint script assignment
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """
        Sets the shared barrier for device synchronization.

        Args:
            barrier (ReusableBarrierSem): The barrier object for synchronization.
        """
        self.barrier = barrier

    def set_locks(self, locks):
        """
        Sets the list of shared locks for protecting sensor data access.

        Args:
            locks (list): A list of Lock objects, one for each possible location.
        """
        self.shared_locks = locks

    def setup_devices(self, devices):
        """
        Configures the device's awareness of other devices in the system
        and initializes shared synchronization primitives (barrier and locks).
        Device 0 is responsible for initializing these shared resources.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        self.devices = devices
        
        # Block Logic: Device 0 acts as the coordinator to initialize the shared barrier.
        if self.device_id == 0:
            # Functional Utility: `ReusableBarrierSem` ensures all device threads
            # reach a specific state before proceeding, using a semaphore-based implementation.
            lbarrier = ReusableBarrierSem(len(devices))
            # Block Logic: Propagates the initialized barrier to all other devices.
            for dev in devices:
                dev.set_barrier(lbarrier)

        # Invariant: `max_loc` identifies the highest numerical identifier used for sensor locations.
        max_loc = max(self.sensor_data.keys(), key=int)
        
        # Block Logic: Initializes shared locks for sensor data locations if they haven't been
        # initialized yet or if the number of locations exceeds the current lock array size.
        if  max_loc+1 > len(self.shared_locks):
            llocks = []
            # Block Logic: Creates a Lock object for each potential sensor data location.
            for _ in range(max_loc+1):
                llocks.append(Lock())
            self.set_locks(llocks)
            # Block Logic: Propagates the newly created shared locks to all devices.
            for dev in self.devices:
                dev.set_locks(llocks)

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location for this device.
        If no script is provided (None), it signals that the timepoint is done.

        Args:
            script (Script or None): The script object to assign, or None if the timepoint is complete.
            location (int): The numerical identifier for the location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Functional Utility: Signals completion of script assignment for the current timepoint,
            # allowing the DeviceThread to proceed with script execution.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location for which to retrieve data.

        Returns:
            any: The sensor data for the specified location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location.

        Args:
            location (int): The location for which to set data.
            data (any): The new data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device thread, waiting for its completion.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Manages the operational lifecycle of a Device, including fetching neighbor
    information, distributing and executing assigned scripts using worker threads,
    and synchronizing with other DeviceThreads at each timepoint.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def distribute_scripts(self, scripts):
        """
        Distributes the list of scripts among the available worker cores.

        Args:
            scripts (list): A list of (script, location) tuples to be executed.

        Returns:
            list: A list of lists, where each inner list contains scripts
                  assigned to a specific worker.
        """
        worker_scripts = []
        for _ in range(self.device.cores):
            worker_scripts.append([])
        i = 0
        # Block Logic: Distributes scripts in a round-robin fashion across available worker cores
        # to ensure load balancing.
        for script in scripts:
            worker_scripts[i % self.device.cores].append(script)
            i = i + 1
        return worker_scripts

    def run(self):
        """
        The main execution loop for the DeviceThread. It continuously
        processes sensor data, executes scripts, and synchronizes with other devices
        until a shutdown signal is received.
        """
        while True:
            # Block Logic: Fetches the current set of neighboring devices from the supervisor.
            # A `None` value indicates a shutdown signal.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Block Logic: Halts execution until the current timepoint's scripts
            # have been assigned and marked as ready by the device.
            self.device.timepoint_done.wait()

            # Functional Utility: Manages the execution of scripts by creating and
            # joining worker threads, effectively forming a local worker pool.
            inner_workers = []
            
            # Block Logic: Divides the assigned scripts among the configured number of cores
            # for parallel processing.
            worker_scripts = self.distribute_scripts(self.device.scripts)
            for worker_scr in worker_scripts:
                # Functional Utility: Each `Worker` thread is responsible for executing a subset
                # of scripts, leveraging multi-core processing for efficiency.
                inner_thread = Worker(worker_scr,\
                                      neighbours,\
                                      self.device)
                inner_workers.append(inner_thread)
                inner_thread.start()

            # Block Logic: Waits for all worker threads to complete their assigned scripts
            # before proceeding to the next synchronization point.
            for thr in inner_workers:
                thr.join()
            # Block Logic: Resets the timepoint completion event, preparing for the
            # next cycle of script assignment and execution.
            self.device.timepoint_done.clear()
            # Functional Utility: Synchronizes with all other DeviceThreads in the simulation
            # using a barrier, ensuring all devices complete their current timepoint
            # processing before proceeding.
            self.device.barrier.wait()



class Worker(Thread):
    """
    A worker thread responsible for executing a subset of scripts assigned
    to a device. It operates on sensor data, potentially involving data
    from neighboring devices, and uses shared locks for data consistency.
    """
    
    def __init__(self, script_loc, neighbours, device):
        """
        Initializes a new Worker instance.

        Args:
            script_loc (list): A list of (script, location) tuples for this worker to execute.
            neighbours (list): A list of neighboring Device instances.
            device (Device): The Device instance this worker belongs to.
        """
        Thread.__init__(self)
        self.script_loc = script_loc
        self.neighbours = neighbours
        self.script_data = [] # Stores collected sensor data for script execution
        self.device = device

    def run(self):
        """
        The main execution method for the Worker thread. It iterates through
        its assigned scripts, acquires locks, collects data, executes the script,
        and updates sensor data.
        """
        # Block Logic: Iterates through each script-location pair assigned to this worker.
        for (script, location) in self.script_loc:
            # Functional Utility: Acquires a lock for the specific location to ensure
            # exclusive access to the sensor data during script execution, preventing
            # race conditions in a multi-threaded environment.
            self.device.shared_locks[location].acquire()
            self.script_data = []
            
            # Block Logic: Gathers sensor data from neighboring devices at the specified location.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    self.script_data.append(data)

            # Block Logic: Retrieves the local device's sensor data for the current location.
            data = self.device.get_data(location)

            if data is not None:
                self.script_data.append(data)
            
            # Pre-condition: `script_data` must not be empty for script execution.
            # Post-condition: If `script_data` is not empty, the script is executed
            # and its result is propagated to neighboring and local device data.
            if self.script_data != []:
                # Functional Utility: Executes the assigned script with the collected data,
                # simulating sensor data processing.
                result = script.run(self.script_data)
                
                # Block Logic: Updates the sensor data of neighboring devices with the script's result.
                for dev in self.neighbours:
                    dev.set_data(location, result)
                
                # Block Logic: Updates the local device's sensor data with the script's result.
                self.device.set_data(location, result)
             
            # Functional Utility: Releases the lock for the current location, allowing other
            # threads to access the sensor data.
            self.device.shared_locks[location].release()
