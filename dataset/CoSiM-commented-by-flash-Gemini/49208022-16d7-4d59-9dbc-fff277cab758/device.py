"""
@49208022-16d7-4d59-9dbc-fff277cab758/device.py
@brief This module defines the `Device` class, representing a node in a distributed
system, and `DeviceThread` for multi-threaded script execution. It utilizes
a `ReusableBarrierCond` for synchronization and a `Queue` for task distribution.
"""

from threading import Event, Thread, Lock, Condition
from barrier import ReusableBarrierCond
from Queue import Queue

THREAD_NR = 8 # Defines the number of worker threads to be used per device.

class Device(object):
    """
    Represents a single device in a simulated distributed system.

    Each device manages its own sensor data, processes scripts using a pool
    of worker threads, and interacts with a supervisor for global coordination.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary representing the sensor data this device holds.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to hold scripts assigned to this device.
        self.setup_finished = Event() # Event to signal that initial setup (barrier, locks) is complete.
        self.dataLock = Lock() # Lock for protecting access to the device's sensor_data.
        self.shared_lock = Lock() # A generic shared lock (usage may vary based on specific needs).
        self.thread_queue = Queue(0) # Queue to hold available worker threads.
        # Barrier for synchronizing worker threads within this device when fetching neighbors.
        self.wait_get_neighbours = ReusableBarrierCond(THREAD_NR)
        self.thread_pool = [] # List to hold instances of DeviceThread (worker threads).
        self.neighbours = [] # List to store neighboring devices fetched from the supervisor.

        # Block Logic: Create and start the pool of worker threads.
        for i in range(0, THREAD_NR):
            thread = DeviceThread(self, i)
            self.thread_pool.append(thread)
            self.thread_queue.put(thread) # Add thread to the available queue.
            thread.start() # Start the worker thread.

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization primitives (barrier and location-specific locks)
        across all devices in the system. This method is called by the supervisor.

        Args:
            devices (list): A list of all Device instances in the simulated system.
        """
        # Block Logic: Only the device with device_id 0 (the 'master' device) initializes global resources.
        if self.device_id == 0:
            # Create a reusable barrier for all threads across all devices.
            self.reusable_barrier = ReusableBarrierCond(len(devices) * THREAD_NR)
            self.location_locks = {} # Dictionary to hold locks for specific data locations.
            # Block Logic: Distribute the shared barrier and location locks to all other devices.
            for device in devices:
                if device.device_id != self.device_id:
                    device.set_location_locks(self.location_locks)
                    device.set_barrier(self.reusable_barrier)

            self.setup_finished.set() # Signal that global setup is complete.

    def set_barrier(self, reusable_barrier):
        """
        Sets the shared reusable barrier for this device.

        Args:
            reusable_barrier (ReusableBarrierCond): The global synchronization barrier.
        """
        self.reusable_barrier = reusable_barrier
        self.setup_finished.set() # Signal that setup is finished after receiving the barrier.

    def set_location_locks(self, location_locks):
        """
        Sets the shared dictionary of location-specific locks for this device.

        Args:
            location_locks (dict): A dictionary mapping data locations to their respective locks.
        """
        self.location_locks = location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to a worker thread for execution.

        If a script is provided, it's assigned directly. If no script, it signals
        all worker threads to process any remaining scripts and then prepare for shutdown.

        Args:
            script (object): The script object to be executed.
            location (int): The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script to the device's list.
            if location not in self.location_locks:
                # Initialize a lock for a new location if it doesn't exist.
                self.location_locks[location] = Lock()

            # Functional Utility: Get an available worker thread from the queue.
            thread = self.thread_queue.get()
            thread.give_script(script, location) # Assign the script to the worker thread.

            
        else:
            # Block Logic: If no script is provided, distribute all pending scripts to workers.
            # This path is typically used for a final flush of scripts or shutdown signaling.
            for (s, l) in self.scripts:
                thread = self.thread_queue.get()
                thread.give_script(s, l)

            # Functional Utility: Signal all worker threads to process any remaining tasks and then terminate.
            for thread in self.thread_pool:
                thread.give_script(None, None) # Send a shutdown signal to each worker.


    def get_data(self, location):
        """
        Retrieves data from the device's sensor_data at the specified location.

        Args:
            location (int): The index or key of the data to retrieve.

        Returns:
            any: The data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates data in the device's sensor_data at the specified location.

        Args:
            location (int): The index or key of the data to set.
            data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data # Update the data if the location exists.

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device, waiting for all worker threads to finish.
        """
        for i in range(THREAD_NR):
            self.thread_pool[i].join() # Wait for each worker thread to complete its execution.


class DeviceThread(Thread):
    """
    Worker thread responsible for processing assigned scripts for a Device.

    Each `Device` maintains a pool of these threads to handle script execution
    concurrently, managing data retrieval from neighbors and local updates.
    """
    

    def __init__(self, device, ID):
        """
        Initializes a DeviceThread instance.

        Args:
            device (Device): The parent Device instance this thread belongs to.
            ID (int): A unique identifier for this worker thread within its device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id = ID # Unique ID for this specific worker thread.
        self.script_queue = Queue(0) # A queue to receive scripts to execute.

    def give_script(self, script, location):
        """
        Adds a script to this worker thread's queue for execution.

        Args:
            script (object): The script object to be executed.
            location (int): The data location relevant to the script.
        """
        self.script_queue.put((script, location))

    def run(self):
        """
        The main execution loop for the worker thread.

        It waits for the device's setup to complete, fetches neighbors (if it's the
        master worker thread), processes scripts from its queue, and participates
        in barrier synchronization.
        """
        while True:
            # Pre-condition: Wait until the device's global setup (e.g., barrier, locks) is finished.
            self.device.setup_finished.wait()

            # Block Logic: Only the first worker thread (ID 0) is responsible for fetching neighbors.
            if self.id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: All worker threads in the device synchronize here after neighbor fetching.
            self.device.wait_get_neighbours.wait()

            # If neighbors are None (e.g., supervisor signals shutdown), terminate the thread.
            if self.device.neighbours is None:
                break

            # Block Logic: Process scripts from this thread's script queue.
            while True:
                (script, location) = self.script_queue.get() # Get a script; blocks if queue is empty.

                if script is None: # A None script signals this thread to finish processing and exit the inner loop.
                    break

                # Pre-condition: Acquire a lock for the specific data location to ensure exclusive access during script execution.
                self.device.location_locks[location].acquire()

                script_data = [] # List to accumulate data for the current script.
                
                # Block Logic: Collect data from neighboring devices.
                for device in self.device.neighbours:
                    # Acquire lock for neighbor's data to ensure consistency.
                    device.dataLock.acquire()
                    data = device.get_data(location) # Retrieve data from the neighbor.
                    device.dataLock.release() # Release neighbor's data lock.

                    if data is not None:
                        script_data.append(data) # Add valid data.

                # Block Logic: Collect data from the current device.
                self.device.dataLock.acquire() # Acquire lock for local device's data.
                data = self.device.get_data(location) # Retrieve local data.
                self.device.dataLock.release() # Release local device's data lock.
                
                if data is not None:
                   script_data.append(data) # Add valid local data.

                self.device.location_locks[location].release() # Release the location lock after data collection.

                # Pre-condition: If there is data to process, execute the script.
                if script_data != []:
                    # Action: Execute the script.
                    result = script.run(script_data)
                    
                    self.device.location_locks[location].acquire() # Re-acquire lock for updating data.

                    # Block Logic: Update data in neighboring devices.
                    for device in self.device.neighbours:
                        device.dataLock.acquire() # Acquire lock for neighbor's data.
                        device.set_data(location, result) # Update neighbor's data.
                        device.dataLock.release() # Release neighbor's data lock.

                    # Block Logic: Update data in the current device.
                    self.device.dataLock.acquire() # Acquire lock for local device's data.
                    self.device.set_data(location, result) # Update local data.
                    self.device.dataLock.release() # Release local device's data lock.
                    self.device.location_locks[location].release() # Release the location lock after data update.

               # Functional Utility: Return this worker thread to the device's available thread queue.
                self.device.thread_queue.put(self)

            # Block Logic: Synchronize all worker threads across all devices at the global barrier.
            self.device.reusable_barrier.wait()