

"""
This module provides components for simulating a device within a distributed system.
It features a multi-threaded architecture where each `Device` instance operates
within its `DeviceThread`. The `DeviceThread` dispatches tasks (scripts)
to `MyWorker` threads via a `Queue` for concurrent execution. Synchronization
among devices is managed through an inline `ReusableBarrier` class (using `threading.Condition`),
and location-specific data access is protected by a shared list of `threading.Lock` objects.

Key Components:
- `ReusableBarrier`: A synchronization primitive allowing multiple threads to wait for each other.
- `Device`: Represents an individual simulated device, managing sensor data and scripts.
- `DeviceThread`: The main thread for a device, orchestrating script execution via worker threads.
- `MyWorker`: Executes individual scripts, handling data acquisition and propagation with locking.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue

class ReusableBarrier(object):
    """
    A reusable barrier synchronization primitive that allows a fixed number of threads
    to wait for each other to reach a common execution point (the barrier).
    Once all threads have arrived, they are all released to proceed.
    It uses a `threading.Condition` object to manage thread blocking and notification,
    and resets itself after each synchronization point for reusability.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier with a specified number of participating threads.

        Args:
            num_threads (int): The total number of threads that must arrive at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Current count of threads yet to reach the barrier.
        self.cond = Condition()              # Condition variable for blocking and waking threads.

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have also called `wait()`.
        Once all threads have arrived, they are all released to proceed.
        The barrier then resets for subsequent use.
        """
        self.cond.acquire()                  # Acquire the lock associated with the condition variable.
        self.count_threads -= 1              # Decrement the count of threads yet to arrive.
        # Block Logic: Check if this is the last thread to arrive at the barrier.
        if self.count_threads == 0:
            self.cond.notify_all()           # If all threads have arrived, notify all waiting threads.
            self.count_threads = self.num_threads # Reset the counter for the next cycle of the barrier.
        else:
            self.cond.wait()                 # If not the last thread, wait (release lock and block) until notified.
        self.cond.release()                  # Release the lock associated with the condition variable.

class Device(object):
    """
    Represents an individual simulated device in a distributed system.
    Each `Device` instance manages its own sensor data, communicates with a
    central supervisor, and processes assigned scripts. It utilizes a dedicated
    `DeviceThread` to manage its operations and coordinates with other devices
    through shared synchronization primitives (`ReusableBarrier` and `Lock` objects).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping location IDs to their
                                current sensor data values.
            supervisor (object): A reference to the central supervisor managing
                                 the distributed system.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when a script has been received.
        self.scripts = []             # List to store (script, location) tuples.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's tasks.
        self.queue = Queue()           # A queue for tasks to be processed by worker threads.
        self.setup = Event()           # Event to signal that the device's setup is complete.
        self.threads = []              # List to hold references to worker threads.
        self.locations_lock = []       # List of Locks for location-specific data access.
        self.barrier = None            # Shared barrier for synchronizing device threads.
        self.thread = DeviceThread(self) # Dedicated thread for this device's operations.
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization primitives (a `ReusableBarrier` and a
        list of `Lock` objects for locations) across a group of devices. The
        device with `device_id == 0` is responsible for creating these resources
        and distributing them to all other devices in the group.

        Args:
            devices (list): A list of Device objects that are part of the same group.
        """
        # Block Logic: Device with ID 0 is responsible for initializing shared resources.
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices)) # Create a new barrier for the group.
            # Functional Utility: Initialize a fixed number of location locks (25 in this case).
            # This implicitly assumes locations can be mapped to indices 0-24.
            for _ in range(25):
                lock = Lock()
                self.locations_lock.append(lock) # Add a new Lock for each location.

            # Block Logic: Distribute the newly created barrier and location locks to all devices.
            for device in devices:
                device.barrier = barrier
                device.locations_lock = self.locations_lock
                device.setup.set() # Signal that this device's setup is complete.

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location on this device.
        If `script` is `None`, it signals that all scripts for the current timepoint are assigned.

        Args:
            script (object or None): The script object to execute, or `None`.
            location (int): The integer identifier for the location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal that a script has been received.
        else:
            self.timepoint_done.set() # Signal that all scripts for this timepoint have been assigned.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's internal state.

        Args:
            location (int): The integer identifier of the location for which to retrieve data.

        Returns:
            Any: The sensor data associated with the location, or `None` if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a given location on this device.

        Args:
            location (int): The integer identifier of the location to update.
            data (Any): The new sensor data value for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device's operational thread.
        This method waits for the device's thread to complete its execution.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    The dedicated operational thread for a `Device` instance. It orchestrates
    the device's simulation lifecycle, including waiting for initial setup,
    creating a pool of `MyWorker` threads, fetching neighbor information from
    the supervisor, dispatching scripts to workers via a queue, and synchronizing
    with other device threads using a shared `ReusableBarrier`.
    """
    

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device's thread.
        It manages worker threads, coordinates with the supervisor,
        queues scripts, and participates in barrier synchronization.
        """
        # Block Logic: Wait for the Device's shared resources (barrier, locations_lock) to be set up.
        self.device.setup.wait()

        # Block Logic: Create and start a fixed number of worker threads.
        for _ in range(8): # Creates 8 worker threads.
            thread = MyWorker(self.device)
            thread.start()
            self.device.threads.append(thread)

        # Block Logic: Main loop for continuous simulation timepoints.
        while True:
            # Pre-condition: Fetch information about neighboring devices from the supervisor.
            # This also serves as a signal for the simulation's continuation or termination.
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                # If neighbours is None, it's a shutdown signal from the supervisor.
                # Block Logic: Put 'None' tasks into the queue for each worker to signal them to shut down.
                for thread in self.device.threads:
                    for _ in range(8): # Send 8 None signals per thread (assuming 8 tasks max per thread).
                        self.device.queue.put(None)
                    thread.join() # Wait for each worker thread to finish processing shutdown signals.
                break # Exit the main simulation loop.
            
            # Block Logic: Wait until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()
            # Functional Utility: Synchronize with other devices using the barrier before script processing.
            self.device.barrier.wait()

            # Block Logic: Queue up all assigned scripts for the worker threads to process.
            # Each script becomes a task (neighbours, location, script_object) in the queue.
            for (script, location) in self.device.scripts:
                self.device.queue.put((neighbours, location, script))

            # Functional Utility: Clear the timepoint_done event, resetting it for the next timepoint.
            self.device.timepoint_done.clear()
            
            # Functional Utility: Synchronize with other devices using the barrier after script processing.
            # This ensures all devices have completed their script processing for the current timepoint.
            self.device.barrier.wait()


class MyWorker(Thread):
    """
    A worker thread within a `DeviceThread` that processes individual tasks
    (scripts) from a shared queue. It's responsible for acquiring location-specific
    locks, gathering data from neighboring devices and its own device, executing
    the assigned script, and propagating the results. It also handles shutdown signals.
    """
    
    def __init__(self, device):
        """
        Initializes a MyWorker thread.

        Args:
            device (Device): The Device instance that this worker thread serves,
                             providing access to its queue, locks, and data.
        """
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        The main execution loop for the worker thread.
        It continuously retrieves tasks from the queue, processes them,
        and signals task completion, handling shutdown when instructed.
        """
        while True:
            # Block Logic: Retrieve a task from the shared queue.
            # A task is typically a tuple: (neighbours, location, script_object).
            elem = self.device.queue.get()
            
            # Pre-condition: Check if the retrieved element is a shutdown signal (None).
            if elem is None:
                break # Exit the worker thread's loop.
            
            # Unpack task elements for clarity.
            neighbours, location, script = elem[0], elem[1], elem[2]

            # Block Logic: Acquire the lock for the specific location.
            # This ensures exclusive access to the data associated with 'location'
            # across all worker threads that might operate on the same location.
            self.device.locations_lock[location].acquire()
            
            script_data = [] # List to accumulate data for the script.
            data = None      # Temporary variable for data retrieval.

            # Block Logic: Gather data from all neighboring devices for the script's location.
            # Note: No individual device locks are explicitly acquired here for neighbors,
            # implying that `get_data` and `set_data` methods handle their own internal consistency
            # or rely on the `locations_lock` for broader coordination.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Gather data from the current device itself for the script's location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Pre-condition: Check if any data was collected before running the script.
            if script_data != []:
                # Functional Utility: Execute the script with the collected data.
                result = script.run(script_data)

                # Block Logic: Propagate the result of the script execution back to all neighboring devices.
                for device in neighbours:
                    device.set_data(location, result)
                
                # Block Logic: Update the current device's own sensor data with the result.
                self.device.set_data(location, result)
            
            # Block Logic: Release the lock for the specific location.
            self.device.locations_lock[location].release()

            # Functional Utility: Signal to the queue that the current task is complete.
            self.device.queue.task_done()
