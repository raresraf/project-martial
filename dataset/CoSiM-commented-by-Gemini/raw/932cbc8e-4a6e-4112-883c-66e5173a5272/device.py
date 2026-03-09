"""
This module simulates a distributed network of devices that process sensor data
concurrently. It models a system where devices execute scripts on data from
themselves and their neighbors, synchronized in discrete time steps.
"""
from threading import Event, Thread, Lock
from Queue import Queue
from barrier import ReusableBarrierSem

class Device(object):
    """
    Represents a single device in the distributed network simulation.

    Each device has its own sensor data, a set of neighboring devices, and a
    pool of worker threads to execute scripts. It communicates with a central
    supervisor and synchronizes with other devices at each time step.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor data,
                                keyed by location.
            supervisor: The central supervisor object that manages the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal script reception (currently unused).
        self.scripts = []  # List to store (script, location) tuples.
        self.devices = []  # List of neighboring devices.
        self.threads = []  # Pool of worker threads.
        self.barrier = None  # Synchronization barrier for all connected devices.
        self.timepoint_done = Event()  # Event to signal the start of a new timepoint.
        self.thread_queue = Queue()  # Task queue for worker threads.
        self.locks = {}  # Dictionary of locks for data locations.
        self.thread = DeviceThread(self)  # The main control thread for this device.
        self.thread.start()


    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device's worker threads and its connections to neighbors.

        Initializes a pool of worker threads and establishes the synchronization
        barrier shared with neighboring devices.

        Args:
            devices (list): A list of neighboring Device objects.
        """
        # Create and start a pool of 8 worker threads.
        for _ in range(8):
            thread = Worker(self)
            thread.start()
            self.threads.append(thread)
        
        # Register neighboring devices.
        for device in devices:
            if device is not None:
                self.devices.append(device)
        
        # If this device is the first in a group, it creates the shared barrier.
        if self.barrier is None:
            self.barrier = ReusableBarrierSem(len(self.devices))
        
        # Propagate the shared barrier to all neighbors.
        for device in self.devices:
            if device is not None:
                if device.barrier is None:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed or signals the end of a timepoint's setup.

        Args:
            script: The script object to be executed.
            location: The data location the script will operate on. If `script` is None,
                      this signals that all scripts for the current timepoint have been
                      assigned, and the device can proceed.
        """
        if script is not None:
            # Add the script and its target location to the list.
            self.scripts.append((script, location))
            
            # Create a lock for the location if it doesn't exist.
            if location is not None:
                if not self.locks.has_key(location):
                    self.locks[location] = Lock()
            self.script_received.set()
        else:
            # A None script is the signal to start processing the timepoint.
            self.timepoint_done.set()

        # Ensure that neighbors share the same locks for the same locations.
        for device in self.devices:
            if not device.locks.has_key(location):
                if self.locks.has_key(location):
                    device.locks[location] = self.locks[location]

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by stopping its main control thread."""
        self.thread.join()


class Worker(Thread):
    """
    A worker thread that executes scripts on sensor data.

    Workers fetch tasks from a queue, acquire locks for data locations,
    gather data from the parent device and its neighbors, run the script,
    and update the data with the result.
    """
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device

    def run(self):
        """The main loop for the worker thread."""
        while True:
            # Block until a task is available in the queue.
            script_loc_neigh = self.device.thread_queue.get()
            
            # Use a sentinel value (None, None, None) to terminate the thread.
            if script_loc_neigh[0] is None:
                if script_loc_neigh[1] is None:
                    if script_loc_neigh[2] is None:
                        self.device.thread_queue.task_done()
                        break
            script_data = []
            
            # Acquire a lock to ensure exclusive access to the data at this location
            # across all devices that share this location.
            self.device.locks[script_loc_neigh[1]].acquire()
            
            # --- Critical Section ---
            # Pre-condition: Gather data from all neighboring devices for the script's location.
            for device in script_loc_neigh[2]:
                data = device.get_data(script_loc_neigh[1])
                if data is not None:
                    script_data.append(data)

            # Also gather data from the parent device itself.
            data = self.device.get_data(script_loc_neigh[1])
            if data is not None:
                script_data.append(data)

            # Invariant: Run the script only if there is data to process.
            if script_data != []:
                # Execute the script with the collected data.
                result = script_loc_neigh[0].run(script_data)
                
                # Update the data with the script's result on all neighbors and self.
                for device in script_loc_neigh[2]:
                    device.set_data(script_loc_neigh[1], result)
                self.device.set_data(script_loc_neigh[1], result)
            # --- End Critical Section ---
            
            self.device.locks[script_loc_neigh[1]].release()
            self.device.thread_queue.task_done()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    This thread orchestrates the device's participation in each synchronized
    timepoint of the simulation.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop, executed once per timepoint."""
        while True:
            
            # Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # A None value for neighbors signals the end of the simulation.
            if neighbours is None:
                break
            
            # Pre-condition: Wait for the supervisor to signal that all scripts for
            # the current timepoint have been assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Reset for the next timepoint.
            
            # Invariant: Dispatch all assigned scripts to the worker thread pool.
            for (script, location) in self.device.scripts:
                self.device.thread_queue.put((script, location, neighbours))
            # Wait for all workers to finish their tasks for this timepoint.
            self.device.thread_queue.join()
            
            # Synchronize with all other devices at the barrier before proceeding
            # to the next timepoint.
            self.device.barrier.wait()
        
        # --- Shutdown Sequence ---
        # Send sentinel values to terminate all worker threads.
        for _ in range(len(self.device.threads)):
            self.device.thread_queue.put((None, None, None))
        # Wait for all worker threads to exit gracefully.
        for thread in self.device.threads:
            thread.join()