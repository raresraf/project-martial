"""
This module implements a multi-threaded simulation for a network of devices.

This version of the simulation framework defines a `Device` that executes scripts
based on sensor data and communicates with neighbors. It creates new worker threads
for each script in every time step, rather than using a persistent thread pool.

NOTE: This implementation has several notable design characteristics, including
inefficient dynamic thread creation and a potentially flawed locking mechanism for
data access (using separate get/set locks).

Classes:
    Device: Represents a single node in the device network.
    DeviceThread: The main control thread for a Device, which spawns WorkerThreads.
    WorkerThread: A short-lived thread created to execute a single script.
"""


from threading import Event, Thread, Lock
# The barrier is assumed to be a standard reusable barrier implementation.
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a device in the simulation network.

    Manages sensor data, script execution, and synchronization with other devices.
    Device 0 is responsible for setting up shared resources for the network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that all scripts for a time step have been received.
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event() # NOTE: This event appears to be unused.
        self.thread = DeviceThread(self)

        # --- Shared resources initialized by device 0 ---
        self.devices = []
        self.locations = [] # A list of Lock objects for locations.
        self.barrier = None # The main barrier for end-of-step synchronization.

        # --- Synchronization Primitives ---
        self.setup_start = Event() # Signals that the initial setup is complete.
        # NOTE: Using separate locks for get and set is not thread-safe.
        # A single lock should be used to protect sensor_data from race conditions.
        self.set_lock = Lock()
        self.get_lock = Lock()
        
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for all devices in the simulation.
        
        This method should only be run by one device (e.g., device_id 0).
        It creates and distributes a shared barrier and a fixed set of location locks.
        """
        self.devices = devices
        num_devices = len(devices)

        # Create a reusable barrier for all devices.
        barrier = ReusableBarrierSem(num_devices)

        if self.device_id == 0:
            # Create a fixed list of 25 locks for locations. This assumes a maximum of 25 locations.
            for _ in range(25):
                lock = Lock()
                self.locations.append(lock)

            # Distribute the shared resources to all devices in the network.
            for device in devices:
                device.locations = self.locations
                device.barrier = barrier
                # Signal each device that setup is complete and it can start its main loop.
                device.setup_start.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device for the current time step.
        A `None` script signals the end of script assignment for this step.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # All scripts for the current step have been assigned; signal the DeviceThread.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.
        
        NOTE: This method is not thread-safe in conjunction with `set_data` because
        they use different locks.
        """
        with self.get_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location.
        
        NOTE: This method is not thread-safe in conjunction with `get_data` because
        they use different locks.
        """
        with self.set_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a single Device.
    
    This thread waits for scripts, creates a new worker for each script,
    manages their execution in batches, and synchronizes with other devices.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main execution loop for the device."""
        # Wait until the initial setup of shared resources is complete.
        self.device.setup_start.wait()

        while True:
            # Get the list of neighbors for the current time step.
            neighbours = self.device.supervisor.get_neighbours()

            # A `None` neighbor list is the signal to shut down.
            if neighbours is None:
                break

            # Wait until all scripts for this time step have been assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            index = 0
            workers = []
            num_scripts = len(self.device.scripts)

            # --- Inefficient Thread Creation ---
            # A new thread is created for every script in every time step.
            # A more efficient design would use a persistent thread pool and a work queue.
            for _ in self.device.scripts:
                worker = WorkerThread(self.device, neighbours, index)
                workers.append(worker)
                index += 1

            # --- Manual Thread Batching ---
            # This section attempts to simulate a thread pool of size 8 by manually
            # starting and joining threads in batches.
            if num_scripts < 8:
                # If fewer than 8 scripts, run them all in parallel.
                for worker in workers:
                    worker.start()

                for worker in workers:
                    worker.join()
            else:
                # If more than 8 scripts, run them in batches of 8.
                aux = 0
                while True:
                    if num_scripts == 0:
                        break
                    
                    if num_scripts >= 8:
                        start = aux
                        end = aux + 8
                        # Start a batch of 8 workers.
                        for i in range(start, end):
                            workers[i].start()
                        # Wait for the batch to complete.
                        for i in range(start, end):
                            workers[i].join()
                        aux += 8
                        num_scripts -= 8
                    elif num_scripts < 8:
                        # Process the final, smaller batch.
                        start = aux
                        end = aux + num_scripts
                        for i in range(start, end):
                            workers[i].start()
                        for i in range(start, end):
                            workers[i].join()
                        break

            # Wait at the global barrier for all other devices to finish this time step.
            self.device.barrier.wait()


class WorkerThread(Thread):
    """
    A short-lived worker thread designed to execute a single script.
    """

    def __init__(self, device, neighbours, index):
        """Initializes the worker with the context needed to run one script."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.index = index # The index of the script in the device's script list.

    def run(self):
        """The main logic for executing one script."""
        # Retrieve the script and its target location.
        (script, location) = self.device.scripts[self.index]

        # Acquire the lock for the target location to ensure mutual exclusion.
        self.device.locations[location].acquire()

        # --- Data Gathering ---
        script_data = []

        # Collect data from all neighboring devices.
        for neighbour in self.neighbours:
            data_neigh = neighbour.get_data(location)
            if data_neigh is not None:
                script_data.append(data_neigh)

        # Collect data from the parent device.
        own_data = self.device.get_data(location)
        if own_data is not None:
            script_data.append(own_data)

        # --- Script Execution and Data Propagation ---
        if script_data:
            # Run the script on the collected data.
            result = script.run(script_data)

            # Propagate the result back to all neighbors and the parent device.
            for neighbour in self.neighbours:
                neighbour.set_data(location, result)
            self.device.set_data(location, result)

        # Release the lock for the location.
        self.device.locations[location].release()
