"""
Models a device in a distributed sensor network simulation.

This script defines the behavior of a single device, its main control loop,
and its worker threads for processing tasks. The system appears to be designed
for a time-stepped simulation where devices operate on sensor data, communicate
with neighbors, and synchronize at each time step.
"""


from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a single device node in the simulated network.

    Each device has its own sensor data, a set of scripts to execute, a queue
    for work, and a list of neighboring devices. It uses multiple synchronization
    primitives to coordinate its main thread and a pool of worker threads with
    other devices in the network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor readings,
                                keyed by location.
            supervisor (Supervisor): An object that manages the overall simulation and
                                     provides neighborhood information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.scripts = []  # Persistent scripts to be executed each timepoint.
        self.work_queue = Queue()  # Queue of (script, location) tuples for workers.
        self.neighbours = []  # List of neighboring Device objects.

        # --- Synchronization Primitives ---
        self.timepoint_done = Event()  # Signals that all scripts for a timepoint are assigned.
        self.setup_ready = Event()     # Signals that the master device (ID 0) has set up shared resources.
        self.neighbours_set = Event()  # Signals that the neighbours for the current timepoint have been set.
        self.scripts_mutex = Semaphore(1)  # Protects access to the self.scripts list.
        self.location_locks_mutex = None  # Mutex for the shared location_locks dictionary itself.
        self.location_locks = {}      # Shared dict of locks, one for each sensor location.
        self.timepoint_barrier = None  # Shared barrier to sync all devices at the end of a timepoint.

        self.thread = DeviceThread(self)
        
        # A pool of worker threads to process scripts concurrently.
        self.workers = [DeviceWorker(self) for _ in range(8)]

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources and starts the device's threads.

        Device 0 acts as the coordinator, creating the shared timepoint barrier and
        location lock dictionary. Other devices wait for Device 0 and then adopt
        these shared objects. This method also populates the shared lock dictionary
        based on all unique sensor locations across all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Device 0 is the master, responsible for creating shared synchronization objects.
        if self.device_id == 0:
            self.timepoint_barrier = ReusableBarrierCond(len(devices))
            self.location_locks_mutex = Lock()
            self.setup_ready.set()  # Signal other devices that shared objects are ready.
        else:
            # Other devices find device 0 and wait for it to be ready.
            device = next(device for device in devices if device.device_id == 0)
            device.setup_ready.wait() 
            # Adopt the shared objects created by device 0.
            self.timepoint_barrier = device.timepoint_barrier
            self.location_locks = device.location_locks
            self.location_locks_mutex = device.location_locks_mutex

        # Atomically populate the shared location_locks dictionary with a lock for each unique location.
        with self.location_locks_mutex:
            for location in self.sensor_data.keys():
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()

        # Start the main device thread and all worker threads.
        self.thread.start()
        for worker in self.workers:
            worker.start()


    def assign_script(self, script, location):
        """
        Assigns a script to the device for a given location.

        This is typically called by the supervisor. It waits until the device's
        neighbours are known for the current timepoint, then adds the work to the
        queue. A `None` script is a sentinel value indicating the end of script
        assignments for the current timepoint.

        Args:
            script (Script): The script object to be executed.
            location (str): The sensor location the script will operate on.
        """
        self.neighbours_set.wait()

        if script is not None:
            with self.scripts_mutex:
                self.scripts.append((script, location))
            self.work_queue.put((script, location))
        else:
            # A None script signals the end of assignments for this timepoint.
            self.neighbours_set.clear() 
            self.timepoint_done.set() 

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location to query.

        Returns:
            The data for the location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specific location.

        Args:
            location (str): The location to update.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Gracefully shuts down the device and its associated threads.
        """
        self.thread.join()
        # Send a sentinel value to each worker to signal it to exit.
        for worker in self.workers:
            self.work_queue.put((None, None))

        for worker in self.workers:
            worker.join()



class DeviceThread(Thread):
    """
    The main control thread for a Device, managing its lifecycle through timepoints.
    """

    def __init__(self, device):
        """
        Initializes the main thread for a device.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main time-stepped loop of the device.

        In each iteration (timepoint):
        1. Gets the current list of neighbours from the supervisor.
        2. Re-queues any persistent scripts for execution.
        3. Signals that it's ready for script assignments (`neighbours_set`).
        4. Waits for the supervisor to finish assigning scripts (`timepoint_done`).
        5. Waits for all work in its queue to be completed by its workers.
        6. Waits at a global barrier, synchronizing with all other devices
           before starting the next timepoint.
        """
        while True:
            # Get the list of neighbours for the current timepoint from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # A None from get_neighbours signals the end of the simulation.
            if self.device.neighbours is None:
                break

            # Add all persistent scripts to the work queue for the new timepoint.
            for (script, location) in self.device.scripts:
                self.device.work_queue.put((script, location))

            # Signal that neighbours are set and the device is ready for new script assignments.
            self.device.neighbours_set.set()

            # Wait for the supervisor to signal that all scripts for this timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Wait until all items in the work queue have been processed by the workers.
            self.device.work_queue.join()

            # Clear the event for the next timepoint.
            self.device.timepoint_done.clear()

            # Wait at the barrier for all other devices to finish the current timepoint.
            self.device.timepoint_barrier.wait()



class DeviceWorker(Thread):
    """
    A worker thread that executes scripts for a device.
    """

    def __init__(self, device):
        """
        Initializes a worker thread.

        Args:
            device (Device): The device this worker belongs to.
        """
        Thread.__init__(self, name="Device %d Worker" % device.device_id)
        self.device = device


    def run(self):
        """
        The main loop for the worker thread.

        Continuously fetches tasks from the device's work queue and executes them.
        A task involves:
        1. Acquiring a lock for the target location to ensure data consistency.
        2. Gathering data for that location from the device itself and its neighbours.
        3. Running the script with the gathered data.
        4. Broadcasting the result by updating the data on the device itself and its neighbours.
        """
        while True:
            # Block until a task is available in the queue.
            (script, location) = self.device.work_queue.get(block=True)

            # A (None, None) sentinel signals the worker to terminate.
            if script is None and location is None:
                self.device.work_queue.task_done()
                break

            # Use a shared, location-specific lock to prevent race conditions
            # when multiple devices' workers access the same location.
            with self.device.location_locks[location]:
                script_data = []

                # Gather data from all neighbours.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Gather data from the device itself.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Execute the script with the collected data.
                    result = script.run(script_data)

                    # Broadcast the result to all neighbours.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    # Update the device's own data.
                    self.device.set_data(location, result)

            self.device.work_queue.task_done()