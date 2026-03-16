"""
This module defines a simulation framework for a network of interconnected devices.

It uses a multi-threaded approach to simulate the behavior of devices that
process sensor data, run computational scripts, and share results with their
neighbors in discrete, synchronized time steps.

The main components are:
- Device: Represents a node in the network, managing its own data, scripts, and a pool of worker threads.
- WorkerThread: Executes computational scripts on data gathered from a device and its neighbors.
- DeviceThread: The main control loop for a device, handling the progression of time steps and synchronization.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
from Queue import Queue

class Device(object):
    """Represents a single device in the simulated network.

    Each device has an ID, local sensor data, and a set of scripts to execute.
    It communicates with a supervisor to get its neighbors and synchronizes with
    other devices using a shared barrier. Device with ID 0 acts as a master
    node, initializing shared synchronization objects.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's initial
                                local data, keyed by location.
            supervisor: An object that manages the overall simulation, providing
                        neighbor information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.queue = Queue()
        self.worker_threads = []
        self.worker_threads_no = 8
        self.timepoint_done = Event()

        # Spawn a pool of worker threads to process tasks concurrently.
        for _ in range(0, self.worker_threads_no):
            worker = WorkerThread(self, self.queue)
            worker.start()
            self.worker_threads.append(worker)

        # Device 0 is the master, responsible for creating shared synchronization primitives.
        if device_id == 0:
            devices_no = len(supervisor.supervisor.testcase.devices)
            # A reusable barrier to synchronize all devices at the end of a time step.
            self.barrier = ReusableBarrierCond(devices_no)
            # A dictionary of locks to ensure thread-safe access to data at specific locations.
            self.dict_location_lock = {}
        else:
            # Other devices will receive these from device 0 in setup_devices.
            self.barrier = None
            self.dict_location_lock = None

        self.all_devs = None

        # The main control thread for this device's simulation loop.
        self.master_thread = DeviceThread(self)
        self.master_thread.start()


    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id


    def setup_devices(self, devices):
        """Sets up shared objects and a reference to all devices.

        This is called after all devices are instantiated. Non-master devices
        find device 0 to get a reference to the shared barrier and lock dictionary.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id != 0:
            for dev in devices:
                if dev.device_id == 0:
                    self.barrier = dev.barrier
                    self.dict_location_lock = dev.dict_location_lock
                    break

        self.all_devs = devices


    def assign_script(self, script, location):
        """Assigns a script to be executed for a given location.

        If the script is None, it signals that all scripts for the current
        time point have been assigned, unblocking the main device thread.

        Args:
            script: The script object to be executed. Must have a `run` method.
            location: The location context for the script execution.
        """
        if script is not None:
            # Create a lock for the location if it's the first time we see it.
            if location not in self.dict_location_lock.keys():
                self.dict_location_lock[location] = Lock()
            self.scripts.append((script, location))
        else:
            # A None script is a sentinel indicating the end of script assignment for this time step.
            self.timepoint_done.set()


    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None


    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data


    def shutdown(self):
        """Shuts down the device by joining its threads."""
        for i in range(0, self.worker_threads_no):
            self.worker_threads[i].join()
        self.master_thread.join()


class WorkerThread(Thread):
    """A thread that executes computational scripts.

    Pulls tasks from a shared queue, gathers data, runs a script, and
    disseminates the result to itself and its neighbors.
    """

    def __init__(self, device, queue):
        """Initializes the worker thread.

        Args:
            device (Device): The parent device this worker belongs to.
            queue (Queue): The queue from which to fetch tasks.
        """
        Thread.__init__(self)
        self.device = device
        self.queue = queue


    def run(self):
        """The main execution loop for the worker."""
        while True:
            # A task is a tuple of (script_location, neighbors).
            (scr_loc, neighbours) = self.queue.get()
            # A None value for neighbors is a sentinel to terminate the thread.
            if neighbours is None:
                return

            (script, location) = scr_loc
            script_data = []

            # Acquire the lock for the location to ensure exclusive access.
            with self.device.dict_location_lock[location]:
                # Pre-condition: Gather data from all neighbors at the specified location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Also gather data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Run the script if there's any data to process.
                if script_data != []:
                    result = script.run(script_data)

                    # Post-condition: Update the data on all neighbors and the local device
                    # with the result of the computation, effectively broadcasting the result.
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

            self.queue.task_done()


class DeviceThread(Thread):
    """The main control thread for a device, orchestrating time steps."""

    def __init__(self, device):
        """Initializes the device's main thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        """The main simulation loop for the device."""
        while True:
            # Get the list of neighbors for the current time step from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If the supervisor signals the end, shut down the worker threads and exit.
            if neighbours is None:
                for _ in range(0, self.device.worker_threads_no):
                    self.device.queue.put((None, None))
                break

            # Wait for the supervisor to finish assigning all scripts for this time step.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Enqueue all assigned scripts for the worker threads to process.
            for src_loc in self.device.scripts:
                self.device.queue.put((src_loc, neighbours))

            # Wait for all tasks in the queue to be completed.
            self.device.queue.join()
            
            # Synchronize with all other devices before proceeding to the next time step.
            self.device.barrier.wait()
