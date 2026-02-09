"""
A framework for simulating a distributed system of devices.

This module defines a `Device` class that operates in a multi-threaded 
environment, processing data based on dynamically assigned scripts. It uses
various synchronization primitives to coordinate actions between devices and
their internal worker threads, simulating a time-stepped execution model where
devices process data in parallel and synchronize at the end of each step.
"""

from threading import Event, Thread
from threading import Semaphore, Lock
from Barrier import ReusableBarrier


class Device(object):
    """
    Represents a single device in the simulated distributed system.

    A Device manages its own sensor data, a pool of worker threads, and its
    state within the larger system. It communicates with a central `supervisor`
    to get information about neighboring devices and receives scripts to execute.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's internal data,
                                keyed by location.
            supervisor: The central supervisor object that manages the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        # Used to signal that the initial setup of shared resources is complete.
        self.initialization_event = Event()
        # Limits the number of concurrent script executions to 8.
        self.free_threads = Semaphore(value=8)
        # Shared list of locks, one for each data location.
        self.locations = []
        # Shared barrier to synchronize all devices at the end of a timepoint.
        self.barrier = None

        self.device_threads = []

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for all devices in the system.

        This method is intended to be called by a single device (device_id 0)
        to create and distribute the shared location locks and the synchronization
        barrier to all other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        num_devices = len(devices)
        # Block Logic: Only the device with ID 0 performs the one-time setup.
        if self.device_id is 0:
            locations = []
            number_of_locations = 30

            # Create a lock for each potential data location.
            while number_of_locations > 0:
                locations.append(Lock())
                number_of_locations = number_of_locations - 1

            barrier = ReusableBarrier(num_devices)

            # Distribute the shared resources to all devices and signal initialization.
            for i in range(0, num_devices):
                devices[i].initialization_event.set()
                devices[i].locations = locations
                devices[i].barrier = barrier

    def assign_script(self, script, location):
        """
        Receives a script from the supervisor to be executed.

        If a valid script is provided, it's added to the queue and the
        `script_received` event is set. If the script is `None`, it signals
        that the current timepoint's script assignment is complete.

        Args:
            script: The script object to execute.
            location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A `None` script signals the end of script assignments for this time step.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location: The location key for the data.

        Returns:
            The data at the given location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location.

        Args:
            location: The location key for the data.
            data: The new data value to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def clear_threads(self):
        """Waits for all currently active worker threads to complete."""
        for thread in self.device_threads:
            thread.join()

        self.device_threads = []

    def shutdown(self):
        """Shuts down the device and its main control thread."""
        self.clear_threads()
        self.thread.join()

def execute(device, script, location, neighbours):
    """
    The target function for a worker thread, executing a single script.

    This function orchestrates the core logic of a data processing task:
    1. It acquires a lock for the target `location` to ensure exclusive access.
    2. It aggregates data from the parent `device` and its `neighbours`.
    3. It runs the `script` on the aggregated data.
    4. It broadcasts the result by updating the data on the parent and neighbours.
    5. It releases the semaphore to allow another worker thread to run.

    Args:
        device (Device): The parent device instance.
        script: The script to be executed.
        location: The data location to operate on.
        neighbours (list): A list of neighbouring devices.
    """
    # Pre-condition: Lock the specific location to prevent data races.
    with device.locations[location]:
        script_data = []
        
        # Block Logic: Aggregate data from all neighbours at the given location.
        for dev in neighbours:
            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)
        
        # Add the device's own data to the set.
        data = device.get_data(location)
        if data is not None:
            script_data.append(data)

        # Invariant: Only run the script if there is data to process.
        if script_data != []:
            # Run the computational script on the collected data.
            result = script.run(script_data)

            # Block Logic: Broadcast the result to all neighbours.
            for dev in neighbours:
                dev.set_data(location, result)
            
            # Update the device's own data.
            device.set_data(location, result)
        
        # Signal that this worker thread slot is now free.
        device.free_threads.release()

class DeviceThread(Thread):
    """
    The main control loop for a Device.

    This thread orchestrates the device's operation in discrete, synchronized
    timepoints. In each timepoint, it spawns worker threads to execute assigned
    scripts and then waits at a barrier for all other devices to finish before
    starting the next timepoint.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent Device instance this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        """The main execution logic for the device's lifecycle."""
        # Wait until the device with ID 0 has finished setting up shared resources.
        self.device.initialization_event.wait()

        while True:
            # Block Logic: This loop represents one full "timepoint" in the simulation.
            # It coordinates script execution and synchronizes with other devices.
            
            neighbours = self.device.supervisor.get_neighbours()
            # The supervisor signals shutdown by returning None.
            if neighbours is None:
                break

            # Wait until the supervisor signals that all scripts for this timepoint
            # have been assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Spawn a worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                # Pre-condition: Acquire a slot from the semaphore. This call blocks
                # if the maximum number of worker threads (8) is already running.
                self.device.free_threads.acquire()
                device_thread = Thread(target=execute, \
                           args=(self.device, script, location, neighbours))

                # Functional Utility: The worker thread is started here to execute a
                # single script in parallel with other workers.
                device_thread.start()
                self.device.device_threads.append(device_thread)

            # Clear the event for the next timepoint.
            self.device.timepoint_done.clear()

            # Invariant: All scripts for the current timepoint must complete before
            # proceeding. This call blocks until all spawned worker threads finish.
            self.device.clear_threads()

            # Invariant: All devices must reach this point before any can start the
            # next timepoint. This synchronizes the entire system.
            self.device.barrier.wait()