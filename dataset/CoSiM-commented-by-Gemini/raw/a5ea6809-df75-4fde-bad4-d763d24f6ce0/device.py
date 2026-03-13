"""
This module defines a simulated Device and its associated threading constructs
for a distributed sensor network simulation.

The Device class represents a node in the network that can hold sensor data,
be assigned scripts to process that data, and communicate with its neighbors.
Synchronization between devices is managed using threading primitives like
Events, Locks, and a custom Condition-based barrier.
"""

from threading import Event, Thread, Condition, Lock

class Device(object):
    """
    Represents a single device in the simulated network.

    Each device has a unique ID, holds sensor data for different locations,
    and executes scripts assigned by a supervisor. It runs in its own thread
    to handle lifecycle events like script assignment and synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor data.
            supervisor (Supervisor): The central supervisor controlling the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()  # Event to signal when all scripts for a timepoint are assigned.
        self.scripts_dict = {}  # Maps locations to a list of scripts to be executed.
        self.locations_locks = {}  # Shared locks for each location to ensure atomic data access.
        self.timepoint_done = None  # A barrier for synchronizing all devices at the end of a timepoint.
        self.neighbours = None  # A list of neighboring devices.
        self.thread = DeviceThread(self)
        self.thread.start()


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources (barriers, locks) across a group of devices.

        This method ensures that all devices in the simulation share the same
        synchronization primitives for timepoints and data locations.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # --- Barrier setup for timepoint synchronization ---
        # If this device's barrier is not set, create one and propagate it
        # to all other devices in the simulation.
        nr_devices = len(devices)
        if self.timepoint_done is None:
            self.timepoint_done = ReusableBarrierCond(nr_devices)
            for device in devices:
                if device.timepoint_done is None and device != self:
                    device.timepoint_done = self.timepoint_done

        # --- Lock setup for per-location data access ---
        # For each location this device is aware of, create a shared lock
        # and propagate it to all other devices. This ensures that operations
        # on the same location are mutually exclusive across the system.
        for location in self.sensor_data.keys():
            if location not in self.locations_locks:
                self.locations_locks[location] = Lock()
                for device in devices:
                    if location not in device.locations_locks and 
                        device != self:
                        device.locations_locks[location] = 
                            self.locations_locks[location]

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location.

        A `None` script is a sentinel value indicating that all scripts for the
        current timepoint have been assigned.

        Args:
            script (Script): The script object to execute.
            location (str): The location whose data the script will process.
        """
        if script is not None:
            # Append the script to the list for the given location.
            if location in self.scripts_dict:
                self.scripts_dict[location].append(script)
            else:
                self.scripts_dict[location] = [script]
        else:
            # If the script is None, it signals the end of script assignment
            # for this timepoint, so we set the event.
            self.scripts_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (str): The location to query.

        Returns:
            The data for the location, or None if the location is not tracked.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates the sensor data for a given location.

        Args:
            location (str): The location to update.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread gracefully."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a Device."""

    def __init__(self, device):
        """
        Initializes the main device thread.

        Args:
            device (Device): The parent device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main lifecycle loop for the device.
        
        This loop continuously gets neighbors, waits for scripts, executes them,
        and synchronizes with other devices at the end of each timepoint.
        """
        while True:
            # At the beginning of each timepoint, get the current set of neighbors.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # A `None` neighbor list is the signal to terminate the simulation.
            if self.device.neighbours is None:
                break

            # Wait until the supervisor signals that all scripts for this
            # timepoint have been assigned.
            self.device.scripts_received.wait()

            # --- Script Execution Phase ---
            # Create and start a worker thread for each location that has scripts.
            threads = []
            for location in self.device.scripts_dict.keys():
                thread = DeviceWorkerThread(self.device, location)
                thread.start()
                threads.append(thread)

            # Wait for all worker threads to complete their script executions.
            for thread in threads:
                thread.join()

            # --- Cleanup and Synchronization Phase ---
            # Clear the received scripts and the event flag in preparation for the next timepoint.
            self.device.scripts_dict.clear()
            self.device.scripts_received.clear()

            # Wait at the barrier until all other devices have also finished this timepoint.
            self.device.timepoint_done.wait()

class DeviceWorkerThread(Thread):
    """A worker thread to execute scripts for a specific location."""

    def __init__(self, device, location):
        """
        Initializes the worker thread.

        Args:
            device (Device): The parent device object.
            location (str): The location this worker is responsible for.
        """
        Thread.__init__(self, name="Device Worker Thread %d for %s" % (device.device_id, location))
        self.device = device
        self.location = location

    def run(self):
        """
        Executes all assigned scripts for the worker's location.
        """
        for script in self.device.scripts_dict[self.location]:
            # --- Data Aggregation and Processing ---
            # Acquire the lock for this location to ensure that data from neighbors
            # is not modified while we are reading it.
            self.device.locations_locks[self.location].acquire()

            # Collect data for the current location from this device and all its neighbors.
            script_data = []
            for device in self.device.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # If there's any data to process, run the script and update the data
            # on this device and all its neighbors with the result.
            if script_data:
                result = script.run(script_data)
                
                # Propagate the result to all neighbors.
                for device in self.device.neighbours:
                    device.set_data(self.location, result)
                # Update the local data as well.
                self.device.set_data(self.location, result)

            # Release the lock for this location.
            self.device.locations_locks[self.location].release()


class ReusableBarrierCond(object):
    """
    A reusable barrier implemented using a Condition variable.

    This allows a set of threads to wait for each other to reach a certain point
    of execution before proceeding. It automatically resets after all threads
    have passed.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads  # Countdown for threads arriving at the barrier.
        self.cond = Condition()  # The underlying condition variable.

    def wait(self):
        """
        Causes the calling thread to wait at the barrier.

        The thread will block until `num_threads` have called this method.
        """
        self.cond.acquire()
        self.count_threads -= 1

        if self.count_threads == 0:
            # This is the last thread to arrive. Notify all waiting threads.
            self.cond.notify_all()
            # Reset the counter for the next use of the barrier.
            self.count_threads = self.num_threads
        else:
            # Not the last thread, so wait to be notified.
            self.cond.wait()

        self.cond.release()
