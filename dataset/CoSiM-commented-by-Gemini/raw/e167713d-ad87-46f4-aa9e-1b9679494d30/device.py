"""
device.py

@brief Simulates a device in a distributed system using a Bulk Synchronous Parallel (BSP) model.
@description This module defines the `Device` and `DeviceThread` classes, which together
simulate a node in a sensor network or distributed computing environment. The system operates
in synchronized timepoints (supersteps), coordinated by a global barrier. In each step,
devices execute scripts using data from themselves and their neighbors.
"""

from threading import Event, Thread, Lock
from barrier import Barrier  # Assumes a custom or third-party Barrier implementation.

class Device(object):
    """
    Represents a single device (or node) in the distributed network.

    A Device manages its own sensor data, a set of scripts to execute, and a pool
    of worker threads (`DeviceThread`) to process the scripts in parallel.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary mapping locations to sensor values.
            supervisor (object): A supervisor object responsible for providing network
                                 topology information (e.g., neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.neighbours = []
        self.scripts = []  # Master list of (script, location) tuples.
        self.temp_scripts = []  # Per-timepoint copy of scripts to be executed.

        # --- Synchronization Primitives ---
        self._thread_list = []
        # Signals that a timepoint's scripts have been assigned and are ready for processing.
        self.timepoint_done = Event()
        # Protects this device's own sensor_data during updates.
        self.device_lock = Lock()
        # Protects access to the temp_scripts list for work-stealing by threads.
        self.script_list_lock = Lock()
        # A shared dictionary of locks for fine-grained, location-based synchronization.
        self.locations_locks = {}
        # A global barrier to synchronize all threads across all devices in the system.
        self.device_thread_barrier = None
        self.thread_number = 8  # Number of worker threads per device.

        # --- Thread Creation ---
        # Pre-allocates a pool of worker threads.
        for thread_id in xrange(self.thread_number):
            thread = DeviceThread(self, thread_id)
            self._thread_list.append(thread)

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the global synchronization mechanisms for the entire system.

        This method sets up the shared barrier and location locks for all devices
        and starts the worker threads. It should be called on one device before
        the simulation begins.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Initialize the global barrier if it hasn't been set up yet.
        # The barrier must accommodate all threads from all devices.
        if self.device_thread_barrier is None:
            self.device_thread_barrier = Barrier(len(devices) * self.thread_number)
            for dev in devices:
                dev.device_thread_barrier = self.device_thread_barrier

        # Block Logic: Initialize the location-based locks if they are not already set.
        # This creates a shared lock for each unique data location across all devices.
        if not self.locations_locks:
            max_location = -1
            for dev in devices:
                for key in dev.sensor_data:
                    if key > max_location:
                        max_location = key
            for i in xrange(max_location + 1):
                self.locations_locks[i] = Lock()
            for dev in devices:
                dev.locations_locks = self.locations_locks

        # Start all worker threads for this device.
        for thread_id in xrange(self.thread_number):
            self._thread_list[thread_id].start()

    def assign_script(self, script, location):
        """
        Assigns a script to be run at a specific location for the next timepoint.

        Args:
            script (object): The script object with a `run` method. If None, it
                             signals the end of the current timepoint's script
                             assignments.
            location (int): The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts for the upcoming timepoint are assigned.
            # This unblocks the device threads waiting to start processing.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads to complete, shutting down the device."""
        for i in xrange(len(self._thread_list)):
            self._thread_list[i].join()


class DeviceThread(Thread):
    """
    A worker thread that executes scripts on behalf of a Device.
    """

    def __init__(self, device, thread_id):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
            thread_id (int): The unique ID of this thread within the device's pool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        The main loop for the worker thread, implementing the BSP model.
        """
        # Invariant: This loop represents the continuous operation of the device,
        # processing one timepoint (superstep) per iteration.
        while True:
            # --- Superstep Start: Synchronization Point ---
            self.device.device_thread_barrier.wait()

            # The first thread of each device is responsible for setup tasks for that device.
            if self.thread_id == 0:
                # Task 1: Update neighbor list from the supervisor.
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # --- Communication Phase: Synchronization Point ---
            self.device.device_thread_barrier.wait()

            # A None neighbor list is the signal from the supervisor to shut down.
            if self.device.neighbours is None:
                break

            # The first thread of each device prepares the scripts for the current timepoint.
            if self.thread_id == 0:
                # Wait for the supervisor to finish assigning all scripts for this timepoint.
                self.device.timepoint_done.wait()
                self.device.timepoint_done.clear()
                # Create a temporary, per-timepoint copy of the scripts to execute.
                self.device.temp_scripts = list(self.device.scripts)

            # --- Computation Phase: Synchronization Point ---
            self.device.device_thread_barrier.wait()

            # Block Logic: Each thread participates in executing scripts from the shared
            # temporary list until it is empty (a form of work-stealing).
            done_iter = False
            while True:
                item = ()
                # Safely pop a script from the shared list.
                self.device.script_list_lock.acquire()
                if len(self.device.temp_scripts) > 0:
                    item = self.device.temp_scripts.pop(0)
                else:
                    done_iter = True
                self.device.script_list_lock.release()

                if done_iter:
                    break

                script, location = item

                # Acquire the lock for the specific data location to prevent race conditions.
                self.device.locations_locks[location].acquire()

                # Gather data from this device and its neighbors for the given location.
                script_data = []
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Only run the script if there is relevant data to process.
                if script_data:
                    # Execute the computation.
                    result = script.run(script_data)

                    # --- Write-back Phase ---
                    # Update the data on all neighboring devices with the new result.
                    for device in self.device.neighbours:
                        device.device_lock.acquire()
                        device.set_data(location, result)
                        device.device_lock.release()

                    # Update the data on the local device.
                    self.device.device_lock.acquire()
                    self.device.set_data(location, result)
                    self.device.device_lock.release()

                # Release the lock for the location, allowing other threads to process it.
                self.device.locations_locks[location].release()
