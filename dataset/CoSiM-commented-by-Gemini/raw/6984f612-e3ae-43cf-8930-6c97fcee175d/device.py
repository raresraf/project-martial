"""
Models a distributed system of devices for parallel data processing.

This module contains the classes to simulate a network of devices that
concurrently execute scripts on sensor data. It features custom thread-level
and system-wide synchronization using barriers and locks to coordinate the
computation across multiple devices and their internal threads.
"""

from threading import Lock, Event, Thread, Condition

class ReusableBarrier():
    """
    A reusable barrier synchronization primitive.

    This barrier allows a specified number of threads to wait for each other to
    reach a certain point of execution before any of them are allowed to proceed.
    After releasing the threads, the barrier resets itself and can be used again.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must call wait()
                               before they are all released.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()


    def wait(self):
        """
        Causes a thread to wait until all threads have called this method.
        """
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                # All threads have arrived, notify all waiting threads.
                self.cond.notify_all()
                # Reset the barrier for the next use.
                self.count_threads = self.num_threads
            else:
                # Wait until all other threads have arrived.
                self.cond.wait()


class Device(object):
    """
    Represents a single device in the distributed network simulation.

    Each device has its own sensor data and a pool of worker threads. It receives
    computation scripts from a supervisor and coordinates with neighboring devices
    to process data in a distributed manner.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary mapping locations to sensor readings.
        supervisor: A reference to the central supervisor object.
        scripts (list): A list of (script, location) tuples to be executed.
        barrier (ReusableBarrier): A system-wide barrier for synchronization.
        lockforlocation (dict): A shared dictionary of locks for each data location.
        threads (list): A list of DeviceThread worker objects.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance and starts its worker threads.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor: The supervisor object managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.gotneighbours = Event()
        self.zavor = Lock() # Lock to protect neighbor discovery.
        self.threads = []
        self.neighbours = []
        self.nthreads = 8
        self.barrier = ReusableBarrier(1) # Initially a placeholder barrier.
        self.lockforlocation = {}
        self.num_locations = supervisor.supervisor.testcase.num_locations
        for i in xrange(self.nthreads):
            self.threads.append(DeviceThread(self, i))
        for i in xrange(self.nthreads):
            self.threads[i].start()


    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization objects for a group of devices.

        This method provides all devices with a common, system-wide barrier and a
        shared set of locks for each location, ensuring correct synchronization
        across the entire simulated network.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Create a single barrier for all threads in all devices.
        barrier = ReusableBarrier(devices[0].nthreads * len(devices))
        # Create a single lock for each location, shared across all devices.
        lockforlocation = {}
        for i in xrange(0, devices[0].num_locations):
            lock = Lock()
            lockforlocation[i] = lock
        for i in xrange(0, len(devices)):
            devices[i].barrier = barrier
            devices[i].lockforlocation = lockforlocation


    def assign_script(self, script, location):
        """
        Assigns a computation script to this device for a specific location.

        Called by the supervisor to provide work. A `None` script is a sentinel
        value indicating that all scripts for the current timepoint have been assigned.

        Args:
            script: The script object to be executed.
            location: The data location the script applies to.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # All scripts for this timepoint received, unblock worker threads.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location: The location to query for data.

        Returns:
            The sensor data, or None if the location is not available.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates the sensor data for a given location.

        Args:
            location: The location to update.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads to complete."""
        for i in xrange(self.nthreads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A worker thread belonging to a Device.

    Executes assigned scripts in a loop, synchronizing with other threads
    and devices at the end of each computation cycle (timepoint).
    """
    def __init__(self, device, id_thread):
        """
        Initializes the worker thread.

        Args:
            device (Device): The parent device object.
            id_thread (int): The unique ID of this thread within the device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id_thread = id_thread

    def run(self):
        """
        The main execution loop for the thread.

        This loop coordinates the entire distributed computation cycle: fetching
        neighbors, waiting for work, processing assigned scripts by gathering
        data from neighbors, executing the script, broadcasting the result, and
        synchronizing at a global barrier.
        """
        while True:

            # Only one thread per device needs to get the list of neighbours.
            with self.device.zavor:
                if not self.device.gotneighbours.is_set():
                    self.device.neighbours = self.device.supervisor.get_neighbours()
                    self.device.gotneighbours.set()

            if self.device.neighbours is None:
                # Shutdown signal from the supervisor.
                break

            # Wait until the supervisor signals that all scripts for the timepoint are assigned.
            self.device.timepoint_done.wait()

            # Statically partition the device's scripts among its threads.
            myscripts = []
            # The +1 in the step seems unusual; typically it would be just self.device.nthreads.
            # This might be a bug or for a specific non-standard workload distribution.
            for i in xrange(self.id_thread, len(self.device.scripts), self.device.nthreads + 1):
                myscripts.append(self.device.scripts[i])

            # Process the assigned partition of scripts.
            for (script, location) in myscripts:
                # Acquire the lock for this specific location to prevent race conditions
                # with other threads (from any device) working on the same location.
                with self.device.lockforlocation[location]:
                    script_data = []

                    # Gather data from all neighboring devices for the given location.
                    for device in self.device.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                    # Include own data.
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data:
                        # Run the computation.
                        result = script.run(script_data)
                        # Broadcast the result back to all neighbors and self.
                        for device in self.device.neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)

            # --- Synchronization Point 1 ---
            # All threads from all devices must finish their computation for the
            # current timepoint before proceeding.
            self.device.barrier.wait()

            # One designated thread (thread 0) resets the state for the next timepoint.
            if self.id_thread == 0:
                self.device.timepoint_done.clear()
                self.device.gotneighbours.clear()

            # --- Synchronization Point 2 ---
            # Ensures no thread loops around and waits on `timepoint_done` before
            # thread 0 has cleared it.
            self.device.barrier.wait()
			