"""
This module provides a framework for simulating a network of interconnected devices.

Each device runs in its own thread, executing scripts that process sensor data.
Devices can communicate with their neighbors, and their execution is synchronized
at discrete timepoints using a reusable barrier. This is characteristic of a
discrete-time simulation for a distributed system or sensor network.
"""

from threading import *


class Device(object):
    """
    Represents a single device in the simulated network.

    Each device has a unique ID, its own sensor data, and is managed by a
    supervisor. It operates in a multi-threaded environment, with a main
    DeviceThread and multiple SlaveThreads for script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor
                                data, keyed by location.
            supervisor (object): The supervisor object that manages the network topology.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been assigned to the device.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that the device is ready to proceed to the next timepoint.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        # A lock to ensure thread-safe access to the device's sensor_data.
        self.lock_data = Lock()
        # A list of locks, one for each location, to manage concurrent access.
        self.lock_location = []
        # A barrier for synchronizing all devices at the end of a timepoint.
        self.time_barrier = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for all devices in the simulation.

        This method should be called on a single designated device (e.g., device_id 0)
        to set up the synchronization barrier and location-based locks that are
        shared across all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Pre-condition: This block should only be executed by one device to prevent
        # multiple initializations of shared resources.
        if self.device_id == 0:
            # Shared barrier to synchronize all devices after each computation step.
            self.time_barrier = ReusableBarrierSem(len(devices))

            for device in devices:
                device.time_barrier = self.time_barrier

            # Determine the total number of unique locations across all devices.
            loc_num = 0
            for device in devices:
                for location in device.sensor_data:
                    loc_num = max(loc_num, location)
            # Create a lock for each location to ensure atomic updates.
            for i in range(loc_num + 1):
                self.lock_location.append(Lock())

            # Distribute the shared location locks to all devices.
            for device in devices:
                device.lock_location = self.lock_location

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        If a script is provided, it is added to the device's script queue.
        If the script is None, it signals that the device has received all scripts
        for the current timepoint.

        Args:
            script (object): The script object to be executed.
            location (int): The location context for the script execution.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script is a sentinel value indicating the end of script assignments
            # for the current timepoint, allowing the device to start execution.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (int): The location from which to get data.

        Returns:
            The sensor data value if the location exists, otherwise None.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Thread-safely updates sensor data for a specific location.

        Args:
            location (int): The location to update.
            data: The new data value.
        """
        with self.lock_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    This thread orchestrates the device's lifecycle, waiting for a signal to
    begin a timepoint, executing all assigned scripts in parallel, and then
    synchronizing with all other devices before starting the next cycle.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device.
        """
        # Invariant: The loop continues as long as the supervisor indicates
        # there are neighbors, implying the simulation is active.
        while True:
            slaves = []

            # Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbours is the signal to terminate the simulation.
                break

            # Block until the supervisor signals the start of the next timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Pre-condition: All scripts for the current timepoint have been assigned.
            # Block Logic: Spawn a separate SlaveThread for each script to execute it
            # in parallel.
            for (script, location) in self.device.scripts:
                slave = SlaveThread(script, location, neighbours, self.device)
                slaves.append(slave)
                slave.start()

            # Wait for all slave threads to complete their execution.
            for i in range(len(slaves)):
                slaves.pop().join()

            # Invariant: All local script computations for this device are complete.
            # Block Logic: Wait at the barrier for all other devices to finish their
            # computations for the current timepoint before proceeding.
            self.device.time_barrier.wait()


class SlaveThread(Thread):
    """
    A thread that executes a single script on a device.

    It gathers data from its own device and its neighbors at a specific location,
    runs the script, and distributes the result back.
    """
    def __init__(self, script, location, neighbours, device):
        """
        Initializes the SlaveThread.

        Args:
            script (object): The script to execute.
            location (int): The location context for the script.
            neighbours (list): A list of neighboring Device objects.
            device (Device): The parent device.
        """

        Thread.__init__(self, name="Slave Thread of Device %d" % device.device_id)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """
        Executes the script's logic.
        """
        device = self.device
        script = self.script
        location = self.location
        neighbours = self.neighbours

        data = device.get_data(location)
        input_data = []
        # Get the specific lock for the location this script is operating on.
        this_lock = device.lock_location[location]

        if data is not None:
            input_data.append(data)

        # Pre-condition: The 'input_data' list is ready for aggregation.
        # Block Logic: Acquire the location-specific lock to ensure that data from
        # this device and its neighbors is read atomically. This prevents race
        # conditions where another device might be updating the same location.
        with this_lock:
            # Aggregate data from all neighbors at the same location.
            for neighbour in neighbours:
                temp = neighbour.get_data(location)

                if temp is not None:
                    input_data.append(temp)

            # Invariant: input_data contains a snapshot of data from self and neighbors.
            # Block Logic: If there is data to process, run the script and
            # broadcast the result to all participants (self and neighbors).
            if input_data != []:
                result = script.run(input_data)

                for neighbour in neighbours:
                    neighbour.set_data(location, result)

                device.set_data(location, result)


class ReusableBarrierSem():
    """
    A reusable barrier implementation using two semaphores.

    This allows a fixed number of threads to synchronize at a point, and then
    reuse the barrier for subsequent synchronization points. It uses a two-phase
    protocol to prevent threads from one "wave" from proceeding before all
    threads from the previous "wave" have left the barrier.
    """

    def __init__(self, num_threads):
        """
        Initializes the reusable barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads

        # Lock to protect access to the internal counters.
        self.counter_lock = Lock()
        # Semaphore for the first phase of the barrier.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for the second phase of the barrier, allowing for reuse.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            # Invariant: The last thread to arrive at the barrier is responsible
            # for releasing all other waiting threads.
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset for the next phase.
                self.count_threads2 = self.num_threads

        # All threads will block here until the last thread releases the semaphore.
        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase to ensure the barrier is safe for reuse."""
        with self.counter_lock:
            self.count_threads2 -= 1
            # Invariant: The last thread to enter the second phase is responsible
            # for resetting the barrier for the next full 'wait' cycle.
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset for the next use of phase1.
                self.count_threads1 = self.num_threads

        # All threads block here, ensuring no thread from this "wave" races ahead
        # into the next use of the barrier before all have completed this one.
        self.threads_sem2.acquire()
