



"""
This module defines a framework for a distributed device simulation using a
statically-partitioned, multi-threaded approach within each device.

It consists of three main classes:
- ReusableBarrierCond: A custom implementation of a reusable barrier using
  threading.Condition for multi-thread synchronization.
- Device: Represents a single device in the simulation. Each device contains its
  own pool of worker threads.
- DeviceThread: The worker thread class. The core logic of script execution and
  synchronization happens here.

The architecture uses a complex system of global and local barriers to ensure
all threads across all devices operate in lock-step through the simulation's
timepoints.
"""
from threading import Event, Thread, Lock, Condition


class ReusableBarrierCond(object):
    """
    A reusable barrier implementation using a Condition variable.

    This barrier causes threads to block until a specified number of threads
    have called the wait() method. Once the required number of threads is
    reached, all waiting threads are released and the barrier resets for
    the next use.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier.

        Args:
            num_threads (int): The number of threads that must wait at the
                               barrier before they are all released.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Causes the calling thread to wait at the barrier.

        The thread will block until the required number of threads have arrived.
        """
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                # Last thread has arrived, notify all waiting threads.
                self.cond.notify_all()
                # Reset the barrier for the next round.
                self.count_threads = self.num_threads
            else:
                self.cond.wait()


class Device(object):
    """
    Represents a single device in the simulated network.

    Each device manages its own sensor data and a fixed-size pool of worker
    threads (`DeviceThread`) to execute assigned scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data.
            supervisor (Supervisor): The central supervisor object for the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        
        # Event to signal when all scripts for a timepoint have been assigned.
        self.timepoint_done = Event()
        
        # Event to signal that the global barrier has been set up.
        self.setup_device = Event()
        
        # Synchronization primitives.
        self.device_barrier = None  # A global barrier for all threads in the simulation.
        self.local_barrier = ReusableBarrierCond(8)  # A local barrier for this device's threads.
        self.location_lock = {}  # A shared dictionary of locks for data locations.
        self.neighbours = []

        # Each device has its own dedicated pool of 8 worker threads.
        self.threads = []
        for i in range(8):
            self.threads.append(DeviceThread(self, i))
        for thread in self.threads:
            thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global synchronization objects for all devices.

        This method must be called on the root device (device_id == 0). It creates a
        single global barrier and a single shared location lock dictionary and
        distributes them to all devices in the simulation.
        """
        if self.device_id == 0:
            # The global barrier must account for every thread from every device.
            my_barrier = ReusableBarrierCond(len(devices) * 8)
            my_location_lock = {}
            for device in devices:
                device.device_barrier = my_barrier
                device.location_lock = my_location_lock
                device.setup_device.set()  # Signal that setup is complete.

    def assign_script(self, script, location):
        """
        Assigns a new script to be executed during the current timepoint.

        Args:
            script (Script): The script to execute. If None, it signals the end
                             of script assignment for the current timepoint.
            location (any): The data location identifier for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Lazily initialize locks for new locations.
            if location not in self.location_lock:
                self.location_lock[location] = Lock()
        else:
            # A None script signals that the supervisor is done assigning scripts
            # for this timepoint, releasing the worker threads.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Gets data from a specific location. Not thread-safe on its own.

        Args:
            location (any): The location of the data to retrieve.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Sets data at a specific location. Not thread-safe on its own.

        Args:
            location (any): The location to write to.
            data (any): The data to be written.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its worker threads."""
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread belonging to a Device.

    It executes a statically partitioned subset of the device's assigned scripts.
    """

    def __init__(self, device, thread_id):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
            thread_id (int): A unique ID (0-7) for this thread within its device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main lifecycle of a worker thread."""
        # Wait until the global barrier is configured by the root device.
        self.device.setup_device.wait()

        while True:
            # --- Start of Timepoint Synchronization ---
            self.device.device_barrier.wait()

            # The first thread of each device is responsible for fetching the neighbor list.
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # Local barrier to ensure all threads on this device have the new neighbor list.
            self.device.local_barrier.wait()

            if self.device.neighbours is None:
                # A None neighbor list is the signal to terminate the simulation.
                break

            # Wait until the supervisor signals that all scripts for this timepoint are assigned.
            self.device.timepoint_done.wait()
            # --- End of Timepoint Synchronization ---

            # Statically distribute the workload. Thread 'i' handles scripts at indices i, i+8, i+16, ...
            index = self.thread_id
            while index < len(self.device.scripts):
                (script, location) = self.device.scripts[index]
                index += 8
                script_data = []

                # Acquire the shared lock for this specific location to ensure data consistency.
                self.device.location_lock[location].acquire()

                # Gather data from all neighbors.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Gather data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Execute the script and propagate the result.
                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                self.device.location_lock[location].release()

            # --- Start of Post-Computation Synchronization ---
            # Global barrier to ensure all threads across all devices have finished computation.
            self.device.device_barrier.wait()

            # The first thread of each device cleans up for the next timepoint.
            if self.thread_id == 0:
                self.device.timepoint_done.clear()
            # --- End of Post-Computation Synchronization ---

