"""
This module defines a more complex, multi-threaded Device simulation for a
distributed system. Each Device has a pool of threads for concurrent script
execution and employs a two-tiered locking mechanism for data consistency.
"""
from threading import Event, Lock
from barrier import ReusableBarrierCond
from device_thread import DeviceThread  # Assuming device_thread.py contains the DeviceThread class

import multiprocessing

class Device(object):
    """
    Represents a multi-threaded device in a simulated distributed system.

    Each device has a pool of worker threads (DeviceThread) to process scripts
    concurrently. It uses a two-level locking system for synchronization:
    global locks for inter-device coordination and local locks for intra-device
    data access.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id: A unique identifier for the device.
            sensor_data: A dictionary representing the device's sensor data.
            supervisor: A supervisor object that manages the device network.
        """
        self.neighbours = []
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # scripts holds the original set of scripts for a timepoint.
        self.scripts = []
        # scripts_aux is a temporary copy for consumption by worker threads.
        self.scripts_aux = []
        self.counter = 0
        self.timepoint_done = Event()

        self.pop_script_lock = Lock()

        # A barrier for synchronizing all devices at the end of a timepoint.
        self.devices_barrier = None

        # Local locks for accessing this device's sensor data locations.
        self.location_locks = {location: Lock() for location in self.sensor_data}

        # Global locks shared among all devices for synchronizing access to
        # data locations across the network.
        self.global_location_locks = {}

        self.threads = []
        self.number_of_threads = 4 * multiprocessing.cpu_count()

        # A barrier for synchronizing the threads within this device.
        self.threads_barrier = ReusableBarrierCond(self.number_of_threads)

        for i in range(self.number_of_threads):
            self.threads.append(DeviceThread(self, i))

        for i in range(self.number_of_threads):
            self.threads[i].start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices_barrier(self, barrier):
        """Sets up the shared device barrier."""
        self.devices_barrier = barrier

    def setup_devices(self, devices):
        """
        Sets up the shared device barrier and global locks. This method is
        called by the master device (device_id 0).
        """
        if self.device_id == 0:
            self.devices_barrier = ReusableBarrierCond(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.devices_barrier = self.devices_barrier
                    device.global_location_locks = self.global_location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed. If the script is None, it signals
        the end of the timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its threads."""
        for i in range(self.number_of_threads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A worker thread for a Device.

    These threads execute scripts. One thread per device (thread_id 0) acts as
    a master, handling setup for each timepoint. All threads on a device
    synchronize using an internal barrier.
    """

    def __init__(self, device, thread_id):
        """
        Initializes the DeviceThread.

        Args:
            device: The Device instance this thread belongs to.
            thread_id: The ID of this thread within the device's thread pool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main execution loop for the device thread."""
        while True:
            # The master thread (thread_id 0) for this device prepares for the timepoint.
            if self.thread_id == 0:
                self.device.timepoint_done.clear()
                self.device.scripts_aux = list(self.device.scripts)
                self.device.neighbours = self.device.supervisor.get_neighbours()

            # All threads on this device wait here until the master thread is done.
            self.device.threads_barrier.wait()

            # If the supervisor signals the end of the simulation, break the loop.
            if self.device.neighbours is None:
                break

            # Process scripts from the auxiliary list.
            while True:
                if self.device.timepoint_done.is_set() and not self.device.scripts_aux:
                    break

                with self.device.pop_script_lock:
                    if self.device.scripts_aux:
                        (script, location) = self.device.scripts_aux.pop(0)
                    else:
                        continue

                script_data = []

                # Ensure a global lock exists for the location.
                if location not in self.device.global_location_locks:
                    # This part is not thread-safe and can lead to a race condition.
                    # A lock should be used here if multiple devices can create this lock.
                    self.device.global_location_locks[location] = Lock()

                # Acquire the global lock for the location to ensure exclusive access across all devices.
                with self.device.global_location_locks[location]:
                    # Gather data from neighbors.
                    for device in self.device.neighbours:
                        if location in device.sensor_data:
                            device.location_locks[location].acquire()
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                    
                    # Gather data from the current device if it's not in the neighbors list.
                    if self.device not in self.device.neighbours and location in self.device.sensor_data:
                        self.device.location_locks[location].acquire()
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                # If data was gathered, run the script and update the data.
                if script_data:
                    result = script.run(script_data)

                    # Update data on neighbors.
                    for device in self.device.neighbours:
                        if location in device.sensor_data:
                            device.set_data(location, result)
                            device.location_locks[location].release()
                    
                    # Update data on the current device.
                    if self.device not in self.device.neighbours and location in self.device.sensor_data:
                        self.device.set_data(location, result)
                        self.device.location_locks[location].release()

            # All threads on this device wait here after processing all scripts for the timepoint.
            self.device.threads_barrier.wait()

            # The master thread waits at the global device barrier for all other devices to finish.
            if self.thread_id == 0:
                self.device.devices_barrier.wait()
