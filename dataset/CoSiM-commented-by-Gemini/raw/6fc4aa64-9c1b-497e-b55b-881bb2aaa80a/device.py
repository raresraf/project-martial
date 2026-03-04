"""
Models a device in a simulated distributed environment, likely for sensor networks or IoT simulations.

This script defines a `Device` class that operates within a multi-threaded simulation. Each device
runs its own thread (`DeviceThread`), and can execute multiple scripts concurrently using worker
threads (`MyThread`). The system uses barriers for synchronization between devices and locks for
managing access to shared data "locations." This structure suggests a simulation of a distributed
system where devices interact with njihovim neighbors and process data at specific points in a
shared space.
"""

from threading import Event, Thread, Lock
import supervisor
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a single device in the simulated network.

    Manages device-specific data, script execution, and synchronization with other devices.
    Each device has a unique ID, its own sensor data, and a reference to a central supervisor.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id: A unique identifier for the device.
            sensor_data: A dictionary or map representing the device's local sensor data,
                         keyed by location.
            supervisor: A reference to the central supervisor object that manages the
                        overall simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.locations = []
        self.get_data_lock = Lock()
        self.ready = Event()
        self.devices = None
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the device with information about all other devices in the simulation.

        This method is likely called by a central setup process. It configures a shared
        reusable barrier for synchronization and a set of shared locks for each location.
        This setup is performed only by device 0, which then distributes the shared
        objects to all other devices.

        Args:
            devices: A list of all Device objects in the simulation.
        """

        self.devices = devices
        barrier = ReusableBarrierSem(len(devices))
        if self.device_id == 0:
            i = 0
            # Initialize a list of locks for each location.
            while i < 150:  
                self.locations.append(Lock())
                i = i + 1

            # Distribute the shared barrier and location locks to all devices.
            for dev in devices:
                dev.barrier = barrier
                dev.locations = self.locations
                dev.ready.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific location.

        If the script is None, it signals that all scripts for the current
        timepoint have been received.

        Args:
            script: The script object to be executed.
            location: The location at which the script should be executed.
        """

        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location in a thread-safe manner.

        Args:
            location: The location for which to retrieve data.

        Returns:
            The sensor data at the given location, or None if not available.
        """
        with self.get_data_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a given location.

        Args:
            location: The location at which to update data.
            data: The new data to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device.

    This thread orchestrates the device's lifecycle, waiting for scripts,
    executing them in parallel, and synchronizing with other devices.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device: The Device object that this thread belongs to.
        """
        Thread.__init__(self, name="%d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop of the device thread.

        It continuously gets neighbors from the supervisor, waits for scripts,
        executes them, and then synchronizes at a barrier.
        """
        self.device.ready.wait()

        while True:
            # Get the current set of neighbors from the supervisor.
            neigh = self.device.supervisor.get_neighbours()
            if neigh is None:
                break

            # Wait until all scripts for the current timepoint are received.
            self.device.script_received.wait()
            self.device.script_received.clear()

            rem_scripts = len(self.device.scripts)

            # Create a worker thread for each script.
            threads = []
            i = 0
            while i < rem_scripts:
                threads.append(MyThread(self.device, neigh, self.device.scripts, i))
                i = i + 1

            # Execute scripts in batches of 8 or less.
            if rem_scripts < 8:
                for thr in threads:
                    thr.start()
                for thr in threads:
                    thr.join()
            else:
                pos = 0
                while rem_scripts != 0:
                    if rem_scripts > 8:


                        for i in range(pos, pos + 8):
                            threads[i].start()
                        for i in range(pos, pos + 8):
                            threads[i].join()
                        pos = pos + 8
                        rem_scripts = rem_scripts - 8
                    else:
                        for i in range(pos, pos + rem_scripts):
                            threads[i].start()
                        for i in range(pos, pos + rem_scripts):
                            threads[i].join()
                        rem_scripts = 0

            # Synchronize with other devices at the barrier after all scripts are done.
            self.device.barrier.wait()


class MyThread(Thread):
    """
    A worker thread for executing a single script.

    Each `MyThread` instance is responsible for executing one script at a specific
    location, gathering data from neighbors, running the script, and updating the data.
    """

    def __init__(self, device, neigh, scripts, index):
        """
        Initializes the worker thread.

        Args:
            device: The parent Device object.
            neigh: A list of neighboring Device objects.
            scripts: The list of scripts to be executed.
            index: The index of the script in the list to be executed by this thread.
        """
        Thread.__init__(self, name="%d" % device.device_id)
        self.device = device
        self.neigh = neigh
        self.scripts = scripts
        self.index = index

    def run(self):
        """

        Executes the script.

        Acquires a lock for the script's location, gathers data from itself and its
        neighbors, runs the script, and updates the data at that location for
        itself and its neighbors.
        """
        (script, loc) = self.scripts[self.index]
        self.device.locations[loc].acquire()
        info = []
        # Gather data from neighbors at the specified location.
        for neigh_iter in self.neigh:
            aux_data = neigh_iter.get_data(loc)
            if aux_data is not None:
                info.append(aux_data)
        # Gather data from the current device.
        aux_data = self.device.get_data(loc)
        if aux_data is not None:
            info.append(aux_data)
        # Run the script if there is data to process.
        if info != []:
            result = script.run(info)
            # Update data for all neighbors and the current device.
            for neigh_iter in self.neigh:
                neigh_iter.set_data(loc, result)
                self.device.set_data(loc, result)
        self.device.locations[loc].release()