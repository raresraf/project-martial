"""
This module provides a framework for simulating a network of interconnected devices.

It defines the `Device` class, representing an individual computational node, and the
`DeviceThread` class, which manages the execution and synchronization logic for each
device. The simulation appears to be time-stepped, with devices synchronizing at each
step to execute scripts that can read and modify data at shared "locations". This
suggests an application in areas like sensor networks, distributed computing simulations,
or multi-agent systems.
"""

from _threading_local import local
from threading import Event, Thread, Lock, RLock, Condition, Semaphore
from barrier import *

class Device(object):
    """
    Represents a single device in the distributed simulation.

    Each device runs in its own thread, processes scripts, and synchronizes with
    other devices via a barrier mechanism. It uses locks to ensure data consistency
    when accessing information associated with specific locations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): A dictionary holding the device's local data,
                                keyed by location.
            supervisor (object): An external entity that manages the overall simulation,
                                 providing information like network topology (neighbors).
        """
        self.device_id = device_id
        self.lock = Lock()
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been assigned to the device.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that the device has completed its work for the current timepoint.
        self.timepoint_done = Event()

        # The main execution thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Barrier for synchronizing all devices at the end of a timepoint.
        self.sync_barrier = None
        self.devices = []
        # Locks for ensuring exclusive access to locations.
        self.location_locks = {}

        self.nbs = []


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """
        Assigns a synchronization barrier to the device.

        Args:
            barrier (Barrier): The barrier instance to be used for synchronization.
        """
        self.sync_barrier = barrier

    def sync_with_others(self):
        """Blocks until all participating devices have reached the barrier."""
        self.sync_barrier.wait()

    def set_locks(self, locks):
        """
        Assigns a collection of location-based locks to the device.

        Args:
            locks (dict): A dictionary mapping locations to Lock objects.
        """
        self.location_locks = locks

    def get_lock(self, location):
        """
        Retrieves the lock for a specific location.

        Args:
            location: The location for which to get the lock.

        Returns:
            Lock: The lock associated with the given location.
        """
        return self.location_locks[location]

    def setup_devices(self, devices):
        """
        Initializes the simulation environment, including barriers and locks.

        This method is intended to be called on a single device (the coordinator, device_id 0)
        to set up the shared synchronization primitives for all devices in the simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.devices = devices
        # Pre-condition: This setup logic should only be executed by one device to avoid redundancy.
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            locks = {}
            # Invariant: Creates a lock for each unique location across all devices.
            for dev in devices:
                for loc in dev.sensor_data:
                    locks[loc] = Lock()
                dev.set_barrier(barrier)
            # Invariant: Distributes the same set of locks to all devices.
            for dev in devices:
                dev.set_locks(locks)

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific location.

        Args:
            script (object): The script object to be executed. Must have a `run` method.
            location: The location context for the script execution.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script is used as a signal to mark the end of a timepoint.
            self.timepoint_done.set()

    def get_scripts(self):
        """
        Retrieves all scripts currently assigned to this device.

        Returns:
            list: A list of tuples, each containing the device itself, a script,
                  its location, and the current list of neighbors.
        """
        return [(self, s, l, self.nbs) for (s, l) in self.scripts]

    def get_data(self, l):
        """
        Retrieves sensor data for a given location in a thread-safe manner.

        Args:
            l: The location from which to retrieve data.

        Returns:
            The data at the specified location, or None if the location is not found.
        """
        self.lock.acquire()
        ret = self.sensor_data[l] if l in self.sensor_data else None
        self.lock.release()
        return ret

    def set_data(self, location, data):
        """
        Updates sensor data at a given location in a thread-safe manner.

        Args:
            location: The location at which to update data.
            data: The new data value to be set.
        """
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()
    def shutdown(self):
        """Shuts down the device by stopping its execution thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The execution thread for a Device.

    This thread manages the device's lifecycle, including waiting for work,
    executing scripts, and synchronizing with other devices.
    """
    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device instance this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
        # Limits the number of concurrent script execution threads.
        self.max_running_threads_cnt = 8

    @staticmethod
    def exec_script(invoker_device, script, location, neighbourhood):
        """
        Executes a given script in the context of a specific location.

        This static method is the target for script execution threads. It gathers data
        from the invoking device and its neighbors at the specified location, runs the
        script on the collected data, and then propagates the result back to all
        involved devices. Access to the location is serialized via a lock.

        Args:
            invoker_device (Device): The device that initiated the script execution.
            script (object): The script to run.
            location: The location context.
            neighbourhood (list): A list of neighboring Device objects.
        """
        script_data = []
        # Pre-condition: Acquire lock to ensure exclusive access to the location's data.
        invoker_device.location_locks[location].acquire()
        
        # Block Logic: Gathers data from all neighbors at the specified location.
        for device in neighbourhood:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        # Also gather data from the invoking device itself.
        data = invoker_device.get_data(location)
        if data is not None:
            script_data.append(data)
        # Invariant: Only run the script if there is data to process.
        if script_data != []:
            
            # The core computation of the script.
            result = script.run(script_data)
            
            # Block Logic: Propagates the script's result to all neighbors.
            for device in neighbourhood:
                device.set_data(location, result)
            
            # Also update the invoking device's data.
            invoker_device.set_data(location, result)
        # Post-condition: Release the lock.
        invoker_device.location_locks[location].release()

    def run(self):
        """The main loop of the device thread."""
        while True:
            
            # At the beginning of each timepoint, get the current set of neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbors signals simulation shutdown.
                break

            self.device.nbs = neighbours
            
            # Block Logic: Waits for the supervisor to signal that all scripts for the
            # current timepoint have been assigned.
            self.device.timepoint_done.wait()

            
            # Block Logic: Synchronize with all other devices before starting script execution.
            # This ensures that all devices have received their scripts for the timepoint.
            self.device.sync_with_others()
            
            scrpts = []
            threads = []

            
            # Gathers all scripts assigned to the device for this timepoint.
            scrpts.extend(self.device.get_scripts())
            running_threads_cnt = 0

            
            # Block Logic: Spawns threads to execute each assigned script.
            # Manages a simple thread pool to limit concurrency.
            for (d, s, l, n) in scrpts:
                thread = Thread(name="T",
                                target=DeviceThread.exec_script,
                                args=(d, s, l, n))
                threads.append(thread)
                thread.start()
                running_threads_cnt += 1
                
                
                # Invariant: If the number of running threads reaches the maximum,
                # wait for the oldest thread to complete before spawning a new one.
                if running_threads_cnt >= self.max_running_threads_cnt:
                    wthread = threads.pop(0)
                    running_threads_cnt -= 1
                    wthread.join()

            
            # Block Logic: Wait for all remaining script execution threads to complete.
            for thread in threads:
                thread.join()

            
            # Reset the timepoint event to prepare for the next simulation step.
            self.device.timepoint_done.clear()