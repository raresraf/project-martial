"""
This module defines a simulation framework for distributed devices that uses a
dynamic, per-timestep threading model for script execution.

Key components in this file are:
- CommonReference: A container for shared synchronization objects (barriers) that
  are distributed to all devices.
- Device: Represents a node in the network. It manages its data and is controlled
  by a main `DeviceThread`.
- DeviceThread: The main control thread for a device. It orchestrates the
  simulation time steps and dynamically creates worker threads for script
  execution within each step.
"""
from threading import Event, Thread, Lock
import multiprocessing
from barrier import SimpleBarrier # Assumed to be in a separate file.


class CommonReference(object):
    """
    A container for synchronization objects shared across all devices.

    This class ensures that all devices in the simulation use the same barrier
    instances, allowing for global synchronization.
    """
    def __init__(self, number_of_devices):
        """
        Initializes the shared reference.

        Args:
            number_of_devices (int): The total number of devices in the simulation.
        """
        self.lock = Lock()
        # A two-barrier system to create a synchronized execution phase.
        self.first_barrier = SimpleBarrier(number_of_devices)
        self.second_barrier = SimpleBarrier(number_of_devices)


class Device(object):
    """
    Represents a single device in the simulation.

    Each device has a main control thread (`DeviceThread`) that dynamically
    spawns and manages worker threads for script execution in each time step.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): The device's local data, keyed by location.
            supervisor (Supervisor): The central simulation supervisor.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        
        # --- Synchronization and Threading ---
        self.script_received = Event() # Signals that script assignment is complete.
        self.wait_for_reference = Event() # Signals that the CommonReference is set up.
        self.synch_reference = None # Will hold the shared CommonReference object.
        self.thread_list = [] # Stores dynamically created worker threads.
        self.number_of_processors = multiprocessing.cpu_count()
        
        # A dictionary of locks, one for each data location.
        self.location_locks = {entry: Lock() for entry in self.sensor_data}
        
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """

        Sets up the shared `CommonReference` object for all devices.

        This must be called on the root device (device_id == 0). It creates the
        shared object and distributes it, ensuring all devices are ready before
        the simulation begins.
        """
        self.devices = devices
        if self.device_id == 0:
            self.synch_reference = CommonReference(len(self.devices))
            for dev in self.devices:
                if dev.device_id != 0:
                    dev.synch_reference = self.synch_reference
            for dev in self.devices:
                dev.wait_for_reference.set()
        else:
            self.wait_for_reference.wait()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed in the current time step.

        The script is added to a list and is not executed until the main
        DeviceThread begins the execution phase.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of script assignment for this time step.
            self.script_received.set()

    def get_data(self, location):
        """
        Acquires a lock and returns data for a location.

        This method implements the first half of a coupled lock/unlock pattern.
        The caller is responsible for ensuring a corresponding `set_data` call
        is made to release the lock.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Sets data for a location and releases its lock.

        This method implements the second half of the coupled lock/unlock pattern
        started by `get_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()

    def shutdown(self):
        """Joins the main control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    Orchestrates the time steps and dynamically creates worker threads to
    execute scripts for each step.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def simple_task(self, neighbours, script, location):
        """
        A task to be run in a thread, executing a single script.

        It manages the coupled locking (`get_data`/`set_data`) for its operation.
        """
        script_data = []
        # Acquire locks and get data from neighbors.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        # Acquire lock and get data from self.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        # Run script and write results, releasing locks.
        if script_data:
            result = script.run(script_data)
            for device in neighbours:
                device.set_data(location, result)
            self.device.set_data(location, result)

    def run_tasks(self, neighbours, list_of_tuples):
        """A task that runs multiple scripts serially within one thread."""
        for (script, location) in list_of_tuples:
            self.simple_task(neighbours, script, location)

    def run(self):
        """The main lifecycle loop of the device."""
        while True:
            # Wait for supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Get neighbor list for this time step.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # --- First Global Barrier: Enter Execution Phase ---
            self.device.synch_reference.first_barrier.wait()

            if self.device in neighbours:
                neighbours.remove(self.device)

            # --- Dynamic Thread Creation and Execution ---
            self.list_of_thread_lists = []
            if len(self.device.scripts) <= (2 * self.device.number_of_processors):
                # Small workload: one thread per script.
                for (script, location) in self.device.scripts:
                    thread = Thread(target=self.simple_task, args=(neighbours, script, location))
                    self.device.thread_list.append(thread)
                    thread.start()
            else:
                # Large workload: partition scripts among cpu_count() threads.
                for i in range(self.device.number_of_processors):
                    self.list_of_thread_lists.append([])
                i = 0
                for script_tuple in self.device.scripts:
                    self.list_of_thread_lists[i % self.device.number_of_processors].append(script_tuple)
                    i += 1
                for i in range(self.device.number_of_processors):
                    if self.list_of_thread_lists[i]:
                        thread = Thread(target=self.run_tasks, args=(neighbours, self.list_of_thread_lists[i]))
                        self.device.thread_list.append(thread)
                        thread.start()
            
            # Wait for all dynamically created threads for this time step to complete.
            for thread in self.device.thread_list:
                thread.join()
            del self.device.thread_list[:]

            # --- Second Global Barrier: End Execution Phase ---
            self.device.synch_reference.second_barrier.wait()
