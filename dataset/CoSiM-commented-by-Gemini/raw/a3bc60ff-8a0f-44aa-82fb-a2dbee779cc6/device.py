"""
Models a device in a distributed sensor network simulation.

This script defines a device's behavior using a thread-per-script execution
model. Devices are synchronized at each time step using a shared, class-level
barrier. Access to sensor data for specific locations is controlled by a
class-level list of locks.
"""


from threading import Event, Thread, Lock, Condition


class Device(object):
    """
    Represents a single device node in the simulated network.

    This implementation uses class-level variables to hold the shared barrier and
    location locks. Each device spawns a new thread for each script execution
    in every time step, rather than using a persistent worker pool.
    """
    
    # Class-level variables for objects shared across all device instances.
    dev_barrier = None
    dev_locks = []

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's local sensor readings, keyed by location.
            supervisor (Supervisor): The central supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Signals that all scripts for a timepoint are assigned.
        self.scripts = []
        self.device_lock = Lock() # Lock for this device's own sensor_data dictionary.
        self.devices = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared, class-level resources for all devices.

        Device 0 is responsible for creating a list of location-based locks
        and a reusable barrier, storing them in class variables (`Device.dev_locks`
        and `Device.dev_barrier`) to be shared by all instances.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.devices = devices

        # Device 0 acts as the master for one-time setup.
        if self.devices[0].device_id == self.device_id:
            list_loc = []
            # Gather all unique locations from all devices.
            for device_ in self.devices:
                for location in list(device_.sensor_data.viewkeys()):
                    if location not in list_loc:
                        list_loc.append(location)

            # Create a lock for each unique location.
            # NOTE: This implementation implicitly assumes locations can be mapped to indices.
            for index in range(len(list_loc)):
                Device.dev_locks.append(Lock())

            # Create the shared barrier if it doesn't exist.
            if Device.dev_barrier is None:
                Device.dev_barrier = ReusableBarrierCond(len(self.devices))

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script signals the end of assignments.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location, protected by an instance-level lock.
        """
        with self.device_lock:
            if location in self.sensor_data:
                return self.sensor_data[location]
            else:
                return None

    def set_data(self, location, data):
        """
        Updates sensor data for a specific location, protected by an instance-level lock.
        """
        self.device_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.device_lock.release()

    def shutdown(self):
        """Waits for the main device thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device, managing its time-stepped execution.
    """

    def __init__(self, device):
        """Initializes the main thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main time-stepped loop of the device.

        In each timepoint, it waits for scripts to be assigned, then spawns a new
        `ScriptThread` for each script. It waits for all its script threads to
        complete before synchronizing with all other devices at a global barrier.
        """
        while True:

            # Get neighbours for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            threads_script = []

            # Create and collect a new thread for each assigned script.
            for (script, location) in self.device.scripts:
                thread = ScriptThread(self.device, script, neighbours, location)
                threads_script.append(thread)

            # Start all script threads for this timepoint.
            for thread in threads_script:
                thread.start()

            # Wait for all script threads to complete.
            for thread in threads_script:
                thread.join()
            
            # Synchronize with all other devices before proceeding to the next timepoint.
            Device.dev_barrier.wait()


class ScriptThread(Thread):
    """
    A short-lived thread created to execute a single script on a given location.
    """

    def __init__(self, device, script, neighbours, location):
        """
        Initializes the script-executing thread.

        Args:
            device (Device): The parent device.
            script (Script): The script to execute.
            neighbours (list): List of neighbouring devices.
            location (int): The location to operate on (used as an index for locking).
        """
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.neighbours = neighbours
        self.location = location

    def run(self):
        """
        Executes the script. It acquires a global, location-based lock,
        gathers data from itself and neighbours, runs the script, and
        broadcasts the result back to itself and neighbours.
        """
        script_data = []

        # Acquire a lock based on location index. This is a critical section
        # shared across all devices operating on the same location.
        Device.dev_locks[self.location].acquire()

        # Gather data from all neighbours.
        for device_ in self.neighbours:
            data = device_.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Gather data from the device itself.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Execute the script with the collected data.
            result = self.script.run(script_data)
            
            # Broadcast the result to all neighbours and self.
            for device_ in self.neighbours:
                device_.set_data(self.location, result)
            self.device.set_data(self.location, result)

        # Release the location-based lock.
        Device.dev_locks[self.location].release()


class ReusableBarrierCond(object):
    """
    A simple, custom implementation of a reusable barrier using a Condition variable.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Causes a thread to wait at the barrier. The last thread to arrive
        resets the barrier and notifies all waiting threads.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # This is the last thread, wake up all others.
            self.cond.notify_all()
            # Reset the barrier for the next use.
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()