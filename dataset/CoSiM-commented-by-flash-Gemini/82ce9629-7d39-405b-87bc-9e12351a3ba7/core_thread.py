"""
This module implements a device simulation framework using a core-thread
architecture. It defines `CoreThread` for executing individual scripts,
a `Device` class to manage sensor data and synchronization primitives,
and a `DeviceThread` to orchestrate script execution with a batching mechanism.
Synchronization relies on semaphores for location-specific data access and
a reusable barrier for global timepoint coordination.
"""

from threading import Thread

class CoreThread(Thread):
    """
    A worker thread responsible for executing a single script at a specific
    location for a device. It collects necessary sensor data from the local
    device and its neighbors, executes the script, and updates the sensor data
    according to locking mechanisms.
    """
    def __init__(self, device, script_id, neighbours):
        """
        Initializes a CoreThread instance.

        Args:
            device (Device): The Device instance this thread is associated with.
            script_id (int): The index of the script in `device.scripts` to be executed.
            neighbours (list): A list of neighboring Device instances.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script_id = script_id
        self.neighbours = neighbours

    def run(self):
        """
        The main execution method for the CoreThread. It retrieves its assigned
        script, acquires a semaphore for the script's location, collects data
        from local and neighbor devices, executes the script, and updates
        sensor data, ensuring thread-safe operations.
        """
        
        # Block Logic: Retrieves the script and its associated location using the `script_id`.
        (script, location) = self.device.scripts[self.script_id]

        # Functional Utility: Acquires a semaphore for the specific location to control
        # access to sensor data at that location, ensuring thread-safe operations.
        self.device.semaphores_list[location].acquire()

        script_data = [] # Accumulator for all data relevant to the script.

        # Functional Utility: Acquires `lock1` to protect access to reading local and neighbor data.
        self.device.lock1.acquire()

        # Block Logic: Gathers sensor data from neighboring devices at the specified location.
        for device in self.neighbours:
            data = device.get_data(location)

            if data is not None:
                script_data.append(data)

        # Block Logic: Retrieves the local device's sensor data for the current location.
        data = self.device.get_data(location)

        # Functional Utility: Releases `lock1` after reading data.
        self.device.lock1.release()

        if data is not None:
            script_data.append(data)

        # Pre-condition: Checks if any data was collected for the script to run.
        if script_data != []:
            # Functional Utility: Executes the assigned script with the collected data,
            # simulating sensor data processing.
            result = script.run(script_data)

            # Block Logic: Updates the sensor data of neighboring devices and the local device
            # with the script's result. This section is protected by `lock2`.
            for device in self.neighbours:

                # Functional Utility: Acquires `lock2` to protect the data modification phase.
                self.device.lock2.acquire()

                device.set_data(location, result) # Update neighbor's data.
                self.device.set_data(location, result) # Update local device's data.

                # Functional Utility: Releases `lock2` after data modification.
                self.device.lock2.release()

        # Functional Utility: Releases the semaphore for the current location, allowing other
        # threads to access the sensor data for that location.
        self.device.semaphores_list[location].release()


from threading import Thread, Event, Semaphore, Lock
from core_thread import CoreThread
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a simulated device in a distributed system. Each device
    manages its own sensor data, processes assigned scripts, and
    coordinates its operations with other devices via a supervisor
    and shared synchronization primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor data, keyed by location.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Signals that a script has been assigned or timepoint done.
        self.scripts = [] # List of (script, location) tuples assigned to this device.
        self.thread = DeviceThread(self) # The main thread for this device.
        self.semaphores_list = [] # List of semaphores for location-specific access control.
        # Functional Utility: `semaphore_setup_devices` is used to synchronize the setup of devices,
        # ensuring that the main thread doesn't start before shared resources are initialized.
        self.semaphore_setup_devices = Semaphore(1)
        self.lock1 = Lock() # A generic lock, possibly for protecting read access to data.
        self.lock2 = Lock() # A generic lock, possibly for protecting write access to data.
        self.barrier = None # Shared barrier for global synchronization.

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the device's shared synchronization mechanisms.
        Device 0 initializes the shared barrier and the list of semaphores
        for location-specific data access, which are then distributed to
        all other devices.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        # Functional Utility: Initializes a reusable barrier for global synchronization
        # across all device threads.
        barrier = ReusableBarrierCond(len(devices))

        # Block Logic: Initializes a list of semaphores, one for each potential location
        # (assuming a maximum of 50 locations). Each semaphore is initialized with a value of 1,
        # acting as a binary semaphore (mutex) for its corresponding location.
        for _ in range(50):
            semaphore = Semaphore(1)
            self.semaphores_list.append(semaphore)

        # Block Logic: Device 0 acts as the coordinator to initialize shared resources
        # and distribute them to all other devices.
        if self.device_id == 0:
            for device in devices:
                device.semaphores_list = self.semaphores_list
                device.barrier = barrier
                # Functional Utility: Acquires `semaphore_setup_devices` to block other
                # device threads from starting their `run` method until setup is complete.
                device.semaphore_setup_devices.acquire()
        # Functional Utility: Starts the main thread for this device.
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location for this device.
        If a script is provided, it is added to the list of scripts.
        If no script is provided (None), it signals that all script assignments
        for the current timepoint are complete.

        Args:
            script (object or None): The script object to assign, or None if the timepoint is complete.
            location (int): The numerical identifier for the location associated with the script.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Functional Utility: Signals to the device thread that all script assignments
            # for the current timepoint have been received.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location for which to retrieve data.

        Returns:
            any: The sensor data for the specified location, or None if not found.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location.

        Args:
            location (int): The location for which to set data.
            data (any): The new data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device thread, waiting for its completion.
        """
        
        self.thread.join()

class DeviceThread(Thread):
    """
    The main thread for a Device. It orchestrates the collection of neighbor
    information, manages the execution of scripts by dispatching them to
    `CoreThread` instances (which might be batched), and handles synchronization
    with other devices using a barrier.
    """

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread. It ensures device setup
        is complete, continuously processes neighbor data, executes scripts
        using `CoreThread`s (possibly in batches), and synchronizes with other
        devices at timepoints.
        """
        
        # Functional Utility: Releases a semaphore to signal that this device's
        # thread has started, allowing the `setup_devices` method to proceed.
        self.device.semaphore_setup_devices.release()

        while True:
            # Block Logic: Fetches the current set of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Pre-condition: Checks if a shutdown signal has been received from the supervisor.
            if neighbours is None:
                break

            # Functional Utility: Blocks until all scripts for the current timepoint have been
            # assigned to the device.
            self.device.script_received.wait()
            # Functional Utility: Clears the event for the next timepoint.
            self.device.script_received.clear()

            cores = [] # List to hold `CoreThread` instances.
            script_id = 0 # Counter for the current script's index.
            scripts_number = len(self.device.scripts) # Total number of scripts for this timepoint.

            # Block Logic: Creates a `CoreThread` for each script assigned to the device.
            for i in xrange(scripts_number):
                core_thread = CoreThread(self.device, script_id, neighbours)
                cores.append(core_thread)
                script_id = script_id + 1

            begin = 0 # Index to keep track of the start of the current batch of scripts.

            # Block Logic: Implements a batching mechanism for executing `CoreThread`s.
            # If the number of scripts exceeds 8, they are executed in batches of 8.
            # This is an optimization for managing concurrent thread execution.
            if scripts_number > 8:

                while scripts_number >= 0:

                    if scripts_number >= 8:
                        index = begin
                        # Block Logic: Starts a batch of 8 `CoreThread`s.
                        i = 0
                        while i < 8:
                            cores[index + i].start()
                            i = i + 1

                        index = begin
                        # Block Logic: Joins (waits for completion of) the batch of 8 `CoreThread`s.
                        i = 0
                        while i < 8:
                            cores[index + i].join()
                            i = i + 1

                        scripts_number = scripts_number - 8 # Decrements remaining scripts count.

                        begin = begin + 8 # Moves the starting index for the next batch.

                    else: # Block Logic: Handles the last batch of scripts if it's less than 8.
                        index = begin
                        # Block Logic: Starts the remaining `CoreThread`s.
                        i = 0
                        while i < scripts_number:
                            cores[index + i].start()
                            i = i + 1

                        index = begin
                        # Block Logic: Joins the remaining `CoreThread`s.
                        i = 0
                        while i < scripts_number:
                            cores[index + i].join()
                            i = i + 1

                        break
            else: # Block Logic: If the total number of scripts is 8 or less, execute all at once.
                # Block Logic: Starts all `CoreThread`s.
                for index in xrange(scripts_number):
                    cores[index].start()

                # Block Logic: Joins all `CoreThread`s, waiting for their completion.
                for index in xrange(scripts_number):
                    cores[index].join()

            # Functional Utility: Synchronizes with all other DeviceThread instances
            # across devices using a shared barrier, ensuring all devices complete
            # their current timepoint processing before proceeding.
            self.device.barrier.wait()
