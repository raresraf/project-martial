"""
Defines a device simulation using a thread-per-work-chunk concurrency model.

Each device's main `DeviceThread` splits the assigned scripts for a timepoint
into a fixed number of chunks (8) and spawns a new thread to process each chunk.
Synchronization is handled by joining these threads and then waiting on a global
barrier.
"""

from threading import Event, Thread, Lock
import Barrier


class Device(object):
    """
    Represents a device node in the simulation.

    Manages its own data, scripts, and a control thread (`DeviceThread`) that
    orchestrates the execution of scripts in parallel.

    Attributes:
        device_id (int): Unique ID for the device.
        sensor_data (dict): The device's local sensor data.
        supervisor: The simulation supervisor.
        scripts (list): A list of scripts to be executed in the current timepoint.
        setup_done (Event): Signals that this device's initial setup is complete.
        barrier (Barrier.Barrier): A global barrier to synchronize all devices.
        locks (dict): A shared dictionary of locks for all data locations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.setup_done = Event()
        self.devices = []
        self.barrier = None
        self.locks = None
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources for all devices in the simulation.

        Device 0 is the master, creating and sharing the global barrier and lock
        dictionary. All devices then signal that their setup is complete.
        """
        # Store references to all other devices.
        for device in devices:
            if self.device_id != device.device_id:
                self.devices.append(device)

        if self.device_id == 0:
            # Device 0 creates the shared barrier and lock dictionary.
            self.barrier = Barrier.Barrier(len(devices))
            self.locks = {}
            # Distribute the shared objects to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.locks = self.locks
        
        # Signal that this device has finished its setup phase.
        self.setup_done.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be processed. Lazily creates locks for new locations.
        """
        if script is not None:
            # If a location is seen for the first time, create a lock for it.
            # This is done by the master device (ID 0) and shared.
            if not (self.locks).has_key(location):
                self.locks[location] = Lock()
            self.scripts.append((script, location))
        else:
            # A None script signals that script assignment is complete for the timepoint.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    A control thread that manages script execution by spawning worker threads.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    @staticmethod
    def split(script_list, number):
        """
        Splits a list into a specified number of sub-lists (chunks).
        """
        res = [[] for i in range(number)]
        i = 0
        while i < len(script_list):
            part = script_list[i]
            res[i%number].append(part)
            i = i + 1
        return res

    def run_scripts(self, scripts, neighbours):
        """
        Target function for worker threads. Processes a list (chunk) of scripts.
        """
        for (script, location) in scripts:
            # Use a 'with' statement to ensure the lock is acquired and released.
            with self.device.locks[location]:
                script_data = []
                # Gather data from neighbors and self.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Run script and propagate results if data was found.
                if script_data:
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

    def run(self):
        """
        Main loop for the control thread.
        
        Waits for setup to complete, then enters a loop for each timepoint where
        it splits work, spawns threads, joins them, and synchronizes.
        """
        # Initial synchronization: wait for this device and all other devices
        # to complete their initial setup.
        self.device.setup_done.wait()
        for device in self.device.devices:
            device.setup_done.wait()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Shutdown signal.
                break

            # Wait for supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()
            
            if len(self.device.scripts) != 0:
                # Split the list of scripts into 8 chunks.
                scripts_list = self.split(self.device.scripts, 8)
                thread_list = []
                
                # Create and start a thread for each chunk of work.
                for scripts in scripts_list:
                    new_thread = Thread(target=self.run_scripts,
                                        args=(scripts, neighbours))
                    thread_list.append(new_thread)
                for thread in thread_list:
                    thread.start()
                
                # Wait for all spawned threads to complete their work.
                for thread in thread_list:
                    thread.join()
            
            # Reset the event for the next timepoint.
            self.device.script_received.clear()
            
            # Wait at the global barrier for all devices to finish this timepoint.
            self.device.barrier.wait()
