from threading import Event, Thread, Lock
# Assumes a 'barrier.py' module exists with a ReusableBarrierCond class.
import barrier

class Device(object):
    """
    Represents a device within a simulated network, featuring a leader-based
    setup for synchronization primitives.

    Architectural Role: This class models a device that executes computational
    scripts in a time-stepped simulation. This version uses a leader-follower
    pattern for initialization, where one device ("leader") creates the shared
    synchronization objects (barrier and locks) and the other devices
    ("followers") retrieve them from the leader. It also employs a two-level
    locking strategy for thread-safe data access.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device instance.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): The initial local data for the device, keyed by location.
            supervisor (object): The central supervisor managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that all scripts for a time-step have been received.
        self.scripts_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        # Lock to protect this device's internal `sensor_data` dictionary.
        self.data_lock = Lock()
        # A dictionary of location-specific locks, shared across all devices.
        self.list_locks = {}
        # The main synchronization barrier, shared across all devices.
        self.barrier = None
        self.devices = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects using a leader-follower model.
        
        Functional Utility: The first device in the list acts as the leader, responsible for
        creating the shared barrier and the dictionary of location-based locks. All other
        devices act as followers, obtaining these shared objects from the leader. This
        ensures all devices use the exact same instances of the synchronization primitives.
        """
        self.devices = devices

        # Block Logic: Leader-follower initialization.
        # Pre-condition: `self.devices` list is populated.
        # Invariant: The first device creates the synchronization objects; all others
        # receive a reference to them.
        if self.device_id == self.devices[0].device_id:
            # Leader device logic.
            self.barrier = barrier.ReusableBarrierCond(len(self.devices))
            # Create a lock for each unique data location across all devices.
            for dev in self.devices:
                for location in dev.sensor_data:
                    self.list_locks[location] = Lock()
        else:
            # Follower device logic.
            self.barrier = devices[0].get_barrier()
            self.list_locks = devices[0].get_list_locks()
        
        # The device's main thread is started only after setup is complete.
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to the device. A `None` script signals processing to start."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A `None` script is the trigger to unblock the DeviceThread and begin the time-step.
            self.scripts_received.set()

    def get_barrier(self):
        """Provides access to the shared barrier instance."""
        return self.barrier

    def get_list_locks(self):
        """Provides access to the shared dictionary of location-based locks."""
        return self.list_locks

    def get_data(self, location):
        """
        Safely retrieves data from the device's local data store.

        Note: This method uses an internal lock to protect the `sensor_data`
        dictionary from concurrent read/write operations.
        """
        with self.data_lock:
            data = self.sensor_data.get(location)
        return data

    def set_data(self, location, data):
        """
        Safely updates data in the device's local data store.
        
        Note: This method also uses the internal `data_lock` for thread safety.
        """
        with self.data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to complete."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a Device, which batches script execution.

    Functional Utility: This thread orchestrates the device's lifecycle. It waits for
    the `scripts_received` event, then processes the assigned scripts by creating
    `ScriptThread` workers. It uniquely batches these workers, running up to 8 at
    a time to control concurrency.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals shutdown by returning `None`.
                break

            # Wait until the supervisor signals that all scripts for the time-step are assigned.
            self.device.scripts_received.wait()
            self.device.scripts_received.clear()

            # Block Logic: Thread batching for script execution.
            # Invariant: Scripts are processed in batches of 8 to limit concurrency.
            threads = []
            for (script, location) in self.device.scripts:
                threads.append(
                    ScriptThread(self.device, script, location, neighbours))
                # When a batch of 8 is ready, start them, wait for them to finish,
                # and then clear the batch.
                if len(threads) == 8:
                    for thr in threads:
                        thr.start()
                    for thr in threads:
                        thr.join()
                    threads = []
            
            # Start and join any remaining threads in the last, possibly incomplete, batch.
            for thr in threads:
                thr.start()
            for thr in threads:
                thr.join()

            # Wait at the barrier for all devices to finish their time-step before proceeding.
            self.device.barrier.wait()

class ScriptThread(Thread):
    """A worker thread that executes a single script on a specific data location."""
    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script logic, ensuring exclusive access to the data location.
        """
        # Acquire the global lock for this specific location to prevent race conditions
        # with other scripts that might be operating on the same location.
        self.device.list_locks[self.location].acquire()

        script_data = []

        # Block Logic: Data gathering phase (under lock).
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Script execution and data propagation (under lock).
        if script_data:
            result = self.script.run(script_data)

            # Propagate the result to neighbors and the local device.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        # Release the location-specific lock.
        self.device.list_locks[self.location].release()