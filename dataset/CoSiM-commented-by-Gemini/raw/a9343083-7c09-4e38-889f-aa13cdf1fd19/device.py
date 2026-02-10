from threading import Event, Thread, Lock
import ReusableBarrier

class Device(object):
    """
    Represents a single device in a simulated network environment.

    Architectural Role: This class models a device that holds local data,
    executes computational scripts, and communicates with its neighbors. It
    operates within a time-stepped simulation, coordinated by a central
    supervisor and synchronized with other devices via a shared barrier and
    a set of locks.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device instance.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): A dictionary representing the device's local data store.
            supervisor (object): The central supervisor managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal the receipt of a new script.
        self.script_received = Event()
        self.scripts = []
        # Event to signal the start of a new time-step.
        self.timepoint_done = Event()
        # The main control thread for this device's lifecycle.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Synchronization barrier, initialized later in `setup_devices`.
        self.barrier = None
        # A list of locks for fine-grained, location-based data access control.
        self.lock = []

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization primitives to all devices.

        Functional Utility: This method creates a single `ReusableBarrier` instance
        and a list of `Lock` objects that are then shared across all devices in
        the simulation. This ensures that all devices synchronize on the same
        barrier and use the same set of locks for data access.

        Args:
            devices (list[Device]): The list of all devices in the simulation.
        """
        # A reusable barrier for synchronizing all devices at the end of a time-step.
        # The ReusableBarrier class is expected to be in a separate file.
        barrier = ReusableBarrier.ReusableBarrier(len(devices))
        
        # Create a pool of locks. The index of the lock corresponds to a data 'location'.
        lock = []
        for _ in range(0, 100):
            newlock = Lock()
            lock.append(newlock)

        # Distribute the shared barrier and lock pool to all devices.
        for dev in devices:
            dev.barrier = barrier
            dev.lock = lock

    def assign_script(self, script, location):
        """Assigns a script to be executed in the next time-step."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A `None` script is used as a trigger to start the next time-step.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from the device's local sensor data store for a given location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates data for a given location in the device's local sensor data store."""
        self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()

class MyThread(Thread):
    """
    A worker thread to execute a single script, with location-based locking.

    Functional Utility: This thread performs a single unit of work, which involves
    acquiring a lock for a specific data location, gathering data from neighbors,
    running a script, and propagating the result. The lock prevents race conditions
    if multiple scripts attempt to access the same data location concurrently.
    """
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        # Acquire a lock specific to the data location to ensure exclusive access.
        self.device.lock[self.location].acquire()
        script_data = []

        # Block Logic: Data gathering phase under lock.
        # Pre-condition: Lock for `self.location` is held.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Script execution and data propagation.
        if script_data:
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
        
        # Bug: This line is outside the `if` block. If `script_data` is empty,
        # `result` will be unassigned, leading to a runtime error. It should
        # likely be indented to be part of the `if` block.
        self.device.set_data(self.location, result)
        
        # Release the lock for the data location.
        self.device.lock[self.location].release()

class DeviceThread(Thread):
    """The main control thread for a Device, managing its simulation lifecycle."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            # Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A `None` response from the supervisor signals simulation shutdown.
                break

            # Wait for the supervisor to signal the start of a time-step.
            self.device.timepoint_done.wait()
            # Logic Anomaly: Immediately setting the event after waiting could cause
            # a race condition where threads are continuously active rather than
            # waiting for a distinct signal for each time-step from the supervisor.
            self.device.timepoint_done.set()

            # Block Logic: Spawn and manage worker threads for the current time-step.
            # Pre-condition: Scripts for the time-step have been assigned.
            threads = []
            for (script, location) in self.device.scripts:
                thread_aux = MyThread(self.device, location, script, neighbours)
                threads.append(thread_aux)
            
            # Start and wait for all worker threads to complete.
            for auxiliar_thread in threads:
                auxiliar_thread.start()
            for auxiliar_thread in threads:
                auxiliar_thread.join()

            # Block Logic: End of time-step synchronization.
            self.device.timepoint_done.clear()
            # Wait at the barrier for all other devices to complete their time-step.
            self.device.barrier.wait()