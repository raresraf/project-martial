"""
This module presents a fourth distinct implementation of a distributed device
simulation. This version features an explicit master/worker role for device setup
and a thread-per-task execution model, where a new thread is spawned for each
script in every time step.
"""
from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a device that uses a master-worker pattern for setup and spawns
    a new thread for each individual script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): The device's local sensor data.
            supervisor: The central supervisor managing the simulation.
        """
        self.are_locks_ready = Event() # Signals when the master has created the locks.
        self.master_id = None # ID of the master device responsible for setup.
        self.is_master = True # Flag indicating if this device is the master.
        self.barrier = None # Shared barrier for all devices.
        self.stored_devices = [] # List of all devices in the simulation.
        self.data_lock = [None] * 100 # A fixed-size list to hold location locks.
        self.master_barrier = Event() # Signals when the master has finished setup.
        self.lock = Lock() # A separate, internal lock for the set_data method.
        self.started_threads = [] # A temporary list of executor threads for a timepoint.
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self) # The main control thread.
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the simulation environment, with one device acting as a master.

        The first device to execute this section becomes the master, responsible for
        creating and distributing the shared synchronization barrier and locks.
        """
        # --- Master Election Logic ---
        # This is a bit racy; the first device to check determines the master.
        for device in devices:
            if device is not None and device.master_id is not None:
                self.master_id = device.master_id
                self.is_master = False
                break

        if self.is_master is True:
            # --- Master Device Logic ---
            self.barrier = ReusableBarrierSem(len(devices))
            self.master_id = self.device_id
            # Create a fixed-size list of locks. Assumes locations are integers from 0-99.
            for i in range(100):
                self.data_lock[i] = Lock()
            self.are_locks_ready.set()
            self.master_barrier.set() # Signal that master setup is complete.
            for device in devices:
                if device is not None:
                    device.barrier = self.barrier
                    self.stored_devices.append(device)
        else:
            # --- Worker Device Logic ---
            # Find the master and wait for it to finish setup.
            for device in devices:
                if device is not None:
                    if device.device_id == self.master_id:
                        device.master_barrier.wait()
                        if self.barrier is None:
                            self.barrier = device.barrier
                    self.stored_devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script to the device.

        Note: This method contains an inefficient design where it repeatedly waits for
        and copies the lock list from the master on every script assignment.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Inefficiently find the master and wait for its locks on every assignment.
            for device in self.stored_devices:
                if device.device_id == self.master_id:
                    device.are_locks_ready.wait()
            # Inefficiently copy the lock list reference on every assignment.
            for device in self.stored_devices:
                if device.device_id == self.master_id:
                    self.data_lock = device.data_lock
            self.script_received.set()
        else:
            # A None script signals the start of the timepoint processing.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data.

        Warning: This method uses a different lock (`self.lock`) than the one used
        by ExecutorThreads (`self.data_lock`), which could lead to race conditions.
        """
        self.lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.lock.release()

    def shutdown(self):
        """Shuts down the device."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread. It uses a thread-per-task model, creating a new
    ExecutorThread for each script in every timepoint.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        """The main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for the signal that script assignment is done for this timepoint.
            self.device.timepoint_done.wait()
            
            # --- Thread-per-Task Spawning ---
            # For each script, create and start a new ExecutorThread.
            for (script, location) in self.device.scripts:
                executor = ExecutorThread(self.device, script, neighbours, location)
                self.device.started_threads.append(executor)
                executor.start()

            # Wait for all spawned executor threads for this timepoint to complete.
            for executor in self.device.started_threads:
                executor.join()
            
            del self.device.started_threads[:]
            self.device.timepoint_done.clear()
            # Synchronize with all other devices at the main barrier.
            self.device.barrier.wait()


class ExecutorThread(Thread):
    """
    A temporary worker thread created to execute a single script for one timepoint.
    """

    def __init__(self, device, script, neighbours, location):
        Thread.__init__(self, name="Executor Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.neighbours = neighbours
        self.location = location

    def run(self):
        """Acquires a lock, gathers data, executes the script, and updates data."""
        # Acquire the lock for the specific data location.
        self.device.data_lock[self.location].acquire()

        if self.neighbours is None:
            return

        script_data = []
        
        # Gather data from neighbors.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Gather data from the local device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Execute the script.
            result = self.script.run(script_data)
            
            # Update data on all involved devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        self.device.data_lock[self.location].release()