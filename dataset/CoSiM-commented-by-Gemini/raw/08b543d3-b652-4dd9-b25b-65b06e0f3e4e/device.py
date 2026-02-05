"""
This module simulates a network of devices using a two-level threading model.
Each device has a main `DeviceThread` that, for each time step, spawns multiple
`ScriptWorker` threads. Each `ScriptWorker` is responsible for executing a single
script and synchronizes using a local barrier, while the main device threads
synchronize globally before proceeding to the next time step.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

class Device(object):
    """
    Represents a device in the network. It manages its own data and state,
    and orchestrates script execution through a main thread which in turn
    manages worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and its main control thread.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor (Supervisor): The central controller providing neighbor information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        # A dedicated lock for this device's data, acquired by external workers.
        self.lock = Lock()
        # The main control thread for this device.
        self.threads = [DeviceThread(self)]
        for thread in self.threads:
            thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared barrier for global synchronization
        among all main device threads. This is performed by device 0.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            # Invariant: The same barrier instance is shared across all devices.
            for device in devices:
                device.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed. A `None` script signals that all scripts
        for the current timepoint have been assigned.

        Args:
            script (Script): The script to be executed.
            location (str): The location associated with the script's execution.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signals the main DeviceThread to start processing the collected scripts.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data. This method is not thread-safe and relies on
        callers (e.g., ScriptWorker) to handle locking.

        Args:
            location (str): The location of the data to retrieve.
        
        Returns:
            The data if the location exists, otherwise None.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data. This method is not thread-safe and relies on
        callers (e.g., ScriptWorker) to handle locking.

        Args:
            location (str): The location of the data to set.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main control thread to complete."""
        for thread in self.threads:
            thread.join()

class ScriptWorker(Thread):
    """
    A one-shot worker thread responsible for executing a single script.
    It gathers data, runs the script, distributes the result, and then
    synchronizes with its parent DeviceThread via a local barrier.
    """

    def __init__(self, data):
        """
        Initializes the worker.

        Args:
            data (dict): A dictionary containing all necessary context, including the
                         parent device, script, location, neighbors, and the worker barrier.
        """
        Thread.__init__(self)
        self.device = data['device']
        self.script = data['script']
        self.location = data['location']
        self.neighbours = data['neighbours']
        self.barrier = data['barrier']

    def run(self):
        """The main logic for the worker thread."""
        script_data = []

        # Gather data from neighbors and the parent device.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Pre-condition: Only run the script if there is data to process.
        if script_data:
            result = self.script.run(script_data)

            # Distribute the result by acquiring a lock on each target device.
            for device in self.neighbours:
                device.lock.acquire()
                device.set_data(self.location, result)
                device.lock.release()

            # Update the local device as well.
            self.device.lock.acquire()
            self.device.set_data(self.location, result)
            self.device.lock.release()

        # Invariant: Signal completion to the parent DeviceThread by waiting on the worker barrier.
        self.barrier.wait()


class DeviceThread(Thread):
    """
    The main control thread for a device. It orchestrates the execution of scripts
    for each timepoint by spawning and managing a new set of ScriptWorkers.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main control loop.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # Wait for the supervisor to signal that script assignment is complete.
            self.device.timepoint_done.wait()

            no_scripts = len(self.device.scripts)
            
            # Create a local barrier for the workers of this timepoint.
            # It synchronizes all workers plus the main DeviceThread itself (+1).
            worker_barrier = ReusableBarrierCond(no_scripts + 1)
            workers = []
            
            # Spawn a ScriptWorker for each assigned script.
            for (script, location) in self.device.scripts:
                workers.append(
                ScriptWorker(
                {
                'device' : self.device,
                'script' : script,
                'location' : location,
                'neighbours' : neighbours,
                'barrier' : worker_barrier
                }
                ))

            for worker in workers:
                worker.start()

            # Wait for all spawned ScriptWorkers to complete their tasks.
            worker_barrier.wait()

            # Clean up all completed worker threads.
            for worker in workers:
                worker.join()

            self.device.timepoint_done.clear()
            # Invariant: Wait at the global barrier to synchronize with other Devices
            # before starting the next timepoint.
            self.device.barrier.wait()