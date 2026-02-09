"""
A simulation framework for a network of devices using a global lock.

This module defines a `Device` class that operates in a synchronized,
time-stepped simulation. It features a custom barrier implementation using a
`Condition` variable and employs a single, global lock to serialize all
data-processing scripts, ensuring only one script runs at a time across the
entire system.
"""

from threading import Event, Thread, Condition, Lock

class ReusableBarrier(object):
    """
    Provides a barrier to synchronize a fixed number of threads.

    This implementation uses a `threading.Condition` to block threads. Threads
    that call `wait()` will block until the last thread arrives. The last thread
    then notifies all waiting threads, allowing them to proceed.
    Note: This is a single-phase barrier and may not be safely reusable under
    certain race conditions (the "lost wake-up" problem).
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
    def wait(self):
        """
        Blocks the calling thread until all participating threads have arrived.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # Invariant: If this is the last thread, it wakes up all other threads
        # and resets the counter for the next use.
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a single device in the simulated network.

    Each device manages its own data and runs a control thread. It participates
    in a coordinated setup where a single global lock and a shared barrier

    are distributed for system-wide use.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's internal sensor data.
            supervisor: The central supervisor managing the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barr = None
        self.lock = None
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources for all devices.

        The first device in the list acts as a master, creating a shared
        `ReusableBarrier` and a single, global `Lock`. These resources are
        then assigned to all other devices in the simulation.
        
        Args:
            devices (list): A list of all Device objects.
        """
        # Block Logic: The master device creates and distributes the shared objects.
        if devices[0].barr is None and devices[0].device_id == self.device_id:
            bariera = ReusableBarrier(len(devices))
            # A single lock is created and shared across all devices.
            lock = Lock()
            for i in devices:
                i.barr = bariera
            for j in devices:
                j.lock = lock
    def assign_script(self, script, location):
        """
        Receives a script from the supervisor for the current timepoint.

        Args:
            script: The script object to execute.
            location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating serialized execution.

    This thread manages the device's operation within synchronized timepoints.
    It processes all its assigned scripts sequentially, using a global lock that
    ensures only one script is running across the entire system at any moment.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop, organized into discrete timepoints."""
        while True:
            # Get neighbours for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to finish assigning scripts.
            self.device.timepoint_done.wait()

            # Block Logic: Sequentially process each assigned script.
            for (script, location) in self.device.scripts:
                # Pre-condition: Acquire the single global lock. This serializes
                # script execution across all devices in the system.
                self.device.lock.acquire()
                script_data = []
                
                # Aggregate data from neighbours and self.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Invariant: Script runs only if there is data to process.
                if script_data != []:
                    result = script.run(script_data)

                    # Broadcast the result to all participants.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                self.device.lock.release()

            self.device.timepoint_done.clear()
            # Invariant: All devices must wait at the barrier, ensuring they all
            # complete the current timepoint before any can proceed.
            self.device.barr.wait()