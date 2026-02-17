"""
Models a distributed network of devices using a fork-join parallelism model.

Each device in this simulation operates in synchronized time steps. For each
timepoint, the device's main thread spawns a new set of worker threads to execute
assigned scripts. This contrasts with a persistent worker pool model. Synchronization
is handled by a two-level barrier system: a local barrier for a device's own
worker threads and a global barrier for all devices in the network.
"""

from threading import Event, Thread, Condition

class Device(object):
    """
    Represents a single device in the network.

    It manages its own data, assigned scripts, and the main control thread.
    Device 0 acts as the master, responsible for global synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and starts its main control thread.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor (object): The supervisor object for network communication.
        """
        self.zero = 0
        self.length = self.zero
        self.device_id = device_id
        self.devices = None
        self.barrier = None
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def setup_devices(self, devices):
        """
        Initializes the global barrier for network-wide synchronization.

        The device with ID 0 creates the barrier and distributes it to all other
        devices.

        Args:
            devices (list): A list of all device objects in the network.
        """
        if self.device_id == self.zero:
            self.length = len(devices)
            self.barrier = BarrierCheck(self.length)
            for dev in devices:
                dev.barrier = self.barrier

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id
   
    def assign_script(self, script, location):
        """
        Assigns a script to the device. A 'None' script signals the end of
        assignments for the current timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data for a given location if it exists."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data for a given location if it exists."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class BarrierCheck(object):
    """
    A custom, non-standard implementation of a reusable barrier.

    Intended to block a set of threads until all have arrived. The implementation
    is complex and differs from standard two-phase barrier patterns.
    """
    def __init__(self, threads):
        self.zero = 0
        self.decrement = 1
        self.check = self.zero
        self.numberofthreads = threads
        self.countthreads = threads
        self.condition = Condition()

    def wait(self):
        self.condition.acquire()

        self.countthreads -= self.decrement
        self.check = self.zero

        # If this is not the last thread to arrive, wait.
        if max(self.countthreads, self.zero) > self.zero:
            self.condition.wait()
            self.check = 1

        # Logic to reset the barrier and notify waiting threads.
        if min(self.check, self.zero) == self.zero:
            self.countthreads = self.numberofthreads
            self.condition.notify_all()

        self.condition.release()


class Scripts(Thread):
    """
    A short-lived worker thread responsible for executing one or more scripts.
    """

    def __init__(self, device, scripts, neighbours):
        """Initializes the script execution thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.ans = None
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):
        """Executes the assigned scripts."""
        for (script, location) in self.scripts:
            script_data = []

            # Aggregate data from the local device and its neighbors.
            if self.device.get_data(location) is not None:
                script_data.append(self.device.get_data(location))

            for device in self.neighbours:
                if device.get_data(location) is not None:
                    # Avoid duplicating data from the local device.
                    if device.get_data(location) != self.device.get_data(location):
                        script_data.append(device.get_data(location))

            # Pre-condition: Only run the script if there is data to process.
            if script_data:
                ans = script.run(script_data)

                # Functional Utility: This logic suggests a maximization algorithm,
                # only updating the value if the new result is greater.
                if ans > self.device.get_data(location):
                    self.device.set_data(location, ans)

                for device in self.neighbours:
                    if ans > device.get_data(location):
                        device.set_data(location, ans)

        # Invariant: After executing its script(s), the worker waits at a local
        # barrier, synchronizing with other workers of the same device.
        self.device.thread.barrier.wait()

class DeviceThread(Thread):
    """
    The main control thread for a device, using a fork-join model.
    """

    def __init__(self, device):
        """Initializes the main device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.zero = 0
        self.increment = 1
        self.maxthreads = 8
        self.currentthread = self.zero
        self.currentscript = self.zero
        self.length = self.zero
        self.threadslist = []
        self.device = device
        self.barrier = None

    def run(self):
        """Main loop that orchestrates work for each timepoint."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Shutdown signal received from the supervisor.
                break

            # Block Logic: Waits for the supervisor to assign all scripts for the timepoint.
            self.device.timepoint_done.wait()
            self.currentthread = self.currentscript = self.zero

            # Block Logic: Creates a new set of worker threads for the assigned scripts.
            # This implements a fork-join pattern for each timepoint.
            for script in self.device.scripts:
                # This logic for grouping scripts appears incomplete/buggy.
                if min(self.currentscript, self.maxthreads) == self.maxthreads:
                    self.currentscript = self.zero
                    # The '.add' method does not exist for lists.
                    # self.threadslist[self.currentthread].scripts.add(script)
                else:
                    # Creates one new thread per script.
                    self.threadslist.append(
                        Scripts(self.device, [script], neighbours))

                self.currentthread += self.increment
                self.currentscript += self.increment

            self.barrier_and_threads()
            self.clear()

    def barrier_and_threads(self):
        """
        Manages the local fork-join execution of worker threads.
        """
        self.length = len(self.threadslist)
        # A local barrier is created just for this device's worker threads.
        self.barrier = BarrierCheck(self.length)

        # Fork: Start all worker threads.
        for thread in self.threadslist:
            thread.start()

        # Join: Wait for all worker threads to complete.
        for thread in self.threadslist:
            thread.join()

    def clear(self):
        """
        Cleans up after a timepoint and synchronizes with the global barrier.
        """
        self.device.timepoint_done.clear()
        # Waits at the global barrier, ensuring all devices have finished the timepoint.
        self.device.barrier.wait()
        self.threadslist = []