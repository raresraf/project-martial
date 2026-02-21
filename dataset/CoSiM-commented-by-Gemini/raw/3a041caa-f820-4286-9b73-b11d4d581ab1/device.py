


"""
Defines a distributed device simulation using a thread-per-task model.

This file contains the classes to simulate a network of devices that
execute scripts in synchronized time steps. The architecture includes:
- `ReusableBarrier`: A custom two-phase barrier for thread synchronization.
- `Device`: Represents a node, holding data and managing its state.
- `DeviceThread`: The main control loop for a device that spawns executor
  threads.
- `NewThread`: A short-lived executor thread created for each script task.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier():
    """A custom implementation of a reusable, two-phase barrier."""

    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier.

        This implementation uses two phases (and two semaphores) to ensure
        that the barrier can be safely reused across multiple synchronization
        points without race conditions.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier wait."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all other waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """Represents a single device in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)
        self.barrier = None
        self.list_thread = []  # Holds executor threads for the current time step.
        self.thread.start()
        self.location_lock = [None] * 100  # Pool of location-specific locks.

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared synchronization objects for the device group.

        If this device doesn't have a barrier yet, it creates one and
        distributes it to all other devices in the group that also don't
        have one. This logic is sensitive to the order of execution.
        """
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier
        # Store a reference to all devices in the group.
        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """Assigns a script to be executed.

        This method also contains complex logic to ensure a lock for the specified
        location exists. It attempts to find and copy a lock from another device
        before creating a new one. This can be racy and unpredictable.
        """
        if script is not None:
            self.scripts.append((script, location))
            # If no lock exists for this location, try to find one from peers.
            if self.location_lock[location] is None:
                found_lock = False
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        found_lock = True
                        break
                # If no peer has a lock, create a new one.
                if not found_lock:
                    self.location_lock[location] = Lock()
            self.script_received.set()
        else:
            # A `None` script signals that all scripts for the time step are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data at a specific sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class NewThread(Thread):
    """A short-lived executor thread for running a single script."""
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """The core execution logic for a task."""
        # Acquire the specific lock for the data location.
        with self.device.location_lock[self.location]:
            script_data = []
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            # Gather data from self.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Run the script and propagate results.
            if script_data:
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """The main control thread that orchestrates a device's lifecycle."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main time step loop.

        Waits for scripts, spawns executor threads, joins them, and then
        synchronizes at a global barrier.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break  # Supervisor signals shutdown.

            # Wait until all scripts for the current time step have been assigned.
            self.device.timepoint_done.wait()

            # For each script, create a new executor thread.
            for (script, location) in self.device.scripts:
                thread = NewThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            # Start all executor threads for this time step.
            for thread_elem in self.device.list_thread:
                thread_elem.start()
            # Wait for all of them to complete.
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            
            self.device.list_thread = []

            # Reset for the next time step.
            self.device.timepoint_done.clear()
            # Wait at the barrier for all other devices to finish their time step.
            self.device.barrier.wait()