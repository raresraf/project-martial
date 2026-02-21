"""A distributed device simulation with non-deterministic setup.

This module implements a simulation of devices that operate in synchronized
time steps. It uses a custom `ReusableBarrier` for synchronization and a model
where each device's main thread spawns new worker threads for each script.

WARNING: The initialization of shared resources (barriers and locks) is
handled via a decentralized, "gossip-like" protocol that is highly
susceptible to race conditions, making the setup non-deterministic and fragile.
"""

from threading import Event, Semaphore, Lock, Thread



"""
Defines a distributed device simulation using a thread-per-task model.

Note: This file is named `device.py` but contains the full simulation logic.

The architecture consists of:
- `ReusableBarrier`: A custom two-phase barrier for thread synchronization.
- `Device`: Represents a node, holding data and managing its state.
- `DeviceThread`: The main control loop for a device, which spawns a new
  `ScriptWorker` thread for each task in a time step.
- `ScriptWorker`: A short-lived thread that executes a single script.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """A custom, reusable barrier implemented with semaphores."""
    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes the calling thread to wait at the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier wait."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads  # Reset for reuse.
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
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.devices = []
        self.barrier = None
        self.workers = []
        # Initialize a dictionary for location-specific locks.
        self.loc_barrier = {key: None for key in range(60)}
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the shared barrier for the device group.

        Note: This implementation contains a race condition. The first device
        to check `self.barrier` creates a new barrier and distributes it.
        Correct behavior depends on execution order.
        """
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """Assigns a script to the device.

        Note: This method contains complex, racy logic for initializing
        location-specific locks. It tries to find an existing lock from other
        devices before creating a new one, which can be unpredictable.
        """
        if script is not None:
            self.scripts.append((script, location))
            if self.loc_barrier[location] is None:
                for device in self.devices:
                    if device.loc_barrier[location] is not None:
                        self.loc_barrier[location] = device.loc_barrier[location]
                        break
            if self.loc_barrier[location] is None:
                self.loc_barrier[location] = Lock()
            self.script_received.set()
        else:
            # A `None` script signals that all scripts for the time step are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets data from a specific sensor location (not thread-safe)."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data at a specific sensor location (not thread-safe)."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class ScriptWorker(Thread):
    """A short-lived worker thread that executes a single script."""
    def __init__(self, device, neighbours, script, location):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """Executes the script logic within a location-specific lock."""
        with self.device.loc_barrier[self.location]:
            # Gather data from neighbors and self.
            script_data = []
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Run script and propagate results if data was found.
            if script_data:
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """The main control thread that orchestrates the device's lifecycle."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main time step loop for the device.

        In each simulation step, it waits for scripts, spawns a new worker thread
        for each script, waits for them to complete, and then synchronizes with
        all other devices at a global barrier.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break  # Supervisor signals shutdown.

            # Wait until all scripts for the current time step are assigned.
            self.device.timepoint_done.wait()

            # Create a new worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                thread = ScriptWorker(self.device, neighbours, script, location)
                self.device.workers.append(thread)

            # Start and then wait for all workers for this step to complete.
            for worker in self.device.workers:
                worker.start()
            for worker in self.device.workers:
                worker.join()

            # Reset state for the next time step.
            self.device.workers = []
            
            # Prepare for the next step and synchronize with all other devices.
            self.device.timepoint_done.clear()
            
            # Synchronize with all other devices before the next step.
            self.device.barrier.wait()
