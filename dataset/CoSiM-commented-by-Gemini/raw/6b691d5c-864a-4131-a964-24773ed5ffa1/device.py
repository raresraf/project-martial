"""
This module provides a distributed device simulation framework utilizing a
"thread-per-script" concurrency model.

The architecture is defined by several key components:
- A custom, two-phase `ReusableBarrier` for synchronizing the main device
  threads at the end of each simulation step.
- A `Device` class that manages its state and a main `DeviceThread`.
- A `DeviceThread` that orchestrates the simulation step by spawning a
  dedicated `ScriptWorker` thread for each assigned script.
- A complex and non-atomic mechanism for lazy initialization and sharing of
  location-based locks.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    A custom, two-phase reusable barrier implemented using Semaphores.

    This implementation uses a list of size 1 for its counters to simulate
    pass-by-reference semantics when calling the internal `phase` method. This
    is an unconventional but functional approach in Python.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters are stored in a list to be passed by reference.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all participating threads have called wait."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the two-phase barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive resets the counter and releases all
                # waiting threads for this phase.
                n_threads = self.num_threads
                while n_threads > 0:
                    threads_sem.release()
                    n_threads -= 1
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Represents a device in the simulation.

    Manages its own sensor data, script assignments, and the main thread
    of execution. It coordinates with other devices using a shared barrier and
    a dictionary of location-based locks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
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
        # Pre-initializes a dictionary for location-based locks.
        keys = range(60)
        self.loc_barrier = {key: None for key in keys}
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes the main barrier to all devices.

        This method ensures all devices share a single barrier instance,
        which is created by the first device to call this method.
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
        """
        Assigns a script and handles the lazy initialization of its location lock.

        If a script is provided, it's added to the list. This method also contains
        a complex, non-atomic logic to find or create a lock for the script's
        location, which is prone to race conditions.
        """
        if script is not None:
            self.scripts.append((script, location))
            # --- Lazy lock initialization (potential race condition) ---
            # It first checks if this device already has a lock for the location.
            if self.loc_barrier[location] is None:
                # If not, it iterates through ALL devices to see if one has created it.
                for device in self.devices:
                    if device.loc_barrier[location] is not None:
                        self.loc_barrier[location] = device.loc_barrier[location]
                        break
            # If no lock was found on any device, it creates a new one.
            # This whole block is not atomic.
            if self.loc_barrier[location] is None:
                self.loc_barrier[location] = Lock()
            self.script_received.set()
        else:
            # A None script signals that all assignments for this step are done.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class ScriptWorker(Thread):
    """
    A worker thread that executes a single script.
    A new instance is created for each script in each time step.
    """
    def __init__(self, device, neighbours, script, location):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script logic: locks the location, gathers data, runs the
        script, updates data, and releases the lock.
        """
        with self.device.loc_barrier[self.location]:
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

            # If any data was found, run the script and update the values.
            if script_data:
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    The main control thread for a device.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main simulation loop. In each step, it spawns, starts, and joins
        worker threads for all assigned scripts.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Shutdown signal from the supervisor.
                break

            # Wait for the supervisor to finish assigning scripts for this step.
            self.device.timepoint_done.wait()

            # Create a new worker thread for each script.
            for (script, location) in self.device.scripts:
                thread = ScriptWorker(self.device, neighbours, script, location)
                self.device.workers.append(thread)

            # Start all worker threads.
            for worker in self.device.workers:
                worker.start()

            # Wait for all worker threads to complete.
            for worker in self.device.workers:
                worker.join()

            # Clean up for the next time step.
            self.device.workers = []
            self.device.timepoint_done.clear()

            # Synchronize with all other devices before the next step.
            self.device.barrier.wait()
