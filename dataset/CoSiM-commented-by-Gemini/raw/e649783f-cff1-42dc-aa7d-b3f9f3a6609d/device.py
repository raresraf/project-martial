"""
This module defines the classes for a simulated distributed device network.

The architecture uses a custom ReusableBarrier for synchronization and dynamically
creates worker threads (`Slave`) for each computational task in every cycle.
- ReusableBarrier: A custom two-phase barrier to synchronize threads.
- Device: Represents a node, holding data and managing its control thread.
- SupervisorThread: The main control thread for a Device, which spawns
  short-lived Slave threads for computation.
- Slave: A worker thread created on-the-fly to execute a single script.
"""
from threading import Event, Thread, Semaphore, Lock

class ReusableBarrier(object):
    """A custom, reusable barrier for synchronizing a fixed number of threads.

    This barrier is implemented using a two-phase protocol, allowing it to be
    used repeatedly within a loop. It ensures that all threads wait at the
    barrier until the required number of threads have arrived.
    """
    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        # Counters for each phase, stored in a list to be mutable.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        # Semaphores to block and release threads for each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier. Blocks until all threads arrive."""
        # The barrier consists of two phases to prevent race conditions on reuse.
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes a single phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            # The last thread to arrive resets the counter and releases all waiting threads.
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        # All threads wait here until the last thread releases the semaphore.
        threads_sem.acquire()

class Device(object):
    """Represents a device node in the distributed simulation."""
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main control thread."""
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []  # Stores (script, location) tuples for the current cycle.
        self.threads = []  # Stores dynamically created Slave threads.
        self.devices = []
        self.semafor = Semaphore(0)
        # Event that is set when all scripts for a timepoint have been assigned.
        self.timepoint_done = Event()
        self.thread = SupervisorThread(self)
        self.thread.start()
        self.num_scr = 8  # Number of slave threads to spawn in a batch.
        self.lock = [None] * 100  # Shared list of location-based locks.

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes shared resources for the device network.

        The first device to call this method creates and shares the global barrier
        and location locks with all other devices.
        """
        # The first device to enter initializes the shared barrier and locks.
        if self.barrier is None:
            self.barrier = ReusableBarrier(len(devices))
            for dev in devices:
                dev.lock = self.lock
                if dev.barrier is None:
                    dev.barrier = self.barrier

        for dev in devices:
            if dev is not None:
                self.devices.append(dev)

    def assign_script(self, script, location):
        """Assigns a script to be executed in the current cycle.

        Args:
            script (Script): The script to run. If None, it signals the end
                             of script assignment for the current cycle.
            location (int): The location key for the data.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Lazily initialize the lock for this location if not already done.
            if self.lock[location] is None:
                # Check if another device has already created the lock.
                for device in self.devices:
                    if device.lock[location] is not None:
                        self.lock[location] = device.lock[location]
                        break
                # If not, create a new one.
                if self.lock[location] is None:
                    self.lock[location] = Lock()
            self.script_received.set()
        else:
            # A None script signals that the device has all its tasks for this cycle.
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets sensor data for a specific location. Not thread-safe by itself."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data for a specific location. Not thread-safe by itself."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main control thread to shut down the device."""
        self.thread.join()


class SupervisorThread(Thread):
    """The main control thread for a Device instance.

    It manages the device's lifecycle, spawning and managing short-lived
    worker (`Slave`) threads to execute scripts for each cycle.
    """
    def __init__(self, device):
        """Initializes the main control thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop of the device."""
        while True:
            # Get neighbor information from the central supervisor.
            neighb = self.device.supervisor.get_neighbours()
            if neighb is None:
                break  # End of simulation.
            
            # Wait for the signal that all scripts for this cycle have been assigned.
            self.device.timepoint_done.wait()
            
            i = 0
            while i < len(self.device.scripts):
                # Process scripts in batches.
                for _ in range(0, self.device.num_scr):
                    pair = self.device.scripts[i]
                    # Dynamically create a new worker thread for each script.
                    new_thread = Slave(self.device, pair[1], neighb, pair[0])
                    self.device.threads.append(new_thread)
                    new_thread.start()
                    i = i + 1
                    if i >= len(self.device.scripts):
                        break
                # Wait for the current batch of worker threads to complete.
                for thread in self.device.threads:
                    thread.join()
                self.device.threads = [] # Clear the batch list.

            # Reset for the next cycle.
            self.device.scripts = []
            self.device.timepoint_done.clear()
            
            # Wait at the global barrier for all other devices to finish their cycle.
            self.device.barrier.wait()

class Slave(Thread):
    """A short-lived worker thread created to execute a single script."""

    def __init__(self, device, location, neighbours, script):
        """Initializes the worker with its specific task."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script


    def run(self):
        """Executes the script on data from a specific location."""
        # Acquire the shared lock for this location to ensure data consistency.
        self.device.lock[self.location].acquire()
        script_data = []
        
        # Gather data for the given location from all neighbors.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Gather data from the parent device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)
            
        if script_data:
            # Execute the script on the collected data.
            result = self.script.run(script_data)
            # Write the result back to all neighbors and the parent device.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
        
        # Release the lock for the location.
        self.device.lock[self.location].release()
