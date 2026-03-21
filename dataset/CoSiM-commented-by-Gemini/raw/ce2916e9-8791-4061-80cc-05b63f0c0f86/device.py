"""
Models a network of communicating devices that execute computational scripts
in synchronized, discrete time steps.

This module defines a `Device` that acts as a node in a distributed system.
Synchronization is achieved using a custom semaphore-based reusable barrier and
a globally shared dictionary of locks for data locations, both of which are
initialized and distributed by a single coordinator device (Device 0).
"""

from threading import Event, Semaphore, Lock, Thread
from Queue import Queue

class Device(object):
    """
    Represents a single device (node) in a simulated distributed network.

    Each device holds sensor data and a list of scripts to execute for a given
    time step. It relies on a main control thread to manage the simulation flow.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding the initial sensor data,
                                keyed by location.
            supervisor: An external entity responsible for managing the
                        network topology and assigning scripts.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.no_th = 8  # Specifies a fixed number of worker threads.

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization primitives.

        Device with device_id 0 acts as the coordinator. It creates a single
        shared barrier and a global dictionary of locks for all unique data
        locations across all devices, then distributes references to these
        objects to every other device.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            barrier = ReusableBarrierSem(len(devices))
            lock_for_loct = {}
            for device in devices:
                device.barrier = barrier
                for location in device.sensor_data:
                    if location not in lock_for_loct:
                        lock_for_loct[location] = Lock()
                device.lock_for_loct = lock_for_loct

    def assign_script(self, script, location):
        """
        Assigns a script to be executed for the current time step.

        A 'None' script is used as a signal from the supervisor that all
        scripts for the current time step have been assigned.

        Args:
            script: The script object to be executed.
            location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_received.set()

    def get_data(self, location):
        """
        Retrieves data for a given location.

        Note: This method is not thread-safe by itself. Synchronization is
        expected to be handled by the calling worker thread via the shared
        location lock.

        Args:
            location: The location from which to retrieve data.

        Returns:
            The data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data, source=None):
        """
        Sets the data for a given location.

        Note: This method is not thread-safe by itself. Synchronization is
        expected to be handled by the calling worker thread via the shared
        location lock.

        Args:
            location: The location at which to set data.
            data: The new data value to be set.
            source: An optional argument, currently unused.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    Orchestrates the simulation by waiting for scripts, spawning a new pool of
    worker threads for each time step, and synchronizing with other devices.
    """

    def __init__(self, device):
        """
        Initializes the main device thread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = []
        self.neighbours = []

    def run(self):
        """
        The main simulation loop for the device.

        For each time step, this loop:
        1. Waits for the supervisor to signal that scripts have been assigned.
        2. Creates a new queue and populates it with the assigned scripts.
        3. Spawns a new pool of `SolveScript` worker threads to process the queue.
        4. Waits for all scripts in the queue to be completed.
        5. Waits at a global barrier to synchronize with all other devices.
        The loop terminates if the supervisor returns no neighbors.
        """
        while True:
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break

            self.device.scripts_received.wait()
            self.device.scripts_received.clear()

            self.queue = Queue()
            for script in self.device.scripts:
                self.queue.put_nowait(script)

            # Inefficiently creates a new set of worker threads for each time step.
            for _ in range(self.device.no_th):
                SolveScript(self.device, self.neighbours, self.queue).start()
            
            self.queue.join()
            
            self.device.barrier.wait()

class SolveScript(Thread):
    """
    A worker thread that executes a single script from the queue.
    """

    def __init__(self, device, neighbours, queue):
        """
        Initializes the worker thread.

        Args:
            device (Device): The parent device.
            neighbours (list): A list of neighboring Device objects.
            queue (Queue): The shared work queue for the current time step.
        """
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.queue = queue

    def run(self):
        """
        Executes a script from the queue.

        The worker performs a distributed read-compute-write cycle by:
        1. Acquiring the global lock for the script's target location.
        2. Gathering data from its own device and all neighbors.
        3. Executing the script on the aggregated data.
        4. Writing the result back to its own device and all neighbors.
        5. Releasing the lock.
        The broad `try...except` block dangerously suppresses all errors.
        """
        try:
            # This loop is structured unusually. The `for` loop iterates over all
            # scripts, but the `queue.get()` inside means each worker will likely
            # process multiple, unrelated scripts.
            for (script, location) in self.device.scripts:
                (script, location) = self.queue.get(False)
                
                # Acquire the global lock for the target location.
                self.device.lock_for_loct[location].acquire()

                script_data = []
                
                # Read phase: Collect data from neighbors.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Read phase: Collect data from self.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Compute and Write phase.
                if script_data != []:
                    result = script.run(script_data)
                    
                    # Distribute result to neighbors.
                    for device in self.neighbours:
                        device.set_data(location, result)
                    
                    # Set result on self.
                    self.device.set_data(location, result)

                # Release the global lock.
                self.device.lock_for_loct[location].release()
                
                self.queue.task_done()
        except:
            # This broad except block will catch and silently ignore any error
            # during script execution, which can hide bugs.
            pass

class ReusableBarrierSem():
    """
    A classic two-phase reusable barrier implemented with semaphores.

    This allows a fixed number of threads to synchronize at a point, wait for
    all participating threads to arrive, and then proceed. It can be reused
    for subsequent synchronization points.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)  # For the first phase
        self.threads_sem2 = Semaphore(0)  # For the second phase (reuse)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads arrive."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread to arrive releases all other threads.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase, allowing the barrier to be safely reused."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # The last thread to arrive resets the barrier for the next use.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()
