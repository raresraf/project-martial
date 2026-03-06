"""
This module provides a framework for simulating a network of interconnected devices,
each running in its own thread.

The simulation uses a discrete-time model. In each time step, devices execute
assigned scripts in parallel `ScriptThread`s. Synchronization is managed through
a combination of a reusable barrier (for time steps) and a mixed locking strategy:
location-specific locks are used during data aggregation, while a single global
lock is used to serialize all data write-backs across the entire system.
"""


from threading import Event, Thread, Semaphore, Lock

class ReusableBarrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This implementation uses a two-phase protocol with semaphores to ensure that
    threads from one synchronization cycle do not interleave with threads from
    the next, allowing the barrier to be used multiple times.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Using a list of one element to create a mutable integer that can be
        # modified across method calls within the instance.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the two-phase barrier protocol.

        Args:
            count_threads (list): The counter for the current phase.
            threads_sem (Semaphore): The semaphore for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # The last thread to arrive resets the counter for the next cycle
            # and releases all waiting threads.
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class ScriptThread(Thread):
    """A short-lived thread responsible for executing a single script."""
    def __init__(self, script, location, device, neighbours):
        """Initializes the ScriptThread."""
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script logic, including data gathering and result writing.
        """
        # Acquire the specific lock for the location this script is operating on.
        self.device.hash_locatie[self.location].acquire()

        # --- Data Aggregation Phase ---
        script_data = []
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # --- Computation and Write-back Phase ---
        if script_data:
            result = self.script.run(script_data)

            # Acquire the single global lock to serialize all writes to sensor_data
            # across the entire simulation. This is a major bottleneck.
            self.device.lock.acquire()
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
            self.device.lock.release()

        # Release the location-specific lock.
        self.device.hash_locatie[self.location].release()

class Device(object):
    """
    Represents a device, holding its state and shared synchronization objects.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        # Shared resources are initialized to None until setup_devices is called.
        self.barrier = None
        self.lock = None # The global write lock.
        self.hash_locatie = None # Dictionary of location-specific locks.
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources to all devices.

        This should be called on one master device. It creates a shared barrier,
        a global write lock, and a dictionary of location-specific locks.
        """
        # Use the first device in the list as the master for setup.
        if self.device_id == devices[0].device_id:
            barrier = ReusableBarrier(len(devices))
            my_lock = Lock() # The single global lock for all write operations.
            hash_locatie = {i: Lock() for i in range(101)}

            # Distribute the shared objects to all devices in the simulation.
            for device in devices:
                device.barrier = barrier
                device.lock = my_lock
                device.hash_locatie = hash_locatie

    def assign_script(self, script, location):
        """
        Assigns a script to the device for the current timepoint.

        A 'None' script signals that all scripts have been assigned.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates sensor data. Note: This method is NOT thread-safe by itself.
        It relies on the calling thread (ScriptThread) to hold the global lock.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to complete."""
        self.thread.join()

class DeviceThread(Thread):
    """The main control thread for a Device instance."""

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop for the device.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            # A None value for neighbours is the signal to terminate.
            if neighbours is None:
                break

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()

            script_list = []
            # Create a new thread for each assigned script.
            for (script, location) in self.device.scripts:
                script_list.append(ScriptThread(script, location, self.device, neighbours))

            for thread in script_list:
                thread.start()

            # Wait for all script threads to complete their execution.
            for thread in script_list:
                thread.join()

            self.device.scripts = [] # Clear scripts for the next timepoint.
            self.device.timepoint_done.clear()
            
            # Synchronize with all other devices before starting the next timepoint.
            self.device.barrier.wait()
