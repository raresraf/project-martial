"""
This module defines a device simulation framework that includes its own implementations
for a reusable barrier and a dedicated thread class for script execution.

The design uses a `Device` class to hold state and a main `DeviceThread` to
orchestrate the simulation steps. For each time step, it spawns a `MyThread`
instance for each assigned script, runs them in parallel, and waits for their
completion before synchronizing all devices with a custom `ReusableBarrier`.

While the parallel execution model is structured, the implementation contains
several race conditions and non-thread-safe data access patterns.
"""

from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    A custom, two-phase reusable barrier implemented with Semaphores.

    This barrier ensures that a group of threads all wait at a certain point
    before any of them are allowed to proceed. It is reusable, meaning it can be
    used multiple times (e.g., in a loop).
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # A list is used to pass the counter by reference to the phase method.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        # The two-phase approach prevents threads from a new cycle from proceeding
        # before all threads from the previous cycle have finished.
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the current thread count for the phase.
            threads_sem (Semaphore): The semaphore used for blocking in this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive resets the counter and releases all waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Represents a device node in the simulation.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)
        self.barrier = None
        self.list_thread = []
        self.thread.start()
        # NOTE: Using a fixed-size array for locks limits the simulation to
        # location IDs less than 100.
        self.location_lock = [None] * 100

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes the shared barrier to all devices.

        NOTE: This setup logic is not robust. It can be called by multiple devices
        and is prone to race conditions, potentially creating multiple, inconsistent
        barrier objects across the simulation.
        """
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        # This logic is inefficient as it's executed by every device.
        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script and attempts to initialize a lock for its location.

        NOTE: The lock initialization logic is complex and has a race condition.
        Multiple threads could simultaneously check for a lock and attempt to create it.

        Args:
            script (object): The script to run. A value of None signals the end of the timepoint.
            location (int): The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Pre-condition: Check if a lock for the location exists.
            if self.location_lock[location] is None:
                flag = 0
                # Attempt to find an already-initialized lock from another device.
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        flag = 1
                        break
                # If no lock is found, create a new one. This is the racy part.
                if flag == 0:
                    self.location_lock[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data for a given location.

        WARNING: This method is not thread-safe.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets data for a given location.

        WARNING: This method is not thread-safe.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class MyThread(Thread):
    """
    A dedicated thread for executing a single script.
    """
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script, ensuring synchronized access to the location's data.
        """
        # Pre-condition: Acquire lock to ensure exclusive access to the location.
        self.device.location_lock[self.location].acquire()
        
        # Block Logic: Gather data from neighbors and the local device.
        script_data = []
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Invariant: Run script only if there is data.
        if script_data:
            result = self.script.run(script_data)
            # Block Logic: Propagate results back to all involved devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
            
        # Post-condition: Release the lock.
        self.device.location_lock[self.location].release()


class DeviceThread(Thread):
    """
    The main orchestrator thread for a device.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main simulation loop.
        """
        while True:
            # Get neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Signal to terminate simulation.

            # Wait for supervisor to signal that all scripts for the timepoint are assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Create and manage threads for script execution.
            # 1. Create a thread for each script.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            # 2. Start all script threads to run them in parallel.
            for thread_elem in self.device.list_thread:
                thread_elem.start()
            
            # 3. Wait for all script threads to complete.
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            
            # Clear the list for the next timepoint.
            self.device.list_thread = []

            # Reset the timepoint event and synchronize with all other devices.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()