# -*- coding: utf-8 -*-
"""
This module simulates a network of devices that process data in synchronized
time steps. This implementation uses a "thread-per-task" model, where each
device spawns a new sub-thread for every script it needs to execute in a
timepoint. Synchronization between devices is handled by a custom, semaphore-based
reusable barrier.

Classes:
    ReusableBarrier: A custom implementation of a reusable barrier.
    Device: Represents a node in the network.
    DeviceThread: The main control loop for a device.
    DeviceSubThread: A short-lived thread to execute a single script.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    A reusable barrier implemented using semaphores and a lock.
    
    This allows a group of threads to synchronize at a point, and to reuse the
    barrier in a loop. It uses a two-phase protocol to prevent race conditions
    where fast threads might "lap" slower threads.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters are wrapped in a list to make them mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0) # Gate for the first phase.
        self.threads_sem2 = Semaphore(0) # Gate for the second phase.

    def wait(self):
        """Causes the calling thread to wait at the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0: # The last thread has arrived.
                # Open the gate for all waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads  # Reset for the next use.
        threads_sem.acquire() # Threads wait here until the gate is opened.

class Device(object):
    """Represents a single device in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()
        self.scripts = []
        self.data_lock = Lock() # A lock for the device's own data dictionary.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Barrier and location_locks are initialized here but properly set up in setup_devices.
        self.barrier = ReusableBarrier(0)
        self.location_locks = {}

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources like the barrier and locks.
        
        The device with ID 0 acts as a leader to create the shared objects.
        """
        if self.device_id == 0:
            # The leader creates a shared lock for each possible location.
            # Note: This hardcodes the number of locations to 100.
            for location in xrange(100):
                self.location_locks[location] = Lock()
            # The leader creates the barrier for all devices.
            self.barrier = ReusableBarrier(len(devices))

        # Distribute the shared objects to all devices.
        for dev in devices:
            if self.device_id == 0:
                dev.barrier = self.barrier
                dev.location_locks = self.location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to be run. A None script signals the end of assignments
        for the current timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Thread-safely sets the sensor data for a given location.
        Note: This uses a per-device lock, which is different from the
        per-location lock used during script execution.
        """
        with self.data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """The main control loop for a device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop, advancing in discrete timepoints.
        
        Synchronization Logic:
        1. Wait for scripts to be assigned for the current timepoint.
        2. Create and run a new sub-thread for each script.
        3. Wait for all sub-threads to complete their work.
        4. Wait at a global barrier to synchronize with all other devices
           before starting the next timepoint.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Supervisor signals shutdown.
                break

            # Wait for supervisor to finish assigning scripts.
            self.device.scripts_received.wait()
            self.device.scripts_received.clear()

            # --- Thread-per-task execution ---
            # Create a sub-thread for each assigned script.
            device_threads = [
                DeviceSubThread(self.device, script, location, neighbours)
                for (script, location) in self.device.scripts
            ]

            # Start all sub-threads.
            for thread in device_threads:
                thread.start()

            # Wait for all sub-threads to complete their execution.
            for thread in device_threads:
                thread.join()

            # --- Global Synchronization Point ---
            # All devices wait here, ensuring no device proceeds to the next
            # timepoint until all have finished the current one.
            self.device.barrier.wait()
            self.device.scripts = []


class DeviceSubThread(Thread):
    """
    A worker thread that executes a single script.
    """

    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script's logic: gather data, run script, and update data.
        """
        # Acquire the lock for the specific location to prevent data races.
        self.device.location_locks[self.location].acquire()
        try:
            script_data = []
            
            # --- Data Gathering Phase ---
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # --- Execution and Dissemination Phase ---
            if script_data:
                result = self.script.run(script_data)
                # Update data on neighbors and self.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)
        finally:
            # Always release the lock.
            self.device.location_locks[self.location].release()