"""
This module provides a framework for simulating a distributed system of devices.

It uses a "thread-per-task" model where a master thread on each device
spawns a new worker thread for every script it needs to execute in a given
timepoint. Synchronization is achieved through a shared barrier and a set of
dynamically-created locks for data locations.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier():
    """
    A reusable, two-phase barrier for synchronizing a fixed number of threads.

    This barrier makes threads wait for each other at a synchronization point.
    It is reusable, meaning it can be used multiple times.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
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
        """Manages one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Represents a single device in the simulation.

    Each device has a master thread (`DeviceThread`) which in turn spawns
    worker threads (`NewThread`) for each assigned task.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): Local sensor data, keyed by location.
            supervisor: The supervisor object for getting neighbor information.
        """
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
        self.location_lock = [None] * 100

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and shares synchronization primitives among all devices.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        if self.barrier is None:
            # The first device to get here creates the shared barrier.
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
        Assigns a script to be executed or signals the end of a timepoint.

        Args:
            script: The script to execute, or None to end the timepoint.
            location (int): The data location for the script.
        """
        ok = 0
        if script is not None:
            self.scripts.append((script, location))
            # Lazy, on-demand initialization of location locks.
            if self.location_lock[location] is None:
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        ok = 1
                        break
                if ok == 0:
                    self.location_lock[location] = Lock()
            self.script_received.set()
        else:
            # A None script signals the end of script assignment for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Gets data for a given location.

        Args:
            location (int): The location index.

        Returns:
            The data at the location, or None if not present on this device.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets data for a given location.

        Args:
            location (int): The location index.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its master thread."""
        self.thread.join()

class NewThread(Thread):
    """
    A short-lived worker thread that executes a single script.
    """
    def __init__(self, device, location, script, neighbours):
        """
        Initializes the worker thread.

        Args:
            device (Device): The parent device.
            location (int): The location the script will operate on.
            script: The script to execute.
            neighbours (list): A list of neighboring devices.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """The main logic for script execution."""
        script_data = []
        # Acquire the shared lock for this location before accessing data.
        self.device.location_lock[self.location].acquire()
        
        # Gather data from neighbors.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
        # Gather local data.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Execute script and broadcast the result to all relevant devices.
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
        
        self.device.location_lock[self.location].release()

class DeviceThread(Thread):
    """
    The main master thread for a device, responsible for orchestrating work.
    """

    def __init__(self, device):
        """
        Initializes the master thread.

        Args:
            device (Device): The parent device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main execution loop for the master thread."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Supervisor signals shutdown.
                break

            # Wait for the supervisor to signal that all scripts for this
            # timepoint have been assigned.
            self.device.timepoint_done.wait()

            # --- Thread-per-task execution ---
            # For each assigned script, spawn a new worker thread.
            for (script, location) in self.device.scripts:
                thread = NewThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            # Start all worker threads for this timepoint.
            for thread_elem in self.device.list_thread:
                thread_elem.start()
            # Wait for all worker threads to complete.
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            self.device.list_thread = []

            # Reset for the next timepoint and synchronize with other devices.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()