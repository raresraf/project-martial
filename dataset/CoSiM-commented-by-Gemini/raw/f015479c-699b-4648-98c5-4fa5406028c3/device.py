"""
This module simulates a network of devices using a multi-threaded approach.
It features a custom two-phase reusable barrier implemented with semaphores
and a worker thread model for script execution.
"""

from threading import Thread, Lock, Semaphore, Event

class ReusableBarrierSem(object):
    """
    A reusable barrier implemented using semaphores and a lock.
    This implementation uses a two-phase signaling to prevent race conditions
    where a thread that has been released re-enters the barrier before all
    other threads have been released.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Waits for all threads to reach the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Device(object):
    """Represents a device in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.lock_neigh = None
        self.lock_mine = Lock()

    def __str__(self):
        """String representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources (barrier and locks) for all devices.
        Should be called by one device at the beginning of the simulation.
        """
        no_devices = len(devices)
        lock_neigh = Lock()
        barrier = ReusableBarrierSem(no_devices)

        if self.device_id == 0:
            for i in range(no_devices):
                devices[i].barrier = barrier
                devices[i].lock_neigh = lock_neigh

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets data from a specific sensor location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Sets data for a specific sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class WorkerThread(Thread):
    """A worker thread that executes a script on a device."""

    def __init__(self, device, script, location, neighbours):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Worker Thread")
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def collect_data(self, location_data):
        """Collects data from the device and its neighbors."""
        location_data.append(self.device.get_data(self.location))
        for i in range(len(self.neighbours)):
            data = self.neighbours[i].get_data(self.location)
            location_data.append(data)

    def update_neighbours(self, result):
        """
        Updates the data of the neighboring devices.
        Uses a shared lock to protect access to neighbor data.
        """
        no_neigh = len(self.neighbours)
        for i in range(no_neigh):
            self.device.lock_neigh.acquire()
            value = self.neighbours[i].get_data(self.location)
            self.neighbours[i].set_data(self.location, max(result, value))
            self.device.lock_neigh.release()

    def update_self(self, result):
        """
        Updates the data of the current device.
        Uses a per-device lock.
        """
        self.device.lock_mine.acquire()
        value = self.device.get_data(self.location)
        self.device.set_data(self.location, max(result, value))
        self.device.lock_mine.release()

    def run(self):
        """
        Main logic for the worker thread. It collects data, runs the script,
        and updates the data on the device and its neighbors.
        """
        location_data = []
        self.collect_data(location_data)

        if len(location_data) > 0:
            result = self.script.run(location_data)
            self.update_neighbours(result)
            self.update_self(result)


class DeviceThread(Thread):
    """The main thread for a device, which spawns worker threads."""

    def __init__(self, device):
        """Initializes the device's main thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device


    def run(self):
        """
        Main loop for the device thread. It waits for scripts, spawns worker
        threads to execute them, and synchronizes with other devices.
        """
        threads = [None] * 200 # A fixed-size list for worker threads.
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.script_received.wait()
            self.device.script_received.clear()
            
            # Start a worker thread for each assigned script.
            for i in range(len(self.device.scripts)):
                (script, location) = self.device.scripts[i]
                threads[i] = WorkerThread(self.device, script, \
                    location, neighbours)
                threads[i].start()

            for i in range(len(self.device.scripts)):
                threads[i].join()

            self.device.barrier.wait()
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
