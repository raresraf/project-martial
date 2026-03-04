"""
A device simulation framework using a worker thread pool and a shared task queue.

This script implements a distributed device simulation where each device has a
pool of worker threads. Scripts are managed via a shared queue. The simulation
synchronizes using a custom reusable barrier. A key feature of this implementation
is the conditional update of sensor data, which only occurs if the new value is
greater than the existing one.
"""

from Queue import Queue
from threading import Semaphore, Lock
from threading import Event, Thread


class Device(object):
    """
    Represents a single device in the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id: A unique identifier for the device.
            sensor_data: The initial sensor data for the device.
            supervisor: The central simulation supervisor.
        """
        self.device_id = device_id
        self.read_data = sensor_data
        self.supervisor = supervisor
        self.active_queue = Queue()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.time = 0

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared reusable barrier for all devices.

        Args:
            devices: A list of all devices in the simulation.
        """
        if self.device_id == 0:
            self.new_round = ReusableBarrierSem(len(devices))
            self.devices = devices
            for device in self.devices:
                device.new_round = self.new_round
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. When a None script is received,
        it pushes all accumulated scripts to a work queue for processing.

        Args:
            script: The script to be executed.
            location: The location for the script's execution.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # End of script batch signal.
            for (script, location) in self.scripts:
                self.active_queue.put((script, location))
            # Add "poison pills" to the queue to terminate the worker threads.
            for x in range(8):
                self.active_queue.put((-1, -1))

    def get_data(self, location):
        """
        Retrieves data for a given location.

        Args:
            location: The location to retrieve data from.

        Returns:
            The data at the location, or None if not available.
        """
        return self.read_data[location] if location in self.read_data else None

    def set_data(self, location, data):
        """
        Sets data for a given location.

        Args:
            location: The location to set data at.
            data: The new data value.
        """
        if location in self.read_data:
            self.read_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device: The parent Device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers_number = 8

    def run(self):
        """
        The main execution loop for the device.

        This implementation inefficiently creates and destroys a new pool of worker
        threads in every simulation cycle.
        """
        neighbours = self.device.supervisor.get_neighbours()
        while True:
            self.workers = []
            self.device.neighbours = neighbours
            if neighbours is None:
                break

            # A new set of workers is created in each iteration of the loop.
            for i in range(self.workers_number):
                new_worker = WorkerThread(self.device)
                self.workers.append(new_worker)
                new_worker.start()

            for worker in self.workers:
                worker.join()
            self.device.new_round.wait()
            neighbours = self.device.supervisor.get_neighbours()


class WorkerThread(Thread):
    """
    A worker thread that processes scripts from a shared queue.
    """
    def __init__(self, device):
        """
        Initializes a WorkerThread.

        Args:
            device: The parent Device object.
        """
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Continuously fetches and executes scripts from the device's active queue.
        """
        while True:
            script, location = self.device.active_queue.get()
            # The "poison pill" signals the worker to terminate.
            if script == -1:
                break
            script_data = []
            matches = []
            # Gather data from neighbors.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    matches.append(device)
                    script_data.append(data)
            # Gather data from the parent device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
                matches.append(self.device)

            if len(script_data) > 1:
                result = script.run(script_data)
                # Conditionally update data only if the new result is greater.
                for device in matches:
                    old_value = device.get_data(location)
                    if old_value < result:
                        device.set_data(location, result)


class ReusableBarrierSem():
    """
    A reusable barrier implemented using semaphores and a lock.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier.

        Args:
            num_threads: The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads


        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Makes the calling thread wait at the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the two-phase barrier protocol."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase of the two-phase barrier protocol."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()