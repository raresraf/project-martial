
"""
This module implements a multi-threaded simulation of a distributed system of devices.

The system consists of multiple `Device` objects that communicate and synchronize
to execute scripts. It uses custom synchronization primitives, including a `Barrier`,
to coordinate the actions of the devices. Each device runs in its own thread
and can execute multiple scripts concurrently.
"""

from threading import Lock, Semaphore, Event, Thread
from sets import Set

class Barrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This barrier implementation uses a two-phase protocol to ensure that all
    threads wait at the barrier before any of them are allowed to proceed.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the Barrier.

        Args:
            num_threads (int): The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have arrived.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Implements one phase of the barrier.

        Args:
            count_threads (list[int]): A list containing the count of remaining threads.
            threads_sem (Semaphore): The semaphore to signal when all threads have arrived.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Represents a device in the distributed system.
    
    Each device has an ID, sensor data, and can execute scripts. It runs in its
    own thread and communicates with other devices.
    """

    barrier = None
    lock_list = []
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary of sensor data for the device.
            supervisor (Supervisor): The supervisor object that manages the device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()
        self.devices = []

    def __str__(self):
        """
        Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device with a list of other devices in the system.

        Initializes the barrier and locks for synchronization.

        Args:
            devices (list[Device]): A list of all devices in the system.
        """
        self.devices = devices
        if Device.barrier is None:
            Device.barrier = Barrier(len(devices))
        if Device.lock_list == []:
            zones = []
            for dev in devices:
                zones.extend(dev.sensor_data.keys())

            Device.lock_list = [Lock() for i in range(len(Set(zones)))]

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script: The script to be executed.
            location: The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.scripts_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location.

        Args:
            location: The location from which to retrieve data.

        Returns:
            The sensor data at the given location, or None if not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location.

        Args:
            location: The location to update.
            data: The new data to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining its thread.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread of execution for a Device.
    
    This thread waits for scripts to be assigned, and then executes them in
    parallel using multiple `ScriptThread`s.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device that this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop of the device thread.

        Waits for scripts, executes them, and synchronizes with other devices.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.scripts_received.wait()
            self.device.scripts_received.clear()

            tasks = [[] for i in xrange(8)]
            i = 0
            for (script, location) in self.device.scripts:
                tasks[i%8].append((script, location))
                i += 1

            script_threads = []
            for i in xrange(8):
                if tasks[i%8] != []:
                    thr = ScriptThread(self.device, neighbours, tasks[i%8])
                    script_threads.append(thr)
                    thr.start()

            for i in xrange(len(script_threads)):
                script_threads[i].join()

            Device.barrier.wait()

class ScriptThread(Thread):
    """
    A thread for executing a batch of scripts on a device.
    """
    def __init__(self, device, neighbours, scripts):
        """
        Initializes the ScriptThread.

        Args:
            device (Device): The device executing the scripts.
            neighbours (list[Device]): A list of neighboring devices.
            scripts (list): A list of scripts to be executed.
        """
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.scripts = scripts

    def run(self):
        """
        Executes each script in the batch.

        For each script, it acquires a lock for the script's location,
        gathers data from neighboring devices, runs the script, and then
        updates the data on all neighbors.
        """
        for (script, location) in self.scripts:
            Device.lock_list[location].acquire()
            script_data = []
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = script.run(script_data)


                for device in self.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
            Device.lock_list[location].release()
