
"""
Models a distributed network of devices that process sensor data concurrently.

This script simulates a system of interconnected devices, featuring a load-balancing
mechanism to distribute script execution tasks among a pool of worker threads.
Synchronization is managed by a custom reusable barrier and shared locks.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue

class ReusableBarrier(object):
    """A reusable barrier for thread synchronization using semaphores.

    This barrier uses a two-phase protocol to ensure that a specified number of
    threads can repeatedly synchronize at a point in an iterative algorithm.

    Attributes:
        num_threads (int): The number of threads to synchronize.
        count_threads1 (list[int]): A list containing the counter for the first phase.
                                    Using a list allows mutable integer behavior.
        count_threads2 (list[int]): A list containing the counter for the second phase.
        count_lock (Lock): A lock to protect access to the counters.
        threads_sem1 (Semaphore): Semaphore for the first synchronization phase.
        threads_sem2 (Semaphore): Semaphore for the second synchronization phase.
    """
    

    def __init__(self, num_threads):
        """Initializes the ReusableBarrier."""
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()
        
        self.threads_sem1 = Semaphore(0)
        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads arrive."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """Represents a single device in the distributed sensor network.

    This device uses a pool of worker threads (`DeviceWorker`) to execute scripts
    and coordinates with other devices using a shared barrier and locks.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding sensor readings.
        supervisor (Supervisor): An object for retrieving neighbor information.
        set_lock (Lock): A lock to protect writes to this device's sensor_data.
        neighbours_lock (Lock): A shared lock for accessing neighbor data.
        neighbours_barrier (ReusableBarrier): A shared barrier for synchronization.
        thread (DeviceThread): The main thread of execution for this device.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.result_queue = Queue.Queue()
        self.set_lock = Lock()
        self.neighbours_lock = None
        self.neighbours_barrier = None

        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and shares synchronization objects among all devices.

        A master device (the first in the list) creates the shared lock and
        barrier, which are then assigned to all other devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == devices[0].device_id:
            self.neighbours_lock = Lock()
            self.neighbours_barrier = ReusableBarrier(len(devices))
        
        else:
            self.neighbours_lock = devices[0].neighbours_lock
            self.neighbours_barrier = devices[0].neighbours_barrier

        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device.

        Args:
            script (Script): The script to execute, or None to signal completion.
            location (str): The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a location in a thread-safe manner."""
        self.set_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.set_lock.release()

    def shutdown(self):
        """Waits for the main device thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The main execution thread for a Device.

    It manages a pool of `DeviceWorker` threads and distributes scripts among
    them using a load-balancing strategy.

    Attributes:
        device (Device): The parent device instance.
        workers (list): A list of `DeviceWorker` threads.
    """
    

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []

    def run(self):
        """The main control loop for the device.

        In each timepoint, it acquires neighbor information, waits for scripts,
        distributes them to worker threads based on location affinity and load,
        runs the workers, and finally synchronizes at a global barrier.
        """
        while True:
            self.device.neighbours_lock.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.neighbours_lock.release()

            if neighbours is None:
                break

            
            self.device.script_received.wait()

            
            self.workers = []
            for i in range(8):
                self.workers.append(DeviceWorker(self.device, i, neighbours))

            
            for (script, location) in self.device.scripts:

                
                added = False
                for worker in self.workers:
                    if location in worker.locations:
                        worker.add_script(script, location)
                        added = True

                
                if added == False:
                    minimum = len(self.workers[0].locations)
                    chosen_worker = self.workers[0]
                    for worker in self.workers:
                        if minimum > len(worker.locations):
                            minimum = len(worker.locations)
                            chosen_worker = worker

                    chosen_worker.add_script(script, location)

            
            for worker in self.workers:
                worker.start()

            
            for worker in self.workers:
                worker.join()

            
            self.device.neighbours_barrier.wait()
            self.device.script_received.clear()


class DeviceWorker(Thread):
    """A worker thread that executes a batch of scripts.

    Each worker can handle multiple scripts, typically grouped by location to
    improve data locality.

    Attributes:
        device (Device): The parent device.
        worker_id (int): A unique ID for the worker within its device.
        scripts (list): A list of script objects to execute.
        locations (list): A corresponding list of location contexts.
        neighbours (list): A list of neighboring Device objects.
    """
    

    def __init__(self, device, worker_id, neighbours):
        """Initializes the DeviceWorker thread."""
        Thread.__init__(self)
        self.device = device
        self.worker_id = worker_id
        self.scripts = []
        self.locations = []
        self.neighbours = neighbours

    def add_script(self, script, location):
        """Adds a script and its location to this worker's task batch."""
        self.scripts.append(script)
        self.locations.append(location)

    def run_scripts(self):
        """Executes all scripts assigned to this worker."""
        for (script, location) in zip(self.scripts, self.locations):

            script_data = []
            
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            
            if script_data != []:
                res = script.run(script_data)

                for device in self.neighbours:
                    device.set_data(location, res)
                self.device.set_data(location, res)

    def run(self):
        """The main entry point for the worker thread."""
        self.run_scripts()
