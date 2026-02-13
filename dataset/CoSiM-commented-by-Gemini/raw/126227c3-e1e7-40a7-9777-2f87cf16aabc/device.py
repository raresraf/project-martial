"""
This module simulates a distributed system of devices where each device has a
main control thread that, for each timepoint, creates a new pool of worker
threads to execute scripts.

Key architectural features:
- Inefficient Threading Model: For every timepoint, a new set of worker threads
  is created, started, and then joined. This repeated creation and destruction
  of threads is highly inefficient. A persistent thread pool would be better.
- Unsafe Data Access: The methods for reading and writing sensor data are not
  synchronized, leading to severe race conditions when multiple workers access
  data from the same location.
"""

from Queue import Queue
from threading import Semaphore, Lock, Thread


class Device(object):
    """
    Represents a single device node in the simulation.

    It holds the device's state, including its data and scripts, and is managed
    by a dedicated control thread (`DeviceThread`).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        # Renamed from sensor_data to avoid confusion.
        self.read_data = sensor_data
        self.supervisor = supervisor
        # A shared work queue for the worker threads of this device.
        self.active_queue = Queue()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.time = 0

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes the shared barrier to all devices."""
        # Device 0 acts as the coordinator to create the global barrier.
        if self.device_id == 0:
            self.new_round = ReusableBarrierSem(len(devices))
            self.devices = devices
            for device in self.devices:
                device.new_round = self.new_round
        self.thread.start()

    def assign_script(self, script, location):
        """
        Collects scripts for a timepoint and then dispatches them to the work queue.
        """
        if script is not None:
            # During a timepoint, scripts are just collected in a list.
            self.scripts.append((script, location))
        else:
            # A `None` script signals the end of assignments. All collected
            # scripts are then put onto the queue for the workers.
            for (script, location) in self.scripts:
                self.active_queue.put((script, location))
            # Add "poison pills" to the queue to terminate the worker threads.
            for x in range(8):
                self.active_queue.put((-1, -1))

    def get_data(self, location):
        """
        Retrieves data for a given location.
        
        @warning Not Thread-Safe: This method reads from a shared dictionary
        without any locking, which can lead to race conditions with `set_data`.
        """
        return self.read_data[location] if location in self.read_data else None

    def set_data(self, location, data):
        """
        Updates data for a given location.

        @warning Not Thread-Safe: This method writes to a shared dictionary
        without any locking, which can lead to race conditions.
        """
        if location in self.read_data:
            self.read_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device. It manages the lifecycle of a timepoint,
    including the creation and destruction of worker threads.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers_number = 8

    def run(self):
        """The main execution loop for the device's control logic."""
        neighbours = self.device.supervisor.get_neighbours()
        while True:
            self.workers = []
            self.device.neighbours = neighbours
            if neighbours is None:
                break # End of simulation.

            # Inefficient Threading: A new pool of workers is created for every timepoint.
            for i in range(self.workers_number):
                new_worker = WorkerThread(self.device)
                self.workers.append(new_worker)
                new_worker.start()

            # Wait for all workers to finish their tasks for the current timepoint.
            for worker in self.workers:
                worker.join()
            
            # Synchronize with all other devices before the next timepoint.
            self.device.new_round.wait()
            neighbours = self.device.supervisor.get_neighbours()


class WorkerThread(Thread):
    """
    A worker thread that executes a single script task. It is created and
    destroyed within a single timepoint.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop for the worker."""
        while True:
            # Get a task from the shared queue.
            script, location = self.device.active_queue.get()
            # The poison pill `(-1, -1)` signals termination.
            if script == -1:
                break
                
            script_data = []
            matches = []
            # Gather data from neighbors. These `get_data` calls are not thread-safe.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    matches.append(device)
                    script_data.append(data)
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
                matches.append(self.device)

            if len(script_data) > 1:
                result = script.run(script_data)
                # This check-then-act on `old_value` and `result` is a race condition
                # because `get_data` and `set_data` are not atomic.
                for device in matches:
                    old_value = device.get_data(location)
                    if old_value < result:
                        device.set_data(location, result)


class ReusableBarrierSem():
    """
    A custom implementation of a reusable barrier for thread synchronization.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """Executes the first phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Executes the second phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()
