"""
This module provides a framework for a multi-threaded device simulation.
It features a different threading model compared to typical worker pools, where
each main device thread spawns a new set of worker threads for each simulation
step. Synchronization between the main device threads is managed by a custom-
implemented two-phase reusable barrier (`ReusableBarrierSem`).
"""

from Queue import Queue
from threading import Semaphore, Lock
from threading import Event, Thread


class Device(object):
    """
    Represents a device in the simulation network.

    Each device holds sensor data, processes scripts assigned by a supervisor,
    and communicates with neighboring devices. It manages a queue of tasks for
    its worker threads.

    Attributes:
        device_id (int): Unique identifier for the device.
        read_data (dict): Sensor data for the device, keyed by location.
        supervisor (object): The central supervisor managing the simulation.
        active_queue (Queue): A queue holding tasks for the worker threads.
        scripts (list): A list of scripts to be processed in the current step.
        thread (DeviceThread): The main execution thread for this device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device."""
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
        Initializes and distributes synchronization primitives.

        Called by a single device (ID 0) to create and share a reusable barrier
        among all devices. It then starts the main thread for this device.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            # Create a single barrier instance to be shared by all devices.
            self.new_round = ReusableBarrierSem(len(devices))
            self.devices = devices
            # Distribute the barrier to all other devices.
            for device in self.devices:
                device.new_round = self.new_round
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script or signals the end of script assignment.

        If a script is provided, it's added to a temporary list. If script is None,
        all collected scripts are pushed to the active work queue, followed by
        sentinel values to terminate the worker threads for this step.

        Args:
            script (object): The script to execute, or None to signal completion.
            location (str): The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # All scripts for this round have been assigned; push to the work queue.
            for (script, location) in self.scripts:
                self.active_queue.put((script, location))
            # Add sentinel values to the queue to signal worker threads to exit.
            for x in range(8):
                self.active_queue.put((-1, -1))

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.
        NOTE: This access is not protected by a lock, which could lead to race
        conditions if multiple workers access the same data concurrently.
        """
        return self.read_data[location] if location in self.read_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location.
        NOTE: This access is not protected by a lock, which could lead to race
        conditions.
        """
        if location in self.read_data:
            self.read_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    Main control thread for a Device.

    In each simulation step, this thread spawns a new set of worker threads
    to process tasks and then waits for all devices to synchronize at a barrier.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers_number = 8

    def run(self):
        """
        The main loop for the device thread.
        It continuously gets neighbors, spawns workers, and synchronizes.
        """
        neighbours = self.device.supervisor.get_neighbours()
        while True:
            self.workers = []
            self.device.neighbours = neighbours
            # Supervisor returning None signals the end of the simulation.
            if neighbours is None:
                break

            # A new set of workers is created for each simulation step.
            for i in range(self.workers_number):
                new_worker = WorkerThread(self.device)
                self.workers.append(new_worker)
                new_worker.start()

            # Wait for all worker threads for this step to complete.
            for worker in self.workers:
                worker.join()
            
            # Wait at the barrier for all other DeviceThreads to finish their step.
            self.device.new_round.wait()
            
            # Get neighbors for the next round.
            neighbours = self.device.supervisor.get_neighbours()


class WorkerThread(Thread):
    """
    A worker thread that executes a single script task.
    These threads are created and destroyed within each simulation step.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Continuously fetches tasks from the device's queue and executes them
        until a sentinel value is received.
        """
        while True:
            script, location = self.device.active_queue.get()
            # Sentinel value (-1, -1) signals the worker to terminate.
            if script == -1:
                break
            
            script_data = []
            matches = []
            # Gather data from neighboring devices.
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

            # The script is only run if data is gathered from more than one device.
            if len(script_data) > 1:
                result = script.run(script_data)
                # Update data on all matching devices, but only if the new
                # result is greater than the old value.
                for device in matches:
                    old_value = device.get_data(location)
                    if old_value < result:
                        device.set_data(location, result)


class ReusableBarrierSem():
    """
    A custom implementation of a reusable barrier using Semaphores.
    This is a two-phase barrier to ensure that threads from a previous `wait()`
    call have all exited the barrier before it can be used again.
    """
    def __init__(self, num_threads):
        """Initializes the reusable barrier."""
        self.num_threads = num_threads
        # Counter for the first phase.
        self.count_threads1 = self.num_threads
        # Counter for the second phase.
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        # Semaphore to block threads in the first phase.
        self.threads_sem1 = Semaphore(0)
        # Semaphore to block threads in the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait until all threads reach the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        First phase of the barrier. Threads are blocked here until all
        threads have entered this phase.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread to arrive releases all other threads.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset counter for the next use of the barrier.
                self.count_threads1 = self.num_threads
        # Threads block here until released by the last thread.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        Second phase to prevent race conditions on barrier reuse. Ensures all
        threads from the previous `wait` call have passed through phase1
        before allowing a new set of `wait` calls to proceed.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()
