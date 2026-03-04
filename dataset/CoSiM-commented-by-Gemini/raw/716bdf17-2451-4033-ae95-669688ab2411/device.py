"""
Implements a device simulation using a custom reusable barrier and a set of worker threads.

This script defines a distributed device simulation where each device uses a pool of
worker threads (`DeviceWorker`) to execute scripts. Synchronization between devices is
handled by a custom `ReusableBarrier` implementation. Scripts are distributed among
the worker threads based on the location they operate on, with a simple load-balancing
mechanism.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue

class ReusableBarrier(object):
    """
    A custom implementation of a reusable barrier.

    This barrier allows a fixed number of threads to synchronize at a point, and
    can be reused multiple times. It uses a two-phase protocol with semaphores
    to prevent race conditions on reuse.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads: The number of threads that will be synchronizing at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()
        
        self.threads_sem1 = Semaphore(0)
        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have reached it.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Represents one phase of the two-phase barrier protocol.

        Args:
            count_threads: A list containing the count of remaining threads for the phase.
            threads_sem: The semaphore used for signaling in this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            
            if count_threads[0] == 0:
                # Last thread to arrive releases all other threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                
                # Reset the count for the next use of the barrier.
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Represents a device in the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id: Unique ID for the device.
            sensor_data: Local sensor data.
            supervisor: The simulation supervisor.
        """
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
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization primitives (lock and barrier) for all devices.

        Args:
            devices: A list of all devices in the simulation.
        """
        
        if self.device_id == devices[0].device_id:
            # Device 0 is responsible for creating the shared objects.
            self.neighbours_lock = Lock()
            self.neighbours_barrier = ReusableBarrier(len(devices))
        
        else:
            # Other devices get a reference to the shared objects.
            self.neighbours_lock = devices[0].neighbours_lock
            self.neighbours_barrier = devices[0].neighbours_barrier

        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device.

        Args:
            script: The script to be executed.
            location: The location for execution.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script is a signal that all scripts for the timepoint are assigned.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data for a given location.

        Args:
            location: The location to retrieve data from.

        Returns:
            The data at the location, or None.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Thread-safely sets data at a given location.

        Args:
            location: The location to set data at.
            data: The data to be set.
        """
        
        self.set_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.set_lock.release()

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, managing a set of worker threads.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device: The parent Device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []

    def run(self):
        """
        The main execution loop for the device.

        It gets neighbors, distributes scripts to worker threads, waits for them
        to complete, and then synchronizes at a global barrier.
        """
        while True:

            # The lock suggests that the supervisor's get_neighbours method is not thread-safe.
            self.device.neighbours_lock.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.neighbours_lock.release()

            if neighbours is None:
                break

            # Wait until scripts are received.
            self.device.script_received.wait()

            # Create a fixed pool of 8 worker threads.
            self.workers = []
            for i in range(8):
                self.workers.append(DeviceWorker(self.device, i, neighbours))

            # Distribute scripts to workers based on location.
            for (script, location) in self.device.scripts:

                # Try to assign the script to a worker already handling this location.
                added = False
                for worker in self.workers:
                    if location in worker.locations:
                        worker.add_script(script, location)
                        added = True

                # If no worker is handling this location, assign it to the least loaded worker.
                if added == False:
                    minimum = len(self.workers[0].locations)
                    chosen_worker = self.workers[0]
                    for worker in self.workers:
                        if minimum > len(worker.locations):
                            minimum = len(worker.locations)
                            chosen_worker = worker

                    chosen_worker.add_script(script, location)

            # Start and wait for all workers to complete.
            for worker in self.workers:
                worker.start()

            for worker in self.workers:
                worker.join()

            # Synchronize with other devices.
            self.device.neighbours_barrier.wait()
            self.device.script_received.clear()


class DeviceWorker(Thread):
    """
    A worker thread that executes a subset of the device's scripts.
    """

    def __init__(self, device, worker_id, neighbours):
        """
        Initializes the DeviceWorker.

        Args:
            device: The parent Device object.
            worker_id: The ID of this worker.
            neighbours: A list of neighboring devices.
        """

        Thread.__init__(self)
        self.device = device
        self.worker_id = worker_id
        self.scripts = []
        self.locations = []
        self.neighbours = neighbours

    def add_script(self, script, location):
        """
        Adds a script to this worker's workload.

        Args:
            script: The script to be added.
            location: The location for the script's execution.
        """
        self.scripts.append(script)
        self.locations.append(location)

    def run_scripts(self):
        """
        Executes all scripts assigned to this worker sequentially.
        """

        for (script, location) in zip(self.scripts, self.locations):

            script_data = []
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Gather data from the parent device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Run the script and distribute the result.
            if script_data != []:
                res = script.run(script_data)

                for device in self.neighbours:
                    device.set_data(location, res)
                self.device.set_data(location, res)

    def run(self):
        """The main entry point for the worker thread."""
        self.run_scripts()