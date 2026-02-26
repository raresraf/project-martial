"""
This module provides a framework for simulating a network of interconnected devices.

It defines a `Device` that acts as a node in the network, a `DeviceThread` that
manages the device's lifecycle and worker threads, and a `MyWorker` class for
executing tasks. The simulation appears to be turn-based, synchronized by a
custom `ReusableBarrier`. This implementation differs from others in its
handling of task distribution and locking.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue

class ReusableBarrier(object):
    """
    A custom implementation of a reusable barrier using a Condition variable.

    This barrier allows a specified number of threads to wait until all of them
    have reached the barrier point, at which point they are all released. It then
    resets itself for the next use.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Causes a thread to wait at the barrier.

        The last thread to arrive will notify all waiting threads and reset the
        barrier for reuse.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread has arrived, notify all waiters and reset the count.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Not all threads have arrived, wait to be notified.
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a single device node in the simulation.

    Each device has its own sensor data and is managed by a dedicated
    `DeviceThread`. It communicates with other devices via a shared task queue
    and synchronizes simulation steps using a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of the device's local sensor data.
            supervisor (object): An object responsible for providing network topology
                                 (e.g., neighbor information).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.queue = Queue()  # Shared task queue for worker threads.
        self.setup = Event()  # Event to signal completion of initial setup.
        self.threads = []     # List to hold worker thread objects.
        # A list of locks that is shared among ALL devices.
        self.locations_lock = []
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (barrier, locks) to all devices.

        This method is intended to be run by a single master device (device_id 0).
        It creates a single barrier and a single list of locks and shares references
        to them with all other devices in the simulation.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            # A fixed-size list of 25 locks is created.
            for _ in range(25):
                lock = Lock()
                self.locations_lock.append(lock)

            # Distribute the same barrier and lock list to all devices.
            for device in devices:
                device.barrier = barrier
                device.locations_lock = self.locations_lock
                device.setup.set() # Signal that setup is complete.

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals the end of a simulation step.

        Args:
            script (object): The script to be executed. If None, it signals the
                             completion of the current timepoint.
            location (int): The location index the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data. Note: This method itself is not thread-safe.
        Locking is handled by the worker threads.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data. Note: This method itself is not thread-safe.
        Locking is handled by the worker threads.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """
    Main control thread for a device.

    This thread is responsible for initializing worker threads and managing the
    main simulation loop, including synchronization and task dispatch.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Initializes workers and runs the main simulation loop."""
        # Wait until the initial setup of shared resources is complete.
        self.device.setup.wait()

        # Spawn a pool of worker threads.
        for _ in range(8):
            thread = MyWorker(self.device)
            thread.start()
            self.device.threads.append(thread)

        while True:
            # Get the list of neighbors for the current simulation step.
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                # If neighbors is None, this is the signal to shut down.
                # Enqueue None for each worker to make it exit its loop.
                for thread in self.device.threads:
                    for _ in range(8):
                        self.device.queue.put(None)
                    thread.join()
                break

            # Wait for the signal that the current timepoint is done.
            self.device.timepoint_done.wait()
            # Synchronize with all other devices before dispatching work.
            self.device.barrier.wait()

            # Dispatch all assigned scripts as tasks to the shared queue.
            for (script, location) in self.device.scripts:
                self.device.queue.put((neighbours, location, script))

            # Clear the event for the next timepoint.
            self.device.timepoint_done.clear()
            
            # Synchronize again after dispatching work.
            self.device.barrier.wait()

class MyWorker(Thread):
    """
    A worker thread that executes computational tasks for a device.
    
    Workers fetch tasks from a shared queue, acquire a lock for the target
    data location, gather data, run a script, and distribute the results.
    """
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device

    def run(self):
        """The main loop for the worker thread."""
        while True:
            # Get a task from the shared queue.
            elem = self.device.queue.get()
            
            # A None element is the signal to terminate.
            if elem is None:
                break
            
            # Acquire a lock based on the location index.
            self.device.locations_lock[elem[1]].acquire()
            
            script_data = []
            data = None

            # --- Data Gathering ---
            # This loop appears to have a flaw; it iterates through all neighbors
            # but only the data from the *last* neighbor in the list is retained.
            for device in elem[0]:
                data = device.get_data(elem[1])
            if data is not None:
                script_data.append(data)
            
            # Gather data from the local device as well.
            data = self.device.get_data(elem[1])
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # --- Script Execution ---
                result = elem[2].run(script_data)

                # --- Result Distribution ---
                # Update the data on all neighboring devices.
                for device in elem[0]:
                    device.set_data(elem[1], result)
                # Update the data on the local device.
                self.device.set_data(elem[1], result)
            
            # Release the lock for the location.
            self.device.locations_lock[elem[1]].release()

            # Indicate that the task from the queue is finished.
            self.device.queue.task_done()