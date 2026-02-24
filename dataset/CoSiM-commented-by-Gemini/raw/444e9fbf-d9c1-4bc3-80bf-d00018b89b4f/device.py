"""
This module implements a simulation framework for a network of devices that
process sensor data. It includes classes for a Device, its corresponding worker
thread (DeviceThread), and a simple ThreadPool for concurrent script execution.

The system appears to be designed for a discrete-time simulation where at each
"timepoint," devices execute scripts on data, possibly communicating with their
neighbors, and then synchronize using a barrier before proceeding to the next
timepoint.
"""

from threading import Thread, Condition, Semaphore
# The following imports suggest this code might be for Python 2.
# 'barrier' is not a standard Python 2/3 library. It might be a custom one.
from barrier import Barrier
# 'threadpool' seems to be another custom or third-party library.
# The code below re-implements a thread pool, which is strange.
from threadpool import ThreadPool
from Queue import Queue


class Device(object):
    """Represents a single device in the simulated network.

    Each device has a unique ID, local sensor data, and can be assigned scripts
    to execute. It runs in its own thread (DeviceThread) and synchronizes with
    other devices using a barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping locations to sensor data values.
            supervisor (object): A supervisor object that manages the network
                                 of devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        # A semaphore for each data location to ensure thread-safe access.
        self.data_semaphores = {loc: Semaphore(1) for loc in sensor_data}
        self.scripts = []  # A list of (script, location) tuples to be executed.

        # Condition variable to signal script assignment and timepoint changes.
        self.new_script = False
        self.timepoint_end = False
        self.cond = Condition()

        self.barrier = None  # Synchronization barrier for all devices.
        self.supervisor = supervisor
        self.thread = DeviceThread(self)

        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the barrier for all devices.

        This method should be called on one device (e.g., device with ID 0) to
        initialize and distribute a shared barrier to all other devices in the
        network.

        Args:
            devices (list): A list of all Device objects in the network.
        """
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            for neigh in devices:
                if neigh.device_id != self.device_id:
                    neigh.set_barrier(self.barrier)

    def set_barrier(self, barrier):
        """Assigns a shared barrier to this device.

        Args:
            barrier (Barrier): The shared barrier instance.
        """
        self.barrier = barrier

    def assign_script(self, script, location):
        """Assigns a new script to be executed by the device.

        This method is called by an external entity (e.g., the supervisor) to
        give work to the device.

        Args:
            script (object): The script to be executed. It should have a `run` method.
            location (str): The data location the script operates on.
        """
        with self.cond:
            if script is not None:
                self.scripts.append((script, location))
                self.new_script = True
            else:
                # A None script signals the end of a timepoint.
                self.timepoint_end = True
            self.cond.notifyAll()

    def timepoint_ended(self):
        """Waits for and signals the end of a simulation timepoint.

        This method is used by the device's thread to block until new scripts
        are assigned or the end of a timepoint is signaled.

        Returns:
            bool: True if the timepoint has ended, False otherwise.
        """
        with self.cond:
            while not self.new_script and not self.timepoint_end:
                self.cond.wait()

            if self.new_script:
                self.new_script = False
                return False
            else: # self.timepoint_end is True
                self.timepoint_end = False
                # If there are scripts, we have new work for the next phase.
                self.new_script = len(self.scripts) > 0
                return True

    def get_data(self, location):
        """Retrieves sensor data from a specific location in a thread-safe manner.

        Args:
            location (str): The data location to retrieve from.

        Returns:
            The sensor data at the given location, or None if the location is
            not available on this device.
        """
        if location in self.sensor_data:
            self.data_semaphores[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data at a specific location in a thread-safe manner.

        Args:
            location (str): The data location to update.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_semaphores[location].release()

    def shutdown(self):
        """Shuts down the device's worker thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main worker thread for a Device.

    This thread is responsible for the device's lifecycle, which involves
    getting neighbors, executing scripts in a thread pool, and synchronizing
    at a barrier at the end of each timepoint.
    """

    def __init__(self, device):
        """Initializes the device thread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    @staticmethod
    def run_script(own_device, neighbours, script, location):
        """Executes a single script.

        This static method gathers data from neighboring devices and the local
        device for a given location, runs the script on the collected data,
        and then distributes the result back to all involved devices.

        Args:
            own_device (Device): The device executing the script.
            neighbours (list): A list of neighboring Device objects.
            script (object): The script to execute.
            location (str): The data location for the script.
        """
        script_data = []

        # Gather data from neighbors.
        for device in neighbours:
            if device is own_device:
                continue
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Gather data from the local device.
        data = own_device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Execute the script on the collected data.
            result = script.run(script_data)

            # Distribute the result back to neighbors.
            for device in neighbours:
                if device is not own_device:
                    device.set_data(location, result)

            # Update the local device's data.
            own_device.set_data(location, result)

    def run(self):
        """The main loop of the device thread."""
        pool_size = 8
        pool = ThreadPool(pool_size)

        while True:
            # Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # If no neighbors, the simulation is likely ending.
                break

            # Process scripts for the current timepoint.
            offset = 0
            while not self.device.timepoint_ended():
                scripts = self.device.scripts[offset:]
                for (script, location) in scripts:
                    pool.add_task(DeviceThread.run_script, self.device,
                                  neighbours, script, location)

                offset = len(self.device.scripts)

            # Wait for all scripts in the current timepoint to complete.
            pool.wait()

            # Synchronize with all other devices before the next timepoint.
            self.device.barrier.wait()

        # Clean up the thread pool.
        pool.terminate()

# The code from here seems to be a re-implementation of a thread pool,
# even though 'threadpool' is imported at the top. This might indicate
# code evolution or a dependency issue.

from Queue import Queue
from threading import Thread

class Worker(Thread):
    """A worker thread that consumes tasks from a queue."""

    def __init__(self, tasks):
        """Initializes the worker.

        Args:
            tasks (Queue): A queue of tasks to be executed.
        """
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True # Set as daemon so they won't block program exit.

    def run(self):
        """The main loop of the worker thread."""
        while True:
            func, args, kargs = self.tasks.get()
            try:
                # Execute the task. A ValueError is used as a signal to terminate.
                func(*args, **kargs)
            except ValueError:
                # This is a graceful termination signal.
                self.tasks.task_done()
                break
            except Exception as e:
                # Log other exceptions if needed.
                print("Error in worker thread: %s" % e)
                self.tasks.task_done()
            else:
                self.tasks.task_done()


class ThreadPool(object):
    """A simple thread pool implementation.

    Manages a pool of worker threads to execute tasks concurrently.
    """

    def __init__(self, num_threads):
        """Initializes the thread pool.

        Args:
            num_threads (int): The number of worker threads in the pool.
        """
        self.tasks = Queue(num_threads)
        self.workers = []
        for _ in range(num_threads):
            worker = Worker(self.tasks)
            self.workers.append(worker)
            worker.start()

    def add_task(self, func, *args, **kargs):
        """Adds a new task to the queue.

        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            **kargs: Keyword arguments for the function.
        """
        self.tasks.put((func, args, kargs))

    def wait(self):
        """Blocks until all tasks in the queue are completed."""
        self.tasks.join()

    def terminate(self):
        """Terminates all worker threads in the pool."""
        self.wait()

        def raising_dummy():
            """A dummy task that raises an exception to signal termination."""
            raise ValueError

        # Send a termination signal to each worker.
        for _ in range(len(self.workers)):
            self.add_task(raising_dummy)
        
        # Wait for all workers to finish.
        for worker in self.workers:
            worker.join()
