"""
This module simulates a network of devices that can execute scripts on sensor data
in a coordinated, multi-threaded fashion. It defines classes for the devices,
worker threads, and a reusable synchronization barrier.
"""

from threading import Event, Thread, Condition, Lock
from Queue import Queue, Empty

class ReusableBarrier(object):
    """
    A reusable barrier synchronization primitive.

    This barrier allows a set of threads to wait for each other to reach a
    certain point of execution before any of them are allowed to continue.
    Once all threads have reached the barrier, they are all released and the
    barrier can be reused.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the
                               barrier before they are all released.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Causes the calling thread to wait at the barrier.

        The thread will block until `num_threads` threads have called this
        method. At that point, all waiting threads are released.
        """
        self.cond.acquire()
        
        self.count_threads -= 1
        if self.count_threads == 0:
            # The last thread has arrived, notify all waiting threads.
            self.cond.notify_all()
            # Reset the barrier for the next use.
            self.count_threads = self.num_threads
        else:
            # Wait for the other threads to arrive.
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a single device in the simulated network.

    Each device has a unique ID, its own sensor data, and can communicate
    with its neighbors. Devices execute scripts on data from a specific
    location, in coordination with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's
                                local sensor data, indexed by location.
            supervisor (object): An object that can provide the device with a
                                 list of its neighbors.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.neighbours = None
        self.num_threads = 24
        self.script_received = Event()
        self.scripts = []
        self.workers = []
        self.queue = None
        self.work_done = None
        self.barrier = None
        self.lock = None
        self.location_lock = []
        self.thread = DeviceThread(self)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device and its connections to other devices.

        One device in the list acts as the leader to initialize shared
        resources like the work queue, barrier, and locks. Other devices
        will use these shared resources.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == devices[0].device_id:
            # This device is the leader, responsible for creating shared resources.
            self.queue = Queue()
            self.work_done = Event()
            self.barrier = ReusableBarrier(len(devices))
            self.lock = Lock()

            # Create a lock for each location.
            for loc in range(100):
                self.location_lock.append(Lock())

            # Create and start the worker threads.
            for thread_id in range(self.num_threads):
                self.workers.append(WorkerThread(self.queue, self.work_done))
                self.workers[thread_id].start()
        else:
            # This device is a follower, using the leader's shared resources.
            self.queue = devices[0].queue
            self.work_done = devices[0].work_done
            self.barrier = devices[0].barrier
            self.lock = devices[0].lock
            self.location_lock = devices[0].location_lock

        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script (object): The script to be executed. Should have a `run` method.
            location (int): The location associated with the script execution.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is a signal that all scripts have been assigned.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location for which to retrieve data.

        Returns:
            The sensor data at the given location, or None if not available.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates sensor data at a given location.

        Args:
            location (int): The location to update.
            data: The new data to be stored.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device and its associated threads.
        """
        self.thread.join()
        if self.workers != []:
            for thread_id in range(self.num_threads):
                self.workers[thread_id].join()


class DeviceThread(Thread):
    """
    The main control thread for a single Device.

    This thread is responsible for getting neighbors, waiting for scripts,
    and dispatching work to the worker threads.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main loop of the device thread."""
        while True:
            # Get the current list of neighbors from the supervisor.
            self.device.lock.acquire()
            self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.lock.release()

            # If there are no neighbors, the simulation is done for this device.
            if self.device.neighbours is None:
                self.device.work_done.set()
                break

            # Wait until all scripts for the current step have been received.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Add all assigned scripts to the shared work queue.
            for (script, location) in self.device.scripts:
                self.device.queue.put((self.device, script, location))

            
            # Wait for all items in the queue to be processed.
            self.device.queue.join()
            
            # Synchronize with all other devices before starting the next step.
            self.device.barrier.wait()

class WorkerThread(Thread):
    """

    A worker thread that processes tasks from a shared queue.
    The tasks involve executing scripts on sensor data.
    """

    def __init__(self, queue, job_done):
        """
        Initializes the WorkerThread.

        Args:
            queue (Queue): The shared queue from which to get tasks.
            job_done (Event): An event to signal when all jobs are done.
        """
        Thread.__init__(self, name="Worker Thread")
        self.tasks = queue
        self.job_done = job_done

    def run(self):
        """The main loop of the worker thread."""
        while True:
            try:
                # Get a task from the queue without blocking.
                (device, script, location) = self.tasks.get(False)
            except Empty:
                # If the queue is empty, check if the simulation is done.
                if self.job_done.is_set():
                    break
                else:
                    continue

            script_data = []
            
            # Acquire a lock for the specific location to ensure data consistency.
            device.location_lock[location].acquire()

            # Gather data from all neighbors for the given location.
            for neighbour in device.neighbours:
                data = neighbour.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Gather data from the current device as well.
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Execute the script with the collected data.
                result = script.run(script_data)

                # Update the data on the current device and all its neighbors.
                device.set_data(location, result)
                for neighbour in device.neighbours:
                    neighbour.set_data(location, result)

            # Release the lock for the location.
            device.location_lock[location].release()
            
            # Signal that the task is done.
            self.tasks.task_done()