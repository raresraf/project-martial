"""
Models a device in a distributed sensor network simulation.

This script defines a device's behavior using a persistent worker pool contained
within the main device thread class. Synchronization between devices is handled
by a custom two-phase semaphore-based barrier.
"""


from threading import Semaphore, Lock, Thread, Event
import Queue

class ReusableBarrier(object):
    """
    A reusable barrier implemented using two semaphores for two-phase synchronization.

    This allows a group of threads to wait for each other to reach a certain
    point before all of them are allowed to proceed.
    """

    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        # Use lists to allow modification of count from within phase() in Python 2.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all threads have called wait(). Consists of two phases."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier.
        The last thread to enter releases all other waiting threads.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread arrived, release all threads waiting on the semaphore.
                for _ in xrange(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads # Reset for next use.
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device node in the simulated network.

    Shared resources (barrier, locks) are created by Device 0 and distributed
    to all other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance and starts its main control thread.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's local sensor readings.
            supervisor (Supervisor): The central supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None # Will be set by Device 0.
        self.locks = None   # Will be set by Device 0.

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources by having Device 0 create and distribute them.
        """
        # Device 0 is the master, it creates the shared objects.
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            locks = {}

            # Directly assign the shared objects to all device instances.
            for i in xrange(len(devices)):
                devices[i].barrier = barrier
                devices[i].locks = locks

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script signals the end of assignments.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location. Not thread-safe by itself."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data for a specific location. Not thread-safe by itself."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device, which also manages the worker pool.
    """

    def __init__(self, device):
        """
        Initializes the main thread and creates a persistent pool of worker threads.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_list = []
        self.queue = Queue.Queue()

        # Create and start the worker pool.
        for _ in xrange(8):
            thread = Thread(target=self.executor)
            thread.start()
            self.thread_list.append(thread)

    def run(self):
        """
        The main time-stepped loop of the device.
        """
        while True:
            # Get neighbours for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()

            # A None from get_neighbours signals the end of the simulation.
            if neighbours is None:
                # Signal workers to terminate.
                for _ in xrange(8):
                    self.queue.put(None)
                self.shutdown()
                self.thread_list = []
                break

            # Wait until the supervisor signals that all scripts for the timepoint are assigned.
            self.device.timepoint_done.wait()

            # Enqueue all scripts for the workers to process.
            for (script, location) in self.device.scripts:
                self.queue.put((script, location, neighbours))

            # Wait for the workers to finish all tasks for this timepoint.
            self.queue.join()

            # Synchronize with all other devices at the barrier.
            self.device.barrier.wait()

            # Clear the event for the next timepoint.
            self.device.timepoint_done.clear()


    def executor(self):
        """
        The target function for the worker threads in the pool.
        
        Fetches tasks from the queue and executes them. This includes dynamically
        creating locks for unseen locations, gathering data, running the script,
        and broadcasting the result.
        """
        while True:
            # Block until a task is available.
            items = self.queue.get()

            # A None item is the signal to terminate.
            if items is None:
                self.queue.task_done()
                break

            script, location, neighbours = items

            # Lazily and non-atomically create locks for locations.
            # NOTE: This can be a race condition if multiple threads see a new
            # location simultaneously.
            if location not in self.device.locks:
                self.device.locks[location] = Lock()

            # Acquire the lock for this location to ensure exclusive access.
            self.device.locks[location].acquire()

            script_data = []
            # Gather data from neighbours.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from self.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Run the script.
                result = script.run(script_data)

                # Broadcast the result.
                for device in neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)

            # Release the lock.
            self.device.locks[location].release()

            # Signal that this task is done.
            self.queue.task_done()

    def shutdown(self):
        """Waits for all worker threads to terminate."""
        for i in xrange(8):
            self.thread_list[i].join()