# -*- coding: utf-8 -*-
"""
Models a distributed system of concurrent devices using a robust ThreadPool
and a producer-consumer architecture.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

# The file imports ReusableBarrierCond, but the class defined and used later
# appears to be ReusableBarrierSem from other examples. We assume the intent
# is to use a reusable barrier for synchronization.
# from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a device in the simulation, managing its state and interaction
    with a ThreadPool for script execution.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device and its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)
        self.thread.start()

        # --- Shared Synchronization Primitives ---
        self.barrier = None
        self.devices_synchronized = Event()
        # A shared dictionary mapping a location to a Semaphore, acting as a lock.
        self.location_semaphores = {}
        self.scripts_lock = Lock()
        # A lock to protect the creation of new location semaphores.
        self.new_location_lock = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources (barrier, semaphores) for all devices.
        Device 0 acts as the leader for this one-time setup.
        """
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            location_semaphores = {}
            new_location_lock = Lock()
            
            # Distribute shared objects to all devices.
            for device in devices:
                device.barrier = barrier
                device.location_semaphores = location_semaphores
                device.new_location_lock = new_location_lock
                # Signal that this device's setup is complete.
                device.devices_synchronized.set()
        
        # All devices wait here until setup is done.
        self.devices_synchronized.wait()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. This method is thread-safe and can be
        called by an external supervisor thread.
        """
        # Atomically check for and create a semaphore for a new location.
        with self.new_location_lock:
            if location not in self.location_semaphores:
                self.location_semaphores[location] = Semaphore()

        # Add the script to the list and notify the device thread.
        with self.scripts_lock:
            if script is not None:
                self.scripts.append((script, location))
            else: # A None script is the sentinel for the end of the timepoint.
                self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread (Producer) for a device."""

    def __init__(self, device):
        """Initializes the thread and its personal ThreadPool."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.executor = ThreadPool(8)

    def run(self):
        """
        Main simulation loop. It submits tasks to the thread pool and waits
        for timepoint completion.
        """
        self.executor.start_workers()
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                self.executor.shutdown()
                break
            
            script_done = {}

            # This complex loop handles "live" assignment of scripts during a timepoint.
            with self.device.scripts_lock:
                while not self.device.timepoint_done.isSet():
                    # Wait for a new script to be assigned by the supervisor.
                    self.device.script_received.wait()
                    self.device.script_received.clear()

                    # Submit any new, un-processed scripts to the thread pool.
                    for (script, location) in self.device.scripts:
                        if (script, location) not in script_done:
                            self.executor.submit((self.device, neighbours, script, location))
                            script_done[(script, location)] = True
            
            # After the timepoint is marked as done, wait for all submitted tasks to finish.
            self.executor.wait_all()
            
            # Synchronize with all other devices before the next time step.
            self.device.barrier.wait()
            
            # Reset for the next timepoint.
            self.device.scripts = []
            self.device.timepoint_done.clear()


class ThreadPool(object):
    """A simple and reusable thread pool implementation."""
    def __init__(self, num_threads):
        """Initializes the queue and workers."""
        self._queue = Queue()
        self._num_threads = num_threads
        self._workers = [WorkerThread(self._queue) for _ in xrange(self._num_threads)]

    def submit(self, args):
        """Adds a task to the work queue."""
        self._queue.put(args)

    def wait_all(self):
        """Blocks until all tasks in the queue are processed."""
        self._queue.join()

    def start_workers(self):
        """Starts all worker threads."""
        for worker in self._workers:
            worker.start()

    def shutdown(self):
        """Stops all worker threads gracefully."""
        for _ in self._workers:
            self._queue.put(None) # Sentinel value to stop workers.
        for worker in self._workers:
            worker.join()


class WorkerThread(Thread):
    """A consumer thread that executes tasks from a queue."""
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        """Continuously gets and processes tasks from the queue."""
        while True:
            task = self.queue.get()
            if task is None: # Shutdown sentinel.
                self.queue.task_done()
                break
            
            current_device, neighbours, script, location = task
            
            # Acquire the global semaphore for this location to ensure exclusive access.
            current_device.location_semaphores[location].acquire()

            script_data = []
            # Aggregate data from neighbors and self.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            data = current_device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Execute script and disseminate results.
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                current_device.set_data(location, result)

            # Release the semaphore and signal that the task is complete.
            current_device.location_semaphores[location].release()
            self.queue.task_done()


# A placeholder for the ReusableBarrier class presumably defined in 'barrier.py'.
class ReusableBarrierCond:
    def __init__(self, num_threads):
        pass
    def wait(self):
        pass
