# -*- coding: utf-8 -*-
"""
This module implements a simulation of a distributed sensor network.

Each device in this simulation has its own local thread pool (`WorkPool`)
for executing computational scripts. Devices synchronize globally using a
shared barrier. This architecture combines local concurrency with global
synchronization.

Classes:
    Device: Represents a device with a local thread pool.
    DeviceThread: The main control thread for a Device.
    ScriptExecutor: A worker thread within a Device's WorkPool.
    WorkPool: A thread pool local to a single Device.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem
from workPool import WorkPool

class Device(object):
    """
    Represents a device in the sensor network, with its own local thread pool.

    Attributes:
        device_id (int): A unique identifier for the device.
        workpool (WorkPool): A local thread pool for executing scripts.
        ... and other attributes for managing state and synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor (Supervisor): The supervisor for this device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.workpool = WorkPool(4, self)

        self.scripts = []
        self.script_storage = []
        self.locks = []
        self.barrier = None
        self.neighbours = None

        self.script_lock = Lock()
        self.supervisor_interact = Event()
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources for all devices.

        The device with the minimum ID acts as the coordinator to create and
        distribute the global barrier and location locks.
        """
        ids = []
        loc = []

        for device in devices:
            ids.append(device.device_id)
            for location, _ in device.sensor_data.iteritems():
                loc.append(location)

        max_locations = max(loc) + 1
        if self.device_id == min(ids):
            barrier = ReusableBarrierSem(len(ids))
            locks = [Lock() for _ in range(max_locations)]
            for device in devices:
                device.assign_barrier(barrier)
                device.set_locks(locks)

    def assign_barrier(self, barrier):
        """Assigns the global barrier to this device."""
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script (Script): The script to execute.
            location (str): The sensor data location for the script.
        """
        if script is not None:
            self.script_lock.acquire()
            self.scripts.append((script, location))
            self.script_lock.release()
        else:
            self.timepoint_done.set()
        self.supervisor_interact.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location.
        """
        return self.sensor_data.get(location)

    def wait_on_scripts(self):
        """Waits for the local workpool to finish all its tasks."""
        self.workpool.wait()

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

    def set_locks(self, locks):
        """Assigns the shared location locks to this device."""
        self.locks = locks

    def lock(self, location):
        """Acquires the lock for a specific location."""
        self.locks[location].acquire()

    def unlock(self, location):
        """Releases the lock for a specific location."""
        self.locks[location].release()

    def execute_scripts(self):
        """
        Moves scripts from the incoming queue to the workpool for execution.
        """
        self.script_lock.acquire()
        for (script, location) in self.scripts:
            self.script_storage.append((script, location))
            self.workpool.add_data(script, location)
        del self.scripts[:]
        self.script_lock.release()

class DeviceThread(Thread):
    """
    The main control thread for a Device.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop of the device thread.

        It coordinates fetching neighbor data, executing scripts via the
        local workpool, and synchronizing with other devices at the end of
        a timepoint.
        """
        while True:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                self.device.workpool.shutdown()
                return
            
            self.device.execute_scripts()

            while True:
                # Wait for a signal from the supervisor (e.g., more scripts).
                self.device.supervisor_interact.wait()
                self.device.supervisor_interact.clear()

                # If new scripts have arrived, execute them.
                self.device.script_lock.acquire()
                if len(self.device.scripts) > 0:
                    self.device.script_lock.release()
                    self.device.execute_scripts()
                else:
                    self.device.script_lock.release()

                # If the timepoint is marked as done, finalize and synchronize.
                if self.device.timepoint_done.is_set():
                    self.device.timepoint_done.clear()
                    
                    if len(self.device.scripts) > 0:
                        self.device.execute_scripts()

                    # Wait for the local workpool to finish.
                    self.device.wait_on_scripts()

                    # Wait on the global barrier for all devices to finish.
                    self.device.barrier.wait()
                    self.device.scripts = self.device.script_storage
                    self.device.script_storage = []
                    break

from threading import Thread

class ScriptExecutor(Thread):
    """
    A worker thread that executes scripts from a WorkPool.
    """

    def __init__(self, index, workpool, device):
        Thread.__init__(self, name="Worker Thread %d" % index)
        self.index = index
        self.workpool = workpool
        self.device = device

    def run(self):
        while True:
            # Wait for a task to be available in the workpool.
            self.workpool.data.acquire()
            if self.workpool.done:
                return

            (script, location) = self.workpool.q.get()

            if self.device.neighbours is None:
                return

            self.device.lock(location)

            # Gather data from neighbors and the local device.
            script_data = []
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                result = script.run(script_data)
                # Update data on neighbors and the local device.
                for device in self.device.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)

            self.device.unlock(location)
            self.workpool.q.task_done()

from threading import Semaphore
from Queue import Queue
from scriptexecutor import ScriptExecutor

class WorkPool(object):
    """
    A thread pool local to a single Device for executing scripts.
    """

    def __init__(self, num_threads, device):
        self.device = device
        self.executors = []
        self.q = Queue()
        self.data = Semaphore(0)
        self.done = False

        for i in range(num_threads + 1):
            executor = ScriptExecutor(i, self, self.device)
            executor.start()
            self.executors.append(executor)

    def add_data(self, script, location):
        """Adds a script execution task to the queue."""
        self.q.put((script, location))
        self.data.release()

    def wait(self):
        """Blocks until all tasks in the queue are completed."""
        if not self.done:
            self.q.join()

    def shutdown(self):
        """Shuts down the work pool and all its worker threads."""
        self.wait()
        self.done = True

        for _ in self.executors:
            self.data.release()

        for executor in self.executors:
            executor.join()
