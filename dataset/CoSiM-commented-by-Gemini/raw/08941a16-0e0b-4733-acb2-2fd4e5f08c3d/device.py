"""
This module defines a simulated distributed device network using a two-level
master-worker threading model. A global barrier synchronizes all devices
at each time step, while within each device, a master thread dispatches
tasks to a pool of worker threads.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue, Empty


class ReusableBarrier(object):
    """
    A custom barrier implementation for thread synchronization.

    NOTE: This implementation is a classic, but flawed, reusable barrier. It is
    susceptible to a race condition where a fast thread can loop around and enter
    the barrier again before all slower threads have woken up from the previous
    wait, leading to deadlocks. A correct reusable barrier requires two phases.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have called this method.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread arrives, wakes up all waiting threads.
            self.cond.notify_all()
            # Reset for next use.
            self.count_threads = self.num_threads
        else:
            # Wait to be notified by the last thread.
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    Represents a single device, which contains its own master-worker thread pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.neighbours = []

        self.barrier = None
        # Fine-grained locks for specific data locations.
        self.locks = []
        # Supervisor signals this event when scripts are assigned for the time step.
        self.timepoint_done = Event()
        # Master thread signals this to workers when tasks are available in the queue.
        self.tasks_ready = Event()
        # A queue for the master to dispatch tasks to its workers.
        self.tasks = Queue()
        self.simulation_ended = False

        # Each device has one master thread.
        self.master = DeviceThreadMaster(self)
        self.master.start()

        # Each device has a pool of 8 worker threads.
        self.workers = []
        for i in xrange(8):
            worker = DeviceThreadWorker(self, i)
            self.workers.append(worker)
            worker.start()

    def __str__(self):
        return "Device [%d]" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects (barrier, locks).
        This is a centralized setup performed by device 0.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            locks = [Lock() for _ in xrange(24)]
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script signals the end of assignments.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal the master thread that all scripts for the step are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the master and all worker threads to terminate."""
        self.master.join()
        for worker in self.workers:
            worker.join()


class DeviceThreadMaster(Thread):
    """
    The master thread within a device. It coordinates its local worker threads
    and synchronizes with other devices on the global barrier.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device [%d] Thread Master" % device.device_id)
        self.device = device

    def run(self):
        """The main control loop for the master thread."""
        while True:
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                # Supervisor signals end of simulation.
                self.device.simulation_ended = True
                self.device.tasks_ready.set() # Wake up workers to let them terminate.
                break

            # 1. Wait for supervisor to finish assigning scripts for this step.
            self.device.timepoint_done.wait()

            # 2. Populate the task queue for the worker threads.
            for task in self.device.scripts:
                self.device.tasks.put(task)

            # 3. Signal workers that the queue is ready.
            self.device.tasks_ready.set()

            # 4. Wait for all tasks in the queue to be processed by workers.
            self.device.tasks.join()

            # 5. Clear events for the next time step.
            self.device.tasks_ready.clear()
            self.device.timepoint_done.clear()

            # 6. Wait at the global barrier for all other devices to finish this step.
            self.device.barrier.wait()


class DeviceThreadWorker(Thread):
    """
    A worker thread that executes tasks from a shared queue within a device.
    """

    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device [%d] Thread Worker [%d]" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main loop for a worker thread."""
        while not self.device.simulation_ended:
            # Wait for the master to signal that tasks are ready.
            self.device.tasks_ready.wait()

            try:
                # Fetch a task from the queue without blocking.
                script, location = self.device.tasks.get(block=False)

                # Acquire the specific lock for the data location.
                self.device.locks[location].acquire()

                script_data = []
                # Gather data from neighbors.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                # Gather data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if len(script_data) > 0:
                    # Execute the script.
                    result = script.run(script_data)
                    # Propagate the result to neighbors and self.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the location-specific lock.
                self.device.locks[location].release()
                # Signal that this task is done.
                self.device.tasks.task_done()
            except Empty:
                # The queue is empty for this worker in this time step.
                pass
