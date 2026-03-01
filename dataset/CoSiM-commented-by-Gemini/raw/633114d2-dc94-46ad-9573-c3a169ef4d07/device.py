"""
@file device.py
@brief A distributed device simulation using a persistent master-worker thread model.

This script models a network of devices operating in synchronized time steps.
Each device has a dedicated "master" thread for coordination and a fixed pool of
"worker" threads for task execution. The master dispatches tasks via a shared
queue, and synchronization is handled by a global barrier and a global set of
location-based locks. This architecture is robust and efficient.
"""

from threading import Event, Thread, Lock, Condition
from Queue import Queue, Empty


class ReusableBarrier(object):
    """A classic reusable barrier implementation for thread synchronization."""

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Blocks until the required number of threads have called wait()."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    Represents a node in the network, containing one master thread and a pool
    of worker threads that it manages.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.neighbours = []

        # Shared resources for synchronization and task management.
        self.barrier = None
        self.locks = []
        self.timepoint_done = Event()
        self.tasks_ready = Event()
        self.tasks = Queue()
        self.simulation_ended = False

        # Create and start the master and worker threads upon initialization.
        self.master = DeviceThreadMaster(self)
        self.master.start()

        self.workers = []
        for i in xrange(8):
            worker = DeviceThreadWorker(self, i)
            self.workers.append(worker)
            worker.start()

    def __str__(self):
        return "Device [%d]" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (global locks and barrier)
        for the entire simulation, orchestrated by device 0.
        """
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            # Note: A hardcoded number of 24 locations is assumed.
            locks = [Lock() for _ in xrange(24)]
            for device in devices:
                device.barrier = barrier
                device.locks = locks

    def assign_script(self, script, location):
        """Assigns a script to the device for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of assignments.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its master and worker threads."""
        self.master.join()
        for worker in self.workers:
            worker.join()


class DeviceThreadMaster(Thread):
    """
    The master thread for a device, responsible for coordinating time-steps
    and dispatching tasks to its worker pool.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device [%d] Thread Master" % device.device_id)
        self.device = device

    def run(self):
        """Main control loop for managing time-steps."""
        while True:
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                # Shutdown sequence
                self.device.simulation_ended = True
                self.device.tasks_ready.set() # Unblock workers to let them terminate.
                break

            # Wait for the supervisor to signal that all scripts for the timepoint are assigned.
            self.device.timepoint_done.wait()

            # Put all script tasks for this timepoint onto the shared queue for workers.
            for task in self.device.scripts:
                self.device.tasks.put(task)

            # Signal to worker threads that the queue is populated for this timepoint.
            self.device.tasks_ready.set()

            # Block and wait until workers have processed all tasks in the queue.
            self.device.tasks.join()

            # Reset events for the next timepoint.
            self.device.tasks_ready.clear()
            self.device.timepoint_done.clear()

            # Wait at the global barrier to synchronize with all other devices.
            self.device.barrier.wait()


class DeviceThreadWorker(Thread):
    """
    A worker thread that executes tasks from a shared queue for its parent device.
    """
    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device [%d] Thread Worker [%d]" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """Main loop for a worker thread."""
        while not self.device.simulation_ended:
            # Wait for the master to signal that tasks are ready for the current timepoint.
            self.device.tasks_ready.wait()

            if self.device.simulation_ended:
                break

            try:
                # Fetch a task from the queue in a non-blocking way.
                script, location = self.device.tasks.get(block=False)

                # Acquire the global lock for this location to ensure mutual exclusion.
                self.device.locks[location].acquire()

                script_data = []
                # Aggregate data from neighbors.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Aggregate data from self.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execute script and disseminate results if data was found.
                if len(script_data) > 0:
                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the global location lock.
                self.device.locks[location].release()

                # Signal that this task is complete.
                self.device.tasks.task_done()
            except Empty:
                # Queue is empty for this worker; it will wait on tasks_ready again.
                pass
