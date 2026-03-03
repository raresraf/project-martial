"""
This module implements a distributed device simulation using a persistent
worker pool for each device and a custom, centralized barrier for synchronization.

Architectural Overview:
- `Device`: Represents a device node. Device 0 acts as a "lead device" that
  manages a centralized barrier for all other devices.
- `DeviceThread`: The main control thread for a device. It manages a
  `WorkerThreadPool`.
- `WorkerThreadPool`: A class that creates and manages a fixed-size pool of
  persistent `SimpleWorker` threads. It maintains a queue of idle workers and
  dispatches tasks to them.
- `SimpleWorker`: A reusable worker thread that waits for a task, executes it,
  and then reports back to the pool to become available for more work.
- Synchronization:
  - A manual barrier is implemented in `Device.notify_finish` using a
    `Condition` variable owned by the lead device.
  - A `Semaphore` is used within the `WorkerThreadPool` to manage the
    availability of idle workers.
  - Fine-grained `Lock`s are used to protect access to data at specific locations.
"""

from threading import Event, Thread, Lock, Condition, Semaphore
from Queue import Queue


class Device(object):
    """
    Represents a device in the simulation. Manages a worker pool and
    participates in a centralized, manually-implemented barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.devices = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        self.lead_device_index = -1
        self.location_locks = []

        # Attributes specific to the lead device (device 0).
        if device_id == 0:
            self.threads_that_finished_no = 0
            # The condition variable used to implement a centralized barrier.
            self.next_time_point_cond = Condition()
            self.can_start = Event()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources. The lead device (device 0) creates and
        distributes a global set of location-based locks.
        """
        self.devices = devices
        # Find the index of the lead device.
        for i in xrange(len(self.devices)):
            if devices[i].device_id == 0:
                self.lead_device_index = i
                break

        if self.device_id == 0:
            self.can_start.clear()
            # The lead device determines the max location and creates a lock for each.
            max_lock = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > max_lock:
                        max_lock = location
            self.location_locks = [Lock() for _ in range(max_lock + 1)]
            # Distribute the list of locks to all other devices.
            for device in devices:
                device.location_locks = self.location_locks
            # Signal that setup is complete.
            self.can_start.set()
        else:
            # Non-lead devices wait for the setup to complete.
            devices[self.lead_device_index].can_start.wait()

    def assign_script(self, script, location):
        """Assigns a script to the device's workload."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that all scripts for this step have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()

    def notify_finish(self):
        """
        Implements a centralized barrier. Each device notifies the lead device
        that it has finished its step.
        """
        lead_device = self.devices[self.lead_device_index]
        with lead_device.next_time_point_cond:
            lead_device.threads_that_finished_no += 1
            if lead_device.threads_that_finished_no == len(self.devices):
                # This is the last device to arrive; notify all waiting devices.
                lead_device.threads_that_finished_no = 0
                lead_device.next_time_point_cond.notifyAll()
            else:
                # Not the last device; wait to be notified.
                lead_device.next_time_point_cond.wait()

class DeviceThread(Thread):
    """The main control thread for a device."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = WorkerThreadPool(device)

    def run(self):
        """The main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                self.thread_pool.shutdown()
                break

            # Wait for the supervisor to signal all scripts are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Dispatch all assigned scripts to the worker pool.
            for (script, location) in self.device.scripts:
                self.thread_pool.do_work(script, location, neighbours)

            # Wait for all dispatched work to be completed by the pool.
            self.thread_pool.wait_to_finish_work()
            
            # Participate in the centralized barrier to end the time step.
            self.device.notify_finish()

class WorkerThreadPool(object):
    """
    Manages a persistent pool of reusable SimpleWorker threads for a device.
    """
    def __init__(self, device):
        self.device = device
        self.work_finished_event = Event()
        self.work_finished_event.set()
        self.worker_pool = []
        self.ready_for_work_queue = []
        # Semaphore acts as a counter for available workers.
        self.read_to_work_thread_sem = Semaphore(8)
        self.queue_lock = Lock()
        
        for _ in xrange(8):
            thread = SimpleWorker(self, self.device)
            self.worker_pool.append(thread)
            self.ready_for_work_queue.append(thread)
            thread.start()

    def do_work(self, script, location, neighbours):
        """Dispatches a task to an available worker."""
        if self.work_finished_event.isSet():
            self.work_finished_event.clear()
        
        self.read_to_work_thread_sem.acquire() # Wait for an available worker.
        with self.queue_lock:
            worker = self.ready_for_work_queue.pop(0)
        worker.do_work(script, location, neighbours) # Assign work to the worker.

    def shutdown(self):
        """Shuts down all worker threads in the pool."""
        for worker in self.worker_pool:
            worker.should_i_stop = True
            worker.data_for_work_ready.release() # Wake up a waiting worker to exit.
        for worker in self.worker_pool:
            worker.join()

    def worker_finished(self, worker):
        """Called by a worker when it finishes a task."""
        with self.queue_lock:
            self.ready_for_work_queue.append(worker)
            # If all workers are now idle, set the event.
            if len(self.ready_for_work_queue) == 8 and not self.work_finished_event.isSet():
                self.work_finished_event.set()
        self.read_to_work_thread_sem.release() # Signal that a worker is now available.

    def wait_to_finish_work(self):
        """Blocks until all workers in the pool are idle."""
        self.work_finished_event.wait()

class SimpleWorker(Thread):
    """A reusable worker thread that executes tasks given by its pool."""
    def __init__(self, worker_pool, device):
        Thread.__init__(self)
        self.worker_pool = worker_pool
        self.should_i_stop = False
        self.data_for_work_ready = Semaphore(0) # Semaphore used as a binary event.
        self.device = device
        self.script = None
        self.location = None
        self.neighbours = None

    def do_work(self, script, location, neighbours):
        """Receives a task from the pool and signals itself to start."""
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.data_for_work_ready.release() # Wake up the run() method.

    def run(self):
        """Main loop for the worker. Waits for work, executes it, and repeats."""
        while True:
            self.data_for_work_ready.acquire() # Block until work is assigned.

            if self.should_i_stop:
                break
            
            # Execute the script with fine-grained locking on the location.
            with self.device.location_locks[self.location]:
                script_data = []
                for device in self.neighbours:
                    data = device.get_data(self.location)
                    if data is not None:
                        script_data.append(data)
                data = self.device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    result = self.script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                    self.device.set_data(self.location, result)
            
            # Notify the pool that this worker is finished and available.
            self.worker_pool.worker_finished(self)
