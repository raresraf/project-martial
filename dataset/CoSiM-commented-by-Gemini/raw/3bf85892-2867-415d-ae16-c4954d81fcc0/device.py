"""A distributed device simulation with critical concurrency flaws.

This module implements a simulation of devices using a queue-based worker
pool for task execution.

WARNING: This implementation contains multiple severe concurrency flaws. The
custom `ReusableBarrier` is not correctly implemented and is prone to race
conditions. More critically, the main `DeviceThread` does not wait for its
worker threads to finish their tasks before proceeding to the barrier, which
breaks the core synchronization logic of the simulation.
"""

from threading import Event, Thread, Lock
from Queue import Queue
# This local import suggests a project structure that was not preserved.
from reusable_barrier_condition import ReusableBarrier


class Device(object):
    """Represents a single device in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device, its worker queue, and its main thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.location_locks = {}
        self.barrier = None
        self.num_threads = 8
        self.queue = Queue(self.num_threads)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared barrier and locks.
        
        WARNING: This setup protocol is racy. The first device to execute this
        block creates the shared resources and distributes them. There is no
        guaranteed master or execution order, which can lead to non-deterministic
        initialization of the shared `barrier` and `location_locks`.
        """
        if self.barrier is None:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.barrier = self.barrier
                for location in device.sensor_data:
                    if location not in self.location_locks:
                        self.location_locks[location] = Lock()
            for device in devices:
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """Adds a script to the workload for the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets data from a specific location (not thread-safe)."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets data at a specific location (not thread-safe)."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully shuts down the device."""
        self.thread.join()

class WorkerThread(Thread):
    """A worker thread that processes script execution tasks from a queue."""

    def __init__(self, queue, device):
        Thread.__init__(self)
        self.queue = queue
        self.device = device

    def run(self):
        """The main loop for the worker thread.
        
        Continuously fetches tasks from the queue and executes them. A sentinel
        value of (None, None, None) signals the thread to terminate.
        """
        while True:
            data_tuple = self.queue.get()

            if data_tuple == (None, None, None):
                break

            script, location, neighbours = data_tuple
            
            # This correctly uses a lock to make the script execution atomic
            # for a given location.
            with self.device.location_locks[location]:
                script_data = []
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)


class DeviceThread(Thread):
    """The main control thread for a single Device."""

    def __init__(self, device):
        """Initializes the thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main, flawed, lifecycle loop for the device.
        
        WARNING: This method contains a critical race condition. It puts tasks
        on the queue for its workers but immediately proceeds to the barrier
        `wait()` without waiting for the workers to complete. The purpose of
        the barrier (to wait for all work in a time step to be done) is defeated.
        """
        threads = []
        for i in range(self.device.num_threads):
            thread = WorkerThread(self.device.queue, self.device)
            threads.append(thread)
            threads[i].start()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # 1. Wait for the signal to start the time step.
            self.device.timepoint_done.wait()

            # 2. Add all script tasks to the worker queue.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location, neighbours))

            # 3. CRITICAL FLAW: Proceeds to the barrier without waiting for
            #    the worker queue to be processed (e.g., via `queue.join()`).
            self.device.barrier.wait()
            self.device.timepoint_done.clear()

        # Shutdown sequence for the worker pool.
        for i in range(self.device.num_threads):
            self.device.queue.put((None, None, None))
        for i in range(self.device.num_threads):
            threads[i].join()


from threading import Condition

class ReusableBarrier(object):
    """A buggy implementation of a reusable barrier using a Condition variable.
    
    WARNING: This implementation is not safe. It is susceptible to race
    conditions, particularly if threads that are released from a `wait()` call
    loop around and call `wait()` again before all other threads have woken up
    from the first call. This can lead to a premature `notify_all()` or cause
    threads to miss a notification, resulting in deadlock.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()


    def wait(self):
        """Causes a thread to block until all threads have called wait."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread arrives: notify all others and reset the counter.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Not the last thread: wait to be notified.
            self.cond.wait()
        self.cond.release()
