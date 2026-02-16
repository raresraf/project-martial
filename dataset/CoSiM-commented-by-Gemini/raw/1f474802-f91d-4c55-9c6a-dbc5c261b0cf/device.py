"""
This module implements a distributed device simulation where each device has its
own private thread pool.

The architecture has several significant design flaws. Most critically, it uses a
dangerous locking protocol where a lock is acquired in `get_data` and released
in `set_data`, creating a high risk of deadlocks. The main dispatcher loop in
`DeviceThread` is a convoluted and buggy state machine.
"""

from threading import Event, Thread, Lock, Semaphore
from Queue import Queue

class ReusableBarrierSem(object):
    """
    A correct, two-phase reusable barrier for thread synchronization.
    This is included here as `barrier.Barrier` is imported by the original code.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        self.phase1()
        self.phase2()

    def phase1(self):
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

# Alias Barrier to the correct implementation.
Barrier = ReusableBarrierSem


class Device(object):
    """
    Represents a device with its own set of locks and a dispatcher thread.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        # Each device has its own private dictionary of locks for its data locations.
        self.locks = {location : Lock() for location in sensor_data}

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Decentralized setup where the master device (id 0) creates the barrier."""
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to be run."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        DANGEROUS LOCKING: Acquires a lock but does not release it.
        This method is part of a broken locking protocol that is prone to deadlock.
        """
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        DANGEROUS LOCKING: Releases a lock that it did not acquire.
        This is the counterpart to the flawed get_data method.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """
    The main dispatcher thread for a device. Manages a private thread pool.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_pool = ThreadPool(8)

    def run(self):
        """
        The main loop, containing a confusing state machine for dispatching tasks.
        """
        # Poor practice: The pool's 'device' attribute is coupled to and set by this thread.
        self.thread_pool.device = self.device
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # --- BUGGY STATE MACHINE ---
            # This inner loop is a confusing and race-prone way to dispatch tasks.
            # It can lead to dispatching the same scripts multiple times.
            while True:
                if self.device.script_received.isSet():
                    self.device.script_received.clear()
                    for (script, location) in self.device.scripts:
                        self.thread_pool.queue.put((neighbours, script, location))
                
                elif self.device.timepoint_done.wait():
                    if self.device.script_received.isSet():
                        self.device.script_received.clear()
                        for (script, location) in self.device.scripts:
                            self.thread_pool.queue.put((neighbours, script, location))
                    else:
                        self.device.timepoint_done.clear()
                        self.device.script_received.set()
                        break
            
            # Wait for all tasks in the queue to be processed.
            self.thread_pool.queue.join()
            # Synchronize with other devices.
            self.device.barrier.wait()
        
        # Shutdown sequence for the thread pool.
        self.thread_pool.queue.join()
        for _ in xrange(len(self.thread_pool.threads)):
            self.thread_pool.queue.put((None, None, None))
        for thread in self.thread_pool.threads:
            thread.join()


class ThreadPool(object):
    """
    A thread pool implementation that processes tasks from a queue.
    """
    def __init__(self, num_threads):
        self.queue = Queue(num_threads)
        self.threads = []
        # Tightly coupled: relies on an external 'device' object to be set.
        self.device = None
        self.workers(num_threads)
        self.start_workers()

    def workers(self, num_threads):
        """Creates the worker threads."""
        for _ in xrange(num_threads):
            new_thread = Thread(target=self.run)
            self.threads.append(new_thread)

    def start_workers(self):
        """Starts the worker threads."""
        for thread in self.threads:
            thread.start()

    def run(self):
        """The main logic for each worker thread."""
        while True:
            # Blocks until a task is available.
            neighbours, script, location = self.queue.get()
            
            # Poison pill to terminate the thread.
            if neighbours is None and script is None:
                self.queue.task_done()
                return
            
            script_data = []
            # Block Logic: Call the flawed get/set methods to perform work.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = script.run(script_data)
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                self.device.set_data(location, result)
            
            self.queue.task_done()
