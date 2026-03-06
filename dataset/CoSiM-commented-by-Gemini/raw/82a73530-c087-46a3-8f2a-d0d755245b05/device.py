"""
This module implements a device simulation using a thread pool pattern. Each
`Device` has a `DeviceThread` controller that manages a pool of persistent worker
threads.

The architecture uses a central task queue (`Queue.Queue`) per device to
distribute work to its pool. Synchronization is managed by a global reusable
barrier and a shared set of location-specific locks.

WARNING: The method for initializing and distributing the location-specific locks
in the `assign_script` method is not thread-safe. It creates a race condition
where multiple devices could attempt to create and distribute a lock for the same
new location concurrently, leading to unpredictable behavior. Shared resources
like these should be initialized centrally before worker threads begin processing.
"""

from threading import Lock, Semaphore, Thread, Event
from Queue import Queue

class RBarrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads, implemented
    using a two-phase protocol with semaphores.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # A list is used to create a mutable integer reference for the counter.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier protocol."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """Represents a device and its associated resources."""
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.script_received = Event()
        self.timepoint_done = Event()
        
        # --- Shared and state resources ---
        self.location_lock = {}
        self.barrier = None
        self.all_devices = []
        self.update = Lock() # A globally shared lock.
        self.got_data = False # Flag for complex synchronization logic.

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes some shared resources."""
        self.all_devices = devices
        if self.device_id == 0:
            self.barrier = RBarrier(len(self.all_devices))
            self.update = Lock()
            # Distribute the barrier and global update lock to all devices.
            for device in self.all_devices:
                device.barrier = self.barrier
                device.update = self.update

    def assign_script(self, script, location):
        """
        Assigns a script and performs a racy, on-the-fly initialization
        of location-specific locks.
        """
        if script is not None:
            # RACE CONDITION: If a new location is received by two devices at once,
            # this block is not safe. Both might try to create and distribute the lock.
            if location not in self.location_lock:
                self.got_data = True
                self.location_lock[location] = Lock()
                with self.update:
                    for device in self.all_devices:
                        device.location_lock[location] = self.location_lock[location]

            if location in self.location_lock:
                self.scripts.append((script, location))
                self.script_received.set() # This event seems unused.
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. Not thread-safe on its own."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data. Not thread-safe on its own."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

class DeviceThread(Thread):
    """
    A controller thread that manages a pool of worker threads for a single device.
    """
    THREAD_NUMBER = 8
    STOP_FLAG = "STOP"

    def __init__(self, device):
        """Initializes the controller and its thread pool."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = Queue(self.THREAD_NUMBER)
        self.threads = [Thread(target=self.thread_func) for _ in range(self.THREAD_NUMBER)]
        for t in self.threads:
            t.start()

    def run(self):
        """The main simulation loop for the device controller."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Main termination signal.
            
            # This inner loop has confusing logic controlled by 'got_data',
            # likely an attempt to handle the first timepoint differently from others.
            while True:
                self.device.timepoint_done.wait()
                if not self.device.got_data:
                    self.device.timepoint_done.clear()
                    self.device.got_data = True
                    break
                else:
                    # Place all scripts for the current timepoint onto the queue.
                    for script_tuple in self.device.scripts:
                        self.queue.put((neighbours, script_tuple[0], script_tuple[1]))
                    self.device.got_data = False

            # Wait for all worker threads to finish processing the tasks in the queue.
            self.queue.join()
            # Synchronize with all other DeviceThreads before the next timepoint.
            self.device.barrier.wait()

        # --- Termination Phase ---
        # Signal all worker threads to stop and wait for them to terminate.
        self.queue.join()
        for _ in range(self.THREAD_NUMBER):
            self.queue.put((self.STOP_FLAG, self.STOP_FLAG, self.STOP_FLAG))
        for t in self.threads:
            t.join()

    def thread_func(self):
        """
        The function executed by each worker thread in the pool.
        """
        # The worker's main loop: get a task, process it, repeat.
        while True:
            neighbours, script, location = self.queue.get()
            if neighbours is self.STOP_FLAG:
                self.queue.task_done()
                break
            
            # Use the shared, location-specific lock to ensure data consistency.
            with self.device.location_lock[location]:
                # Data aggregation, protected by the location lock.
                script_data = []
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Computation and write-back, also protected by the location lock.
                if script_data:
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
            
            # Signal that this task is complete.
            self.queue.task_done()
