"""
Defines a hierarchical, manager-worker threading model for a sensor network simulation.

This module features a `Device` class whose main `DeviceThread` acts as a
dispatcher, managing a pool of stateful `Worker` threads. Communication between
the manager and workers is handled via a command-passing system using `threading.Event`.
The file also includes a `ReusableBarrier` for synchronization and a custom
reader-writer lock implementation for data access.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    A reusable barrier for a fixed number of threads, using a two-phase protocol.
    """
    def __init__(self, num_threads):
        """Initializes the barrier for `num_threads`."""
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Internal logic for one phase of the barrier."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                i = 0
                while i < self.num_threads:
                    threads_sem.release()
                    i += 1
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """
    Represents a device node with a manager thread and a pool of worker threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device, its threads, and synchronization primitives."""
        self.max_threads = 8
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.notification = Event()
        self.timepoint_done = Event()
        self.notification.clear()
        self.timepoint_done.clear()
        # Implements a reader-writer lock for each location.
        self.update_locks = {} # The 'write' lock.
        self.read_locations = {} # The 'read' event flag.
        self.external_barrier = None # Syncs with other Devices.
        self.internal_barrier = ReusableBarrier(self.max_threads) # Syncs own workers.
        self.workers = self.setup_workers()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_workers(self):
        """Creates the pool of worker threads for this device."""
        workers = []
        i = 0
        while i < self.max_threads:
            workers.append(Worker(self))
            i += 1
        return workers

    def start_workers(self):
        """Starts all worker threads in the pool."""
        for i in range(0, self.max_threads):
            self.workers[i].start()

    def stop_workers(self):
        """Joins all worker threads, waiting for them to complete."""
        for i in range(0, self.max_threads):
            self.workers[i].join()

    def setup_devices(self, devices):
        """
        Sets up shared resources across all devices.

        Device 0 creates the global barrier; other devices poll in a busy-wait
        loop to get a reference to it.
        """
        if self.device_id == 0:
            self.external_barrier = ReusableBarrier(len(devices))
        else:
            for device in devices:
                if device.device_id == 0:
                    # This is a busy-wait, which is inefficient.
                    while device.external_barrier is None:
                        pass
                    self.external_barrier = device.external_barrier
                    break

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals the end of a timepoint.
        """
        self.notification.set() # Notify manager thread a script has arrived.
        if script is not None:
            # Lazily initialize locking primitives for the new location.
            if location not in self.update_locks:
                self.update_locks[location] = Lock()
                self.read_locations[location] = Event()
                self.read_locations[location].set()
            self.scripts.append((script, location))
        else:
            # A None script signals the end of script assignment for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Safely reads data from a location, respecting the reader-writer lock.
        Blocks if a write is in progress.
        """
        if location not in self.sensor_data:
            return None
        else:
            # Lazily initialize locks if accessed for the first time via get.
            if location not in self.read_locations:
                self.update_locks[location] = Lock()
                self.read_locations[location] = Event()
                self.read_locations[location].set()
            # Wait until no write is in progress.
            self.read_locations[location].wait()
            return self.sensor_data[location]

    def set_data(self, location, data):
        """
        Safely writes data to a location using a reader-writer lock pattern.
        Blocks other readers and writers for the same location.
        """
        if location in self.sensor_data:
            self.update_locks[location].acquire() # Acquire write lock.
            self.read_locations[location].clear() # Block readers.
            self.sensor_data[location] = data
            self.read_locations[location].set() # Allow readers again.
            self.update_locks[location].release() # Release write lock.

    def shutdown(self):
        """Stops all worker threads and the main manager thread."""
        self.stop_workers()
        self.thread.join()


class DeviceThread(Thread):
    """
    The manager thread that dispatches jobs to a pool of `Worker` threads.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def find_free_worker(self):
        """Finds an available worker thread in the pool."""
        for i in range(0, self.device.max_threads):
            if self.device.workers[i].is_free:
                return i
        return -1

    def run(self):
        """Main loop for the manager/dispatcher thread."""
        self.device.start_workers()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Shutdown signal
                for i in range(0, self.device.max_threads):
                    self.device.workers[i].update(None, None, None, "end")
                break

            # Wait for the first script to arrive for the current timepoint.
            if len(self.device.scripts) == 0:
                self.device.notification.wait()

            # Dispatch scripts to available workers until all are assigned.
            curr_scr = 0
            while (curr_scr < len(self.device.scripts)) or 
                  (self.device.timepoint_done.is_set() is False):
                worker_idx = self.find_free_worker()
                if (worker_idx >= 0) and (curr_scr < len(self.device.scripts)):
                    (script, location) = self.device.scripts[curr_scr]
                    self.device.workers[worker_idx].update(location, script, neighbours, "run")
                    curr_scr += 1
                else:
                    continue # Spin-wait if no worker is free.
            
            # Signal end of timepoint to all workers and wait for them to finish.
            for i in range(0, self.device.max_threads):
                self.device.workers[i].update(None, None, None, "timepoint_end")
            self.device.timepoint_done.clear()
            self.device.notification.clear()
            
            # Wait at the global barrier for all other devices to finish.
            self.device.external_barrier.wait()


class Worker(Thread):
    """
    A stateful worker thread that receives commands and executes scripts.
    """
    def __init__(self, device):
        """Initializes the worker's state and synchronization events."""
        Thread.__init__(self)
        self.device = device
        self.init_start = Event() # Signals readiness to receive a new command.
        self.exec_start = Event() # Signals that a new command is ready to execute.
        self.location = None
        self.script = None
        self.neighbours = None
        self.is_free = True
        self.mode = "" # Command mode ("run", "end", "timepoint_end").
        self.exec_start.clear()
        self.init_start.set()

    def update(self, location, script, neighbours, mode):
        """
        Receives a new command and payload from the manager thread.
        This acts as the communication channel.
        """
        self.init_start.wait() # Wait until ready for a new command.
        self.init_start.clear()
        self.location = location
        self.script = script
        self.neighbours = neighbours
        self.mode = mode
        self.is_free = False
        self.exec_start.set() # Signal run() to start execution.

    def run(self):
        """
        Main loop that waits for commands and executes them based on the mode.
        """
        while True:
            self.exec_start.wait() # Wait for a command from the manager.
            self.exec_start.clear()

            if self.mode == "end":
                break # Exit the loop to terminate the thread.
            
            elif self.mode == "timepoint_end":
                # Synchronize with other workers before finishing the timepoint.
                self.device.internal_barrier.wait()
                self.is_free = True
                self.init_start.set() # Signal readiness for a new command.
            else: # "run" mode
                script_data = []
                # Gather data from neighbors and the parent device.
                for device in self.neighbours:
                    data = device.get_data(self.location)
                    if data is not None:
                        script_data.append(data)
                data = self.device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
                
                # Execute the script and propagate results.
                if script_data:
                    result = self.script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                    self.device.set_data(self.location, result)
                
                self.is_free = True
                self.init_start.set() # Signal readiness for a new command.
