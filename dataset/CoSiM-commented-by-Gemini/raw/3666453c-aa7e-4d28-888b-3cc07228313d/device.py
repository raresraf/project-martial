"""
This module implements a highly detailed and complex simulation of a distributed
device network. It features stateful worker threads that are managed manually
by a main device thread, a two-level barrier synchronization system, and a
custom reader-writer lock implementation for data access.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """
    A reusable two-phase barrier implemented with semaphores, for synchronizing
    multiple threads at a specific point in their execution, multiple times.
    """

    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the caller until all threads have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes a single phase of the two-phase barrier protocol."""
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
    Represents a device in the network, managing a pool of stateful worker threads
    and complex synchronization primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.max_threads = 8
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.notification = Event()
        self.timepoint_done = Event()
        self.notification.clear()
        self.timepoint_done.clear()
        self.update_locks = {}
        self.read_locations = {}
        self.external_barrier = None # For synchronizing with other devices.
        self.internal_barrier = ReusableBarrier(self.max_threads) # For synchronizing own workers.
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
        """Joins all worker threads, waiting for them to terminate."""
        for i in range(0, self.max_threads):
            self.workers[i].join()

    def setup_devices(self, devices):
        """
        Initializes shared resources. Device 0 acts as master, while others
        busy-wait to receive the shared barrier.
        @note The while-loop constitutes busy-waiting, which is inefficient.
        """
        if self.device_id == 0:
            self.external_barrier = ReusableBarrier(len(devices))
        else:
            for device in devices:
                if device.device_id == 0:
                    while device.external_barrier is None:
                        pass # Busy-wait for the master to create the barrier.
                    self.external_barrier = device.external_barrier
                    break

    def assign_script(self, script, location):
        """Assigns a script or signals the end of a timepoint."""
        self.notification.set()
        if script is not None:
            # Lazily initialize synchronization primitives for new locations.
            if location not in self.update_locks:
                self.update_locks[location] = Lock()
                self.read_locations[location] = Event()
                self.read_locations[location].set()
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Reads data for a location using a custom reader-writer lock scheme.
        Waits if a write is in progress.
        """
        if location not in self.sensor_data:
            return None
        else:
            if location not in self.read_locations:
                # Lazy initialization if accessed first by a reader.
                self.update_locks[location] = Lock()
                self.read_locations[location] = Event()
                self.read_locations[location].set()
            self.read_locations[location].wait() # Readers wait if event is cleared by a writer.
            return self.sensor_data[location]

    def set_data(self, location, data):
        """
        Writes data to a location using a custom reader-writer lock scheme.
        Blocks other readers and writers.
        """
        if location in self.sensor_data:
            self.update_locks[location].acquire() # Exclusive lock for writing.
            self.read_locations[location].clear() # Block readers.
            self.sensor_data[location] = data
            self.read_locations[location].set() # Unblock readers.
            self.update_locks[location].release()

    def shutdown(self):
        """Shuts down all worker threads and the main device thread."""
        self.stop_workers()
        self.thread.join()


class DeviceThread(Thread):
    """
    Main control thread for a device, responsible for orchestrating worker
    threads and inter-device synchronization.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def find_free_worker(self):
        """Finds an idle worker thread in the pool."""
        for i in range(0, self.device.max_threads):
            if self.device.workers[i].is_free:
                return i
        return -1

    def run(self):
        """
        The main simulation loop. Manages script assignment to workers and
        synchronizes at the end of each timepoint.
        """
        self.device.start_workers()

        while True:
            neighbours = self.device.supervisor.get_neighbours()
            
            if neighbours is None:
                # Supervisor signals end of simulation.
                for i in range(0, self.device.max_threads):
                    self.device.workers[i].update(None, None, None, "end")
                break

            # Wait for supervisor to provide scripts.
            if len(self.device.scripts) == 0:
                self.device.notification.wait()

            # Block Logic: Assigns all scripts for the current timepoint to workers.
            curr_scr = 0
            while (curr_scr < len(self.device.scripts)) or \
                  (self.device.timepoint_done.is_set() is False):
                worker_idx = self.find_free_worker()
                if (worker_idx >= 0) and (curr_scr < len(self.device.scripts)):
                    (script, location) = self.device.scripts[curr_scr]
                    self.device.workers[worker_idx].update(location, script, neighbours, "run")
                    curr_scr += 1
                else:
                    continue # Continue until all scripts are dispatched.

            # Block Logic: Signal end of timepoint to workers and synchronize.
            # This forms a two-level barrier system.
            # 1. Signal workers to go to their internal barrier.
            for i in range(0, self.device.max_threads):
                self.device.workers[i].update(None, None, None, "timepoint_end")
            self.device.timepoint_done.clear()
            self.device.notification.clear()
            
            # 2. Wait at the external barrier for all other devices to finish.
            self.device.external_barrier.wait()


class Worker(Thread):
    """
    A stateful, long-running worker thread that executes tasks based on commands
    from its parent DeviceThread.
    """

    def __init__(self, device):
        
        Thread.__init__(self)
        self.device = device
        self.init_start = Event() # Signals worker is ready for a new task.
        self.exec_start = Event() # Signals worker has a new task to execute.
        self.location = None
        self.script = None
        self.neighbours = None
        self.is_free = True
        self.mode = ""
        self.exec_start.clear()
        self.init_start.set()

    def update(self, location, script, neighbours, mode):
        """
        Receives a new task from the DeviceThread. This is a hand-off mechanism.
        """
        self.init_start.wait() # Wait until worker is ready.
        self.init_start.clear()
        self.location = location
        self.script = script
        self.neighbours = neighbours
        self.mode = mode
        self.is_free = False
        self.exec_start.set() # Signal worker to start execution.

    def run(self):
        """
        Main loop of the worker. Waits for tasks and executes them based on the mode.
        """
        while True:
            self.exec_start.wait() # Wait for a new task.
            self.exec_start.clear()
            if self.mode == "end":
                # Terminate the worker thread.
                break
            elif self.mode == "timepoint_end":
                # Synchronize with other workers of the same device.
                self.device.internal_barrier.wait()
                self.is_free = True
                self.init_start.set() # Signal readiness for a new task.
            else: # "run" mode
                # Execute the script.
                script_data = []
                
                for device in self.neighbours:
                    data = device.get_data(self.location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
                if script_data != []:
                    result = self.script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                    
                    self.device.set_data(self.location, result)
                self.is_free = True
                self.init_start.set() # Signal readiness for a new task.
