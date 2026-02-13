"""
This module simulates a distributed system where each device is an active thread,
spawning new worker threads for each task within every timepoint.

Key architectural features:
- Device-as-Thread: The main `Device` class inherits from `threading.Thread`,
  managing its own execution lifecycle in the `run` method.
- Per-Task Threading: For each timepoint, the `Device` thread creates, starts,
  and joins a new `Worker` thread for every single script it needs to execute.
  This is a highly inefficient threading model due to high thread creation/destruction
  overhead.
- Asymmetric Locking: A dangerous locking pattern is used where data-reading
  functions acquire semaphores and data-writing functions release them, creating
  a high risk of deadlocks.
"""

from threading import Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    A custom implementation of a reusable barrier using semaphores.

    This employs a standard two-phase protocol to ensure all threads wait at the
    barrier before any are released, allowing for immediate reuse.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """Executes the first phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Executes the second phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

def start_threads(threads):
    """Helper function to start a list of threads."""
    for thread in threads:
        thread.start()

def join_threads(threads):
    """Helper function to join a list of threads."""
    for thread in threads:
        thread.join()

def create_semaphores(devices):
    """
    Creates and distributes location-based semaphores to all devices.
    
    This function determines the highest location ID across all devices and creates
    a semaphore for each location up to that maximum.
    """
    max_locations = 0
    for dev in devices:
        temp_max = max(dev.sensor_data, key=int)
        if max_locations < temp_max:
            max_locations = temp_max

    for dev in devices:
        for i in range(0, max_locations + 1):
            dev.sems_locations[i] = Semaphore(1)

class Device(Thread):
    """
    Represents a device node, implemented as a main control thread.

    This thread waits for scripts, spawns worker threads to execute them, and
    synchronizes with other devices at the end of each timepoint.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        Thread.__init__(self)
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.workers = []
        self.lock = None # This lock appears to be unused.
        self.bariera = None
        self.bar_workers = None
        self.neighbours = None
        # A semaphore used to signal that all scripts for a timepoint have been assigned.
        self.script_received = Semaphore(0)
        # A dictionary to hold location-specific semaphores.
        self.sems_locations = {}
        self.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes the global barrier and semaphores."""
        if self.device_id == 0:
            bariera = ReusableBarrierSem(len(devices))
            lock = Lock()
            for dev in devices:
                dev.bariera = bariera
                dev.lock = lock
            create_semaphores(devices)

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A `None` script signals the end of script assignment for the timepoint.
            # This releases the main device thread to start processing.
            self.script_received.release()

    def get_data(self, location):
        """
        Retrieves data for a location and ACQUIRES the corresponding semaphore.

        @warning Dangerous Asymmetric Locking: This function acquires a semaphore but
        does not release it. The release is expected in `set_data`. If `get_data`
        is called twice for the same location without an intervening `set_data`
        (e.g., from two different workers in the same script), it will deadlock.
        """
        self.sems_locations[location].acquire()
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Updates data for a location and RELEASES the corresponding semaphore.

        @warning Dangerous Asymmetric Locking: This function releases a semaphore it
        did not acquire. This pattern is highly prone to deadlocks and race conditions.
        """
        self.sensor_data[location] = data
        self.sems_locations[location].release()

    def shutdown(self):
        """Shuts down all worker threads and the main device thread."""
        join_threads(self.workers)
        self.join()

    def run(self):
        """The main execution loop for the Device thread."""
        while True:
            # Get neighbor list from the central supervisor.
            self.neighbours = self.supervisor.get_neighbours()
            if self.neighbours is None:
                break # End of simulation signal.

            # Wait for the supervisor to signal that all scripts are assigned.
            self.script_received.acquire()
            
            self.workers = []

            # Inefficient Threading Model: Creates a new thread for every script.
            if self.scripts is not None:
                for (script, location) in self.scripts:
                    worker = Worker(self, self.neighbours, location, script)
                    self.workers.append(worker)

                start_threads(self.workers)
                # Blocks here until all worker threads for this timepoint complete.
                join_threads(self.workers)

            # Synchronize with all other devices before starting the next timepoint.
            self.bariera.wait()

class Worker(Thread):
    """
    A worker thread created to execute a single script.
    """
    def __init__(self, parent, vecini, location, script):
        Thread.__init__(self)
        self.parent = parent
        self.worker_id = parent.device_id
        self.parent_neighbours = vecini
        self.location_for_script = location
        self.script = script

    def __str__(self):
        return "Worker " + str(self.worker_id)

    def run(self):
        """The main execution logic for the worker."""
        script_data = []

        # Gather data from neighbors. This may cause deadlocks due to the
        # flawed locking in the get_data method.
        for neighbour in self.parent_neighbours:
            if neighbour.device_id != self.worker_id:
                data = neighbour.get_data(self.location_for_script)
                if data is not None:
                    script_data.append(data)

        # Gather data from the parent device.
        data = self.parent.get_data(self.location_for_script)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = self.script.run(script_data)

            # Broadcast the result, releasing the locks acquired during get_data.
            for neighbour in self.parent_neighbours:
                if neighbour.device_id != self.worker_id:
                    neighbour.set_data(self.location_for_script, result)

            self.parent.set_data(self.location_for_script, result)
