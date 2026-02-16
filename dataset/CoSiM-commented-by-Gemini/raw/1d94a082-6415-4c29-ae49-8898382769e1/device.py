"""
This module implements a distributed device simulation using a manual thread-pooling
strategy within each device to process tasks concurrently.

Each `Device`'s main `DeviceThread` partitions its assigned scripts and creates a
new set of threads for each simulation time step to execute them. Synchronization
across devices is handled by a correct two-phase `Barrier`, but the initialization
of shared locks is not thread-safe.
"""

from threading import Event, Thread, Lock, Semaphore

class Device(object):
    """
    Represents a device in the simulation. It receives scripts and uses a
    `DeviceThread` to manage their parallel execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Event to signal that all scripts for a timepoint have been assigned.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # The number of threads to manually create for parallel script execution.
        self.max_threads = 8
        # The shared barrier for end-of-step synchronization.
        self.barrier = None
        # The shared dictionary of location-based locks.
        self.location_locks = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources.

        NOTE: This method is fragile. It assumes it will be called exactly once
        with the complete list of all devices. It creates new shared objects on
        each call, which would break synchronization if called incorrectly.
        """
        barrier = Barrier(len(devices))
        location_locks = {}

        for device in devices:
            device.barrier = barrier
            device.location_locks = location_locks

    def assign_script(self, script, location):
        """Adds a script to the device's workload for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data for a given location, or None if not present."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device. It partitions scripts and creates
    worker threads to execute them in parallel for each time step.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main loop for the device's lifecycle."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation

            # Wait for the signal that all scripts for this step are assigned.
            self.device.timepoint_done.wait()

            # Block Logic: Partition the scripts into sub-lists for parallel execution.
            # This creates 'max_threads' piles of scripts to be processed.
            scripts_to_threads = [self.device.scripts[i::self.device.max_threads]
                                  for i in range(self.device.max_threads)]
            threads = []

            # Create and start a new thread for each non-empty pile of scripts.
            for scripts in scripts_to_threads:
                if scripts != []:
                    thread = Thread(target=self.run_thread, args=(neighbours, scripts))
                    thread.start()
                    threads.append(thread)

            # Wait for all worker threads for this time step to complete.
            for thread in threads:
                thread.join()

            # Synchronize with all other devices before the next time step.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()


    def run_thread(self, neighbours, scripts):
        """
        The target function for worker threads. Executes a sub-list of scripts.
        """
        for (script, location) in scripts:

            # --- CRITICAL RACE CONDITION ---
            # The check for the key's existence and the creation of the lock
            # is not an atomic operation. Two threads could check simultaneously,
            # find the key is missing, and then both create a lock, with one
            # overwriting the other. This needs an external lock.
            if location not in self.device.location_locks:
                self.device.location_locks[location] = Lock()
            self.device.location_locks[location].acquire()

            # Block Logic: Gather data, execute script, and disseminate results
            # while holding the location-specific lock.
            data = [dev.get_data(location) for dev in neighbours if dev.get_data(location)]
            if self.device.get_data(location):
                data += [self.device.get_data(location)]

            if data != []:
                result = script.run(data)
                self.device.set_data(location, result)
                for neighbour in neighbours:
                    neighbour.set_data(location, result)

            self.device.location_locks[location].release()

class Barrier(object):
    """
    A correct, two-phase reusable barrier for thread synchronization.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks a thread until all threads have reached the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second synchronization phase to prevent premature re-entry."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()
