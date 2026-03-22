"""
This module implements a distributed device simulation with several major flaws.

Key architectural features & flaws:
- A "master" device (device 0) creates and distributes shared resources.
- A single global lock (`lock_neigh`) is used for all updates to neighboring
  devices, creating a massive performance bottleneck that serializes most of
  the parallel work.
- The `ReusableBarrierSem` implementation is buggy, as it resets its counter
  prematurely, making it unsafe for reuse in some preemption scenarios.
- Each device's main thread (`DeviceThread`) dynamically creates a new worker
  thread for every script in every time step, which is highly inefficient.
- The control flow logic in `DeviceThread` uses two events (`script_received`
  and `timepoint_done`) redundantly, making it confusing.

Note: This script uses Python 2 syntax.
"""

from threading import Thread, Lock, Semaphore, Event

class ReusableBarrierSem(object):
    """
    A two-phase, semaphore-based reusable barrier.

    @warning This implementation is buggy. The counter for a phase is reset
    within that same phase, before all threads are guaranteed to have passed the
    `acquire()` call. This creates a race condition that makes the barrier unsafe
    for reuse under certain thread scheduling scenarios.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all `num_threads` have called it."""
        self.phase1()
        self.phase2()

    def phase1(self):
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # BUG: This reset should occur in phase2 to be safe.
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # BUG: This reset should occur in phase1.
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Device(object):
    """
    Represents a device node in the simulation.
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
        # --- Shared Objects ---
        self.barrier = None
        self.lock_neigh = None # A single, global lock for all neighbor updates.
        self.lock_mine = Lock()  # A personal lock for updating self.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared resources using a master pattern."""
        no_devices = len(devices)
        lock_neigh = Lock()
        barrier = ReusableBarrierSem(no_devices)

        if self.device_id == 0:
            # Device 0 creates the global barrier and the global neighbor lock.
            for i in range(no_devices):
                devices[i].barrier = barrier
                devices[i].lock_neigh = lock_neigh

    def assign_script(self, script, location):
        """Assigns a script to be run."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Setting both events here makes the second wait in DeviceThread redundant.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Non-thread-safe method to get data."""
        return self.sensor_data.get(location)


    def set_data(self, location, data):
        """Non-thread-safe method to set data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class WorkerThread(Thread):
    """
    A short-lived worker thread that executes a single script.
    """
    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self, name="Worker Thread")
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def collect_data(self, location_data):
        """Gathers data from the local device and all its neighbors."""
        location_data.append(self.device.get_data(self.location))
        for neigh in self.neighbours:
            location_data.append(neigh.get_data(self.location))

    def update_neighbours(self, result):
        """
        Updates all neighbors with the computed result.
        
        This method is a major performance bottleneck, as it uses a single global
        lock, serializing all neighbor updates across the entire system.
        """
        for neigh in self.neighbours:
            with self.device.lock_neigh:
                value = neigh.get_data(self.location)
                # The use of max() here implies a specific script logic.
                neigh.set_data(self.location, max(result, value))

    def update_self(self, result):
        """Updates the local device with the computed result."""
        with self.device.lock_mine:
            value = self.device.get_data(self.location)
            self.device.set_data(self.location, max(result, value))

    def run(self):
        """Executes the script logic: collect, compute, and update."""
        location_data = []
        self.collect_data(location_data)
        # Filter out None values before processing.
        location_data = [item for item in location_data if item is not None]

        if location_data:
            result = self.script.run(location_data)
            self.update_neighbours(result)
            self.update_self(result)


class DeviceThread(Thread):
    """
    The main control thread for a device.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main execution loop for the device."""
        threads = [None] * 200 # Fixed-size list for worker threads.
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for supervisor to signal that scripts are ready.
            self.device.script_received.wait()
            self.device.script_received.clear()
            
            # Inefficiently create a new thread for each script.
            for i in range(len(self.device.scripts)):
                (script, location) = self.device.scripts[i]
                threads[i] = WorkerThread(self.device, script, \
                    location, neighbours)
                threads[i].start()

            # Wait for all workers for this device to finish.
            for i in range(len(self.device.scripts)):
                threads[i].join()

            # Wait for all other devices to finish their work for this time step.
            self.device.barrier.wait()
            
            # This wait is redundant, as the event was set at the same time
            # as the script_received event.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
