"""
This module defines a simulated distributed device network where all threads from
all devices form a single global pool, synchronized using a Bulk Synchronous
Parallel (BSP) model with a two-phase barrier.
"""

from threading import Lock, Event, Thread, Semaphore, Condition


class ReusableBarrierSem():
    """
    A correct, two-phase reusable barrier for synchronizing multiple threads.
    It uses two semaphores to prevent race conditions where fast threads could
    lap slow threads.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters for each phase.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        # Semaphores for each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Waits at the barrier; composed of two distinct synchronization phases."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread arrives, releases all others for this phase.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next use.
        self.threads_sem1.acquire()

    def phase2(self):
        """Second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Device(object):
    """
    Represents a device that contributes a pool of threads to the global simulation.
    """
    # Class-level (static) variables shared by all threads of all devices.
    location_locks = []
    barrier = None
    nr_t = 8 # Number of threads per device.

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.neighbours_event = Event()

        # Each device creates its own pool of worker threads.
        self.threads = []
        for i in xrange(Device.nr_t):
            self.threads.append(DeviceThread(self, i))
        for i in xrange(Device.nr_t):
            self.threads[i].start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes a single global barrier for all threads from all devices.
        """
        Device.barrier = ReusableBarrierSem(Device.nr_t * len(devices))

    def assign_script(self, script, location):
        """
        Assigns a script and lazily creates a lock for its location if one doesn't exist.
        """
        # Note: This check-then-add operation is not thread-safe and could lead
        # to duplicate locks if called concurrently for the same new location.
        if location not in [elem[0] for elem in Device.location_locks]:
            Device.location_locks.append((location, Lock()))

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Signals that all scripts for the step have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        for i in xrange(Device.nr_t):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A worker thread. All instances across all devices operate as peers in a single pool.
    """
    def __init__(self, device, index):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.index = index
        self.neighbours = None

    def run(self):
        """The main BSP loop for a worker thread."""
        while True:
            # --- Neighbor Synchronization (Intra-Device) ---
            # Thread 0 of each device is the leader for fetching neighbor data.
            if self.index == 0:
                self.neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours_event.set() # Signal siblings.
            else:
                self.device.neighbours_event.wait() # Wait for leader.
                self.neighbours = self.device.threads[0].neighbours
            
            if self.neighbours is None:
                break # End of simulation.

            # Wait for the supervisor to signal that scripts are ready for this step.
            self.device.timepoint_done.wait()

            # --- Static Work Distribution ---
            # Each thread processes a statically assigned subset of its device's scripts.
            for j in range(self.index, len(self.device.scripts), Device.nr_t):
                location = self.device.scripts[j][1]
                script = self.device.scripts[j][0]

                # Inefficiently search for the location-specific lock. O(L)
                lock = None
                for loc, l in Device.location_locks:
                    if location == loc:
                        lock = l
                        break
                
                if lock:
                    lock.acquire()

                try:
                    script_data = []
                    # Gather data from neighbors and self.
                    for device in self.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                    
                    # Run script and propagate results.
                    if script_data:
                        result = script.run(script_data)
                        for device in self.neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)
                finally:
                    if lock:
                        lock.release()

            # --- Global Barrier: Phase 1 ---
            # All threads from all devices synchronize here.
            Device.barrier.wait()
            
            # Leader thread resets events for the next time step.
            if self.index == 0:
                self.device.timepoint_done.clear()
                self.device.neighbours_event.clear()

            # --- Global Barrier: Phase 2 ---
            # Ensures all threads have passed phase 1 before the next step begins.
            Device.barrier.wait()
