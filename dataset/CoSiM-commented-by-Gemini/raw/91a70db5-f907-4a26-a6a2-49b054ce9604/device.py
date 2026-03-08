# -*- coding: utf-8 -*-
"""
Models a distributed system of concurrent devices using a persistent thread pool
and a static work partitioning scheme.

NOTE: This implementation contains several flaws, including a race condition in
lock creation, an inefficient lock lookup mechanism, and a confusing double
barrier wait at the end of its main loop.
"""

from threading import Lock, Event, Thread, Semaphore, Condition


class ReusableBarrierSem():
    """A correct two-phase reusable barrier using semaphores."""
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


class Device(object):
    """
    Represents a device in the simulation. Manages state and a pool of workers.
    Shared resources like locks and barriers are stored as class variables.
    """
    # --- Class-level shared state ---
    location_locks = []
    barrier = None
    nr_t = 8 # Number of threads per device.

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device and starts its persistent worker threads."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.neighbours_event = Event()
        self.threads = [DeviceThread(self, i) for i in xrange(Device.nr_t)]
        for thread in self.threads:
            thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes the single shared barrier for all devices."""
        Device.barrier = ReusableBarrierSem(Device.nr_t * len(devices))

    def assign_script(self, script, location):
        """
        Assigns a script to the device.

        NOTE: This method has a race condition when creating locks for new locations.
        """
        # --- FLAW: Race Condition on Lock Creation ---
        # The check for `location` and the subsequent `append` are not atomic.
        # Multiple threads could add a lock for the same new location simultaneously.
        if location not in [elem[0] for elem in Device.location_locks]:
            Device.location_locks.append((location, Lock()))

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its worker threads."""
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """A persistent worker thread within a Device."""
    def __init__(self, device, index):
        """Initializes the worker with a unique index for partitioning work."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.index = index
        self.neighbours = None

    def run(self):
        """Main simulation loop for a worker thread."""
        while True:
            # --- Leader election for neighbor discovery ---
            if self.index == 0: # Thread 0 of each device gets neighbors.
                self.neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours_event.set() # Signal others.
            else: # Other threads wait and copy the result.
                self.device.neighbours_event.wait()
                self.neighbours = self.device.threads[0].neighbours
            
            if self.neighbours is None: # Shutdown signal
                break

            # Wait for supervisor to signal all scripts for the timepoint are assigned.
            self.device.timepoint_done.wait()

            # --- Static Work Partitioning ---
            # Each thread processes a unique subset of scripts using its index.
            for j in range(self.index, len(self.device.scripts), Device.nr_t):
                location = self.device.scripts[j][1]
                script = self.device.scripts[j][0]

                # --- FLAW: Inefficient Lock Lookup ---
                # This performs a linear scan to find the lock. A dictionary would be O(1).
                lock = None
                for loc, l in Device.location_locks:
                    if location == loc:
                        lock = l
                        break
                
                if lock:
                    with lock:
                        script_data = []
                        # Aggregate data
                        for device in self.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        if script_data:
                            # Execute script and disseminate results
                            result = script.run(script_data)
                            for device in self.neighbours:
                                device.set_data(location, result)
                            self.device.set_data(location, result)

            # --- Synchronization and Reset ---
            # All threads from all devices sync here.
            Device.barrier.wait()
            
            # Leader thread resets state for the next timepoint.
            if self.index == 0:
                self.device.timepoint_done.clear()
                self.device.scripts = []
                self.device.neighbours_event.clear()
            
            # --- FLAW: Confusing Double Barrier Wait ---
            # This second barrier wait is unnecessary with a correct two-phase barrier
            # and can complicate reasoning about the program state.
            Device.barrier.wait()
