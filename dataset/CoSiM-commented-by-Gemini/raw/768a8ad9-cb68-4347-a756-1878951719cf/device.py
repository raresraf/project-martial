"""
This module implements a multi-threaded device simulation framework using a
leader-election pattern for setup and a semaphore to limit concurrency.

This version demonstrates a more robust approach to setting up shared
synchronization primitives (barriers and locks) compared to previous variants.
A "leader" device (the one with the minimum ID) creates the shared resources
and distributes them to others. It also uses a Semaphore to effectively limit
the number of concurrently active worker threads to 8.

NOTE: Despite improvements in setup and concurrency limiting, this implementation
still suffers from a critical flaw: the `get_data` and `set_data` methods are
not thread-safe, which will lead to data races. The threading model is also
inefficient as it creates new threads for every script. This script appears
to be written for Python 2, indicated by `xrange`.
"""

from threading import Semaphore, Event, Lock, Thread

class ReusableBarrier(object):
    """A correct, two-phase reusable barrier using Semaphores."""

    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in xrange(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase to ensure reusability."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in xrange(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """Represents a device in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.none_script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()
        # --- Shared resources ---
        self.barrier = None
        self.lock_hash = None # A shared dictionary mapping locations to locks.

    def __str__(self):
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """Receives the shared barrier from the leader device."""
        self.barrier = barrier

    def set_locks(self, lock_hash):
        """Receives the shared lock dictionary from the leader device."""
        self.lock_hash = lock_hash

    def setup_devices(self, devices):
        """
        Sets up shared resources using a leader election model.

        The device with the minimum ID creates the shared barrier and location locks
        and distributes them to all other devices.
        """
        ids_list = [dev.device_id for dev in devices]

        if self.device_id == min(ids_list):
            # This device is the leader.
            self.barrier = ReusableBarrier(len(devices))
            self.lock_hash = {}

            # Discover all unique locations and create one shared lock for each.
            for dev in devices:
                for location in dev.sensor_data:
                    if location not in self.lock_hash:
                        self.lock_hash[location] = Lock()

            # Distribute the shared objects to the other devices.
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.set_barrier(self.barrier)
                    dev.set_locks(self.lock_hash)

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that all scripts for the time step have been received.
            self.none_script_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data.
        
        BUG: This method is not thread-safe. It lacks a lock to protect against
        concurrent `set_data` calls, leading to potential data races.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates sensor data.

        BUG: This method is not thread-safe. It lacks a lock to protect against
        concurrent reads or writes, leading to potential data races.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        # This semaphore limits the number of concurrently running worker threads to 8.
        self.semaphore = Semaphore(value=8)

    def run(self):
        """Main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal.

            # Wait until all scripts for the current step are assigned.
            self.device.none_script_received.wait()
            self.device.none_script_received.clear()

            thread_list = []
            
            # --- Inefficient Threading Model ---
            # Create a new thread for each script. A persistent pool would be more efficient.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, neighbours, script, location, self.semaphore)
                thread.start()
                thread_list.append(thread)

            # Wait for all worker threads to complete.
            for thread in thread_list:
                thread.join()

            # Wait at the global barrier for all other devices to finish their step.
            self.device.barrier.wait()

class MyThread(Thread):
    """A worker thread that executes a single script."""

    def __init__(self, device, neighbours, script, location, semaphore):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.semaphore = semaphore

    def run(self):
        # Acquire the semaphore to limit concurrency. Blocks if 8 threads are active.
        self.semaphore.acquire()

        try:
            # Acquire the specific lock for the target location.
            self.device.lock_hash[self.location].acquire()

            try:
                script_data = []

                # Gather data from neighbors (UNSAFE READ).
                for device in self.neighbours:
                    data = device.get_data(self.location)
                    if data is not None:
                        script_data.append(data)

                # Gather data from self (UNSAFE READ).
                data = self.device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    result = self.script.run(script_data)

                    # Propagate results (UNSAFE WRITE).
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                    self.device.set_data(self.location, result)
            finally:
                # Ensure the location lock is always released.
                self.device.lock_hash[self.location].release()
        finally:
            # Ensure the concurrency-limiting semaphore is always released.
            self.semaphore.release()
