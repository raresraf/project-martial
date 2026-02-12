# -*- coding: utf-8 -*-
"""
This module provides a highly concurrent simulation of networked devices.
The architecture features a two-level synchronization model:
1. Inter-device: All devices synchronize at the start and end of a timepoint
   using a global barrier.
2. Intra-device: Each device has a master thread and a pool of 8 worker
   threads, which synchronize with each other using a local barrier.

The locking strategy is per-device, meaning all access to a single device's
data is serialized, which differs from per-location locking models.

Classes:
    ReusableBarrierSem: A semaphore-based reusable barrier.
    Device: A node in the network with its own internal thread pool.
    DeviceThread: The master thread within a device, coordinating its workers.
    Worker: A long-lived thread that executes assigned scripts.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """A standard two-phase reusable barrier using semaphores."""

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes the calling thread to wait at the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase of the barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second synchronization phase to prevent thread lapping."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a device that contains its own master thread and a pool of
    8 worker threads, all started upon initialization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        # --- State and Events ---
        self.scripts = []
        self.script_received = Event()
        self.timepoint_done = Event()
        self.setup_done = Event()
        self.terminate = Event() # Global shutdown signal for all threads.

        # --- Synchronization Objects ---
        self.lock = {} # Per-device locks, shared across the system.
        self.barrier = None # The main inter-device barrier.
        # An intra-device barrier for the master and its 8 workers.
        self.threads_barrier = ReusableBarrierSem(9)
        
        # --- Thread Creation ---
        # Create and start the master thread for this device.
        self.master = DeviceThread(self, self.terminate, self.barrier, self.threads_barrier, \
                                    self.setup_done)
        self.master.start()

        # Create and start the pool of 8 worker threads for this device.
        self.threads = [
            Worker(self.master, self.terminate, self.threads_barrier)
            for _ in range(8)
        ]
        for thread in self.threads:
            thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources, led by device 0.
        This method creates a per-device lock, rather than per-location.
        """
        if self.device_id == 0:
            # The leader creates the main barrier for inter-device sync.
            self.barrier = ReusableBarrierSem(len(devices))
            # Create one lock for each device object.
            for dev in devices:
                self.lock[dev] = Lock()
            # Distribute shared objects to all other devices.
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier
                    dev.lock = self.lock
                    dev.setup_done.set() # Signal worker to proceed.
            self.setup_done.set() # Signal self to proceed.

    def assign_script(self, script, location):
        """Assigns a script to the device's script list."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Terminates the master and all worker threads."""
        self.terminate.set()
        for i in range(8):
            self.threads[i].script_received.set() # Wake up waiting threads.
            self.threads[i].join()
        self.master.join()


class DeviceThread(Thread):
    """The master thread within a device, coordinating its 8 workers."""

    def __init__(self, device, terminate, barrier, threads_barrier, setup_done):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = []
        self.terminate = terminate
        self.barrier = barrier
        self.threads_barrier = threads_barrier
        self.setup_done = setup_done

    def run(self):
        """Main loop for the master thread."""
        self.setup_done.wait()
        # Initial synchronization with all other device master threads.
        self.device.barrier.wait()

        while True:
            # 1. Synchronize with other devices at the start of the timepoint.
            self.device.barrier.wait()

            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break

            # 2. Wait for the supervisor to assign all scripts.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            
            # 3. Distribute the collected scripts among the 8 worker threads.
            scripts = [[] for _ in range(8)]
            for i, script_item in enumerate(self.device.scripts):
                scripts[i % 8].append(script_item)

            for i in range(8):
                self.device.threads[i].scripts = scripts[i]
                self.device.threads[i].script_received.set() # Wake up workers.

            # 4. Wait on the intra-device barrier until all 8 workers are done.
            if not self.terminate.is_set():
                self.threads_barrier.wait()
            
            # 5. Synchronize with other devices at the end of the timepoint.
            self.device.barrier.wait()


class Worker(Thread):
    """A worker thread that executes a subset of a device's scripts."""

    def __init__(self, master, terminate, barrier):
        Thread.__init__(self)
        self.master = master
        self.script_received = Event()
        self.terminate = terminate
        self.scripts = []
        self.barrier = barrier

    @staticmethod
    def append_data(device, location, script_data):
        """Thread-safely gets data from a device using a per-device lock."""
        device.lock[device].acquire()
        data = device.get_data(location)
        device.lock[device].release()
        if data is not None:
            script_data.append(data)

    @staticmethod
    def set_data(device, location, result):
        """Thread-safely sets data on a device using a per-device lock."""
        device.lock[device].acquire()
        device.set_data(location, result)
        device.lock[device].release()

    def run(self):
        """Main loop for the worker thread."""
        while True:
            # 1. Wait to be assigned scripts by the master thread.
            self.script_received.wait()
            self.script_received.clear()

            if self.terminate.is_set():
                break
            
            # 2. Process all assigned scripts.
            if self.scripts:
                for (script, location) in self.scripts:
                    script_data = []
                    # Gather data from neighbors.
                    if self.master.neighbours:
                        for device in self.master.neighbours:
                            self.append_data(device, location, script_data)
                    # Gather data from self.
                    self.append_data(self.master.device, location, script_data)

                    # Execute and disseminate.
                    if script_data:
                        result = script.run(script_data)
                        if self.master.neighbours:
                            for device in self.master.neighbours:
                                self.set_data(device, location, result)
                        self.set_data(self.master.device, location, result)
            
            # 3. Wait on the intra-device barrier to signal completion to the master.
            self.barrier.wait()