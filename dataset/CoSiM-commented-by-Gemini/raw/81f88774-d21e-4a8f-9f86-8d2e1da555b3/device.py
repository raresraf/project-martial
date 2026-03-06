"""
This module implements a device simulation with a unique architecture that
separates data access (I/O) from pure computation.

The main `DeviceThread` acts as an orchestrator that performs all locking and data
retrieval/storage. It delegates the actual script execution (computation) to a
pool of `Master` worker threads.

WARNING: This implementation contains a critical race condition. The location-specific
lock is acquired and released during the data gathering phase, *before* the
computation and write-back phases. This allows multiple devices to interleave
their operations on the same location, potentially leading to computations being
performed on stale data. The locking logic in the write-back phase does not
mitigate this initial race.
"""

from threading import *


class Device(object):
    """Represents a device, holding its data and synchronization objects."""
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        # Each device has its own lock, presumably to protect its own data.
        self.lock = Lock()
        # A list of locks for locations, shared among all devices.
        self.locks = []

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared locks and the main barrier."""
        if self.device_id == 0:
            # Create 100 location-specific locks and a barrier.
            for i in xrange(100):
                aux_lock = Lock()
                # Distribute these shared resources to all devices.
                for j in devices:
                    j.add_lock(aux_lock)
            
            barrier = ReusableBarrierSem(len(devices))
            for i in devices:
                i.barrier = barrier

    def assign_script(self, script, location):
        """Assigns a script for the next timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. Not thread-safe on its own."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Sets sensor data only if the new value is greater.
        Note: This is not thread-safe and relies on external locking.
        """
        if location in self.sensor_data and self.sensor_data[location] < data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

    def add_lock(self, lock):
        self.locks.append(lock)

class DeviceThread(Thread):
    """
    The main orchestrator thread for a device. It handles I/O and delegates
    computation to a pool of `Master` worker threads.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Create a new pool of worker threads for each timepoint.
            threads = [Master(i) for i in xrange(8)]
            index = 0

            self.device.timepoint_done.wait()
            
            # --- PHASE 1: Data Gathering (by Main Thread) ---
            # This loop contains a race condition.
            for (script, location) in self.device.scripts:
                self.device.locks[location].acquire()
                
                script_data = []
                # Inefficiently lock/unlock each neighbor for each data read.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        with device.lock:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                
                with self.device.lock:
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                if script_data:
                    # Assign the pre-gathered data to a worker for computation.
                    threads[index % 8].set_worker(script, script_data)
                    threads[index % 8].add_location(location)
                    index += 1

                # RACE CONDITION: The location lock is released here, before the
                # result of the computation has been written back. Another thread
                # could now read this same location and compute based on stale data.
                self.device.locks[location].release()

            # --- PHASE 2: Computation (by Worker Threads) ---
            for i in xrange(8):
                threads[i].start()
            for i in xrange(8):
                threads[i].join()

            # --- PHASE 3: Write-Back (by Main Thread) ---
            for i in xrange(8):
                result_list = threads[i].get_result()
                location_list = threads[i].get_location()
                for j, (r, l) in enumerate(zip(result_list, location_list)):
                    # Re-acquire locks to write the results.
                    with self.device.locks[l]:
                        for device in neighbours:
                            with device.lock:
                                device.set_data(l, r)
                        with self.device.lock:
                            self.device.set_data(l, r)
            
            self.device.scripts = [] # Clear scripts for next round.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()


class ReusableBarrierSem():
    """A standard two-phase reusable barrier using semaphores."""
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
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Master(Thread):
    """
    A pure computation worker thread. It receives pre-gathered data and a script,
    runs the script, and stores the result. It performs no locking or I/O.
    """
    def __init__(self, id):
        Thread.__init__(self)
        self.Thread_script = []
        self.Thread_script_data = []
        self.Thread_location = []
        self.Thread_result = []
        self.Thread_id = id
        self.Thread_iterations = 0

    def add_result(self, result):
        self.Thread_result.append(result)

    def add_script(self, script):
        self.Thread_script.append(script)

    def add_script_data(self, script_data):
        self.Thread_script_data.append(script_data)

    def add_location(self, location):
        self.Thread_location.append(location)

    def set_worker(self, script, script_data):
        self.add_script(script)
        self.add_script_data(script_data)
    
    def get_result(self):
        return self.Thread_result

    def get_location(self):
        return self.Thread_location
    
    def run(self):
        self.Thread_iterations = len(self.Thread_script)
        for i in xrange(self.Thread_iterations):
            aux_script = self.Thread_script[i]
            aux_script_data = self.Thread_script_data[i]
            aux_rez = aux_script.run(aux_script_data)
            self.add_result(aux_rez)
