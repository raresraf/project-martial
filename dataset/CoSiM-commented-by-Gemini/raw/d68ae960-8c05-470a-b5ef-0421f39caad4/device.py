"""
This module contains a distributed device simulation.
WARNING: This implementation has severe architectural flaws, including a major
memory leak due to repeated thread creation and a locking protocol that is
highly susceptible to deadlocks. The comments describe the intended logic,
but the code as written is fundamentally broken.
"""

from threading import *


class Device(object):
    """
    Represents a device in the simulation. This version creates worker threads
    on-the-fly within its main coordinator thread's loop.
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
        self.barrier = None
        self.lock = Lock()  # A device-specific lock.
        self.locks = []     # A list of shared, location-specific locks.

    def __str__(self):
        return f"Device {self.device_id}"

    def setup_devices(self, devices):
        """
        Performs a centralized setup where device 0 creates and distributes
        shared synchronization objects (a barrier and location locks).
        """
        if self.device_id == 0:
            # Pre-allocate a fixed number of location locks.
            for _ in range(100):
                self.add_lock(Lock())
            # Share the same list of locks with all other devices.
            for j in devices:
                if j.device_id != 0:
                    j.locks = self.locks
            
            # Create and share a single barrier instance for all devices.
            nr = len(devices)
            barrier = ReusableBarrierSem(nr)
            for i in devices:
                i.barrier = barrier

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets data for a specific location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Sets data for a location, only if the new data is greater than the old.
        This implies the simulation tracks a maximum value.
        """
        if location in self.sensor_data and self.sensor_data[location] < data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

    def add_lock(self, lock):
        self.locks.append(lock)


class DeviceThread(Thread):
    """
    The main coordinator thread for a device.
    Its design is critically flawed.
    """

    def __init__(self, device):
        Thread.__init__(self, name=f"Device Thread {device.device_id}")
        self.device = device

    def run(self):
        """
        The main simulation loop.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # MEMORY LEAK: Creates 8 new threads on every single iteration of the loop.
            threads = [Master(i) for i in range(8)]
            index = 0

            self.device.timepoint_done.wait()
            for (script, location) in self.device.scripts:
                script_data = []

                # DEADLOCK RISK: Acquires a location lock, then iterates neighbors
                # to acquire their device-specific locks. This is a classic
                # AB-BA deadlock pattern if two devices do this concurrently.
                self.device.locks[location].acquire()
                
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
                    # Distributes work to the newly created worker threads.
                    threads[index].set_worker(script, script_data)
                    threads[index].add_location(location)
                    index = (index + 1) % 8
                
                # The location lock is held for the entire duration of data gathering.
                self.device.locks[location].release()

            # Start, join, and process results from workers, all within one time-step.
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Process and propagate results, repeating the deadlock-prone locking pattern.
            for i in range(8):
                result_list = threads[i].get_result()
                location_list = threads[i].get_location()
                for j, r in enumerate(result_list):
                    l = location_list[j]
                    with self.device.locks[l]:
                        for device in neighbours:
                            with device.lock:
                                device.set_data(l, r)
                        with self.device.lock:
                            self.device.set_data(l, r)

            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            # This barrier correctly synchronizes all DeviceThreads.
            if self.device.barrier:
                self.device.barrier.wait()


class ReusableBarrierSem():
    """A correct, two-phase reusable barrier using semaphores."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = num_threads
        self.count_threads2 = num_threads
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


class Master(Thread):
    """
    A worker thread that executes a batch of assigned scripts.
    """
    def __init__(self, id):
        Thread.__init__(self)
        self.Thread_script = []
        self.Thread_script_data = []
        self.Thread_location = []
        self.Thread_result = []

    def set_worker(self, script, script_data):
        """Assigns a script and its data to this worker."""
        self.Thread_script.append(script)
        self.Thread_script_data.append(script_data)
        
    def add_location(self, location):
        """Records the location for a given script."""
        self.Thread_location.append(location)

    def get_result(self):
        return self.Thread_result

    def get_location(self):
        return self.Thread_location

    def run(self):
        """Executes all assigned scripts and stores their results."""
        for i in range(len(self.Thread_script)):
            result = self.Thread_script[i].run(self.Thread_script_data[i])
            self.Thread_result.append(result)
