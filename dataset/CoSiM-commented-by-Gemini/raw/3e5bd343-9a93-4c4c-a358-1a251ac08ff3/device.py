
"""
@brief A device simulation using a dynamic master and semaphore-limited workers.
@file device.py

This module implements a distributed device simulation featuring a dynamic master
(the device with the minimum ID) for initialization. For script execution, it uses
an inefficient model of creating new worker threads (`MyThread`) for each time step,
but adds a `Semaphore` to limit the number of concurrently active workers to 8.

WARNING: This implementation contains multiple severe concurrency and design flaws.
1.  **Startup Race Condition**: In `setup_devices`, worker devices do not wait for
    the master device to finish initializing and distributing the shared `barrier`
    and `lock_hash`. This can cause workers to access uninitialized resources,
    leading to a crash.
2.  **Inefficient Threading**: The `DeviceThread` creates and destroys a new set
    of worker threads for every time step, which is a highly inefficient model
    that introduces significant overhead.
3.  **Flawed `ReusableBarrier`**: The barrier implementation holds a lock while
    releasing waiting threads, a classic anti-pattern that can cause deadlocks.
4.  **Unsafe Lock Management**: `MyThread` uses manual `acquire()` and `release()`
    calls. If an error occurs between these calls, the lock will not be
    released, causing a permanent deadlock. A `with` statement is required for safety.
"""

from threading import Semaphore, Event, Lock, Thread

class ReusableBarrier(object):
    """
    A flawed implementation of a reusable two-phase barrier using semaphores.

    WARNING: This implementation is not safe. It holds `counter_lock` while
    releasing the semaphores, which is an anti-pattern that can serialize
    thread wakeup and create deadlocks. `xrange` is also Python 2 syntax.
    """
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all threads reach the barrier. Consists of two phases."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in xrange(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in xrange(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """Represents a device node in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.none_script_received = Event()

        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start() # RACE CONDITION: Thread starts before setup is complete.
        self.barrier = None
        self.lock_hash = None

    def __str__(self):
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """Setter for the shared barrier object."""
        self.barrier = barrier

    def set_locks(self, lock_hash):
        """Setter for the shared dictionary of locks."""
        self.lock_hash = lock_hash

    def setup_devices(self, devices):
        """
        Initializes shared resources using the device with the minimum ID as master.
        
        RACE CONDITION: Worker devices do not wait for this method to complete
        on the master device, so they may access `barrier` and `lock_hash`
        before they are initialized.
        """
        ids_list = [dev.device_id for dev in devices]

        if self.device_id == min(ids_list):
            # Master device creates and distributes the shared resources.
            self.barrier = ReusableBarrier(len(devices))
            self.lock_hash = {}

            for dev in devices:
                for location in dev.sensor_data:
                    if location not in self.lock_hash:
                        self.lock_hash[location] = Lock()
            
            for dev in devices:
                if dev.device_id != self.device_id:
                    dev.set_barrier(self.barrier)
                    dev.set_locks(self.lock_hash)


    def assign_script(self, script, location):
        """Adds a script to the device's workload for the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.none_script_received.set()

    def get_data(self, location):
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, which spawns worker threads.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # This semaphore limits the number of active worker threads to 8.
        self.semaphore = Semaphore(value=8)

    def run(self):
        """Main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for supervisor to assign all scripts for this time step.
            self.device.none_script_received.wait()
            self.device.none_script_received.clear()

            thread_list = []

            # --- INEFFICIENCY WARNING ---
            # Spawns a new thread for each script in every time step.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, neighbours, script, location,
                    self.semaphore)
                thread.start()
                thread_list.append(thread)

            # Wait for all of this step's worker threads to complete.
            for i in xrange(len(thread_list)):
                thread_list[i].join()

            # Wait at the global barrier for all other devices.
            self.device.barrier.wait()

class MyThread(Thread):
    """A short-lived worker thread that executes one script."""
    def __init__(self, device, neighbours, script, location, semaphore):
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.semaphore = semaphore

    def run(self):
        # Limit the number of concurrent workers.
        self.semaphore.acquire()

        # --- UNSAFE LOCKING ---
        # Manual acquire/release is risky. A `with` statement should be used.
        self.device.lock_hash[self.location].acquire()

        script_data = []

        # Gather data from neighbors and self.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # If data is available, run the script and propagate results.
        if script_data != []:
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)

        self.device.lock_hash[self.location].release()

        self.semaphore.release()
