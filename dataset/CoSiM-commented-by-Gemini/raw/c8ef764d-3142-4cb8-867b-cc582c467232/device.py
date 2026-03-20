"""
This module provides a framework for simulating a distributed network of devices.

The simulation uses a multi-threaded approach where each `Device` has a main
control thread (`DeviceThread`) that spawns worker threads (`MyThread`) to execute
scripts. A `ReusableBarrier` (implemented with semaphores) is used for global
synchronization between devices, and a combination of a semaphore and location-based
locks are used to manage concurrency within and between devices.
"""

from threading import Semaphore, Event, Lock, Thread

class ReusableBarrier(object):
    """A reusable barrier for thread synchronization, implemented using semaphores.

    This barrier ensures that a fixed number of threads wait for each other at a
    synchronization point before any of them are allowed to continue. It uses a
    two-phase protocol to be reusable across multiple synchronization steps.
    Note: This implementation is for Python 2, as indicated by `xrange`.
    """

    def __init__(self, num_threads):
        """Initializes the barrier for a specific number of threads."""
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes the calling thread to block until all threads reach the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the two-phase barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread has arrived; release all waiting threads for this phase.
                for _ in xrange(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase of the two-phase barrier synchronization."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # The last thread has arrived; release all waiting threads and reset.
                for _ in xrange(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """Represents a single device in the distributed simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a device and its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.none_script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.timepoint_end = 0
        self.barrier = None
        self.lock_hash = None

    def __str__(self):
        return "Device %d" % self.device_id

    def set_barrier(self, barrier):
        """Assigns a global synchronization barrier to the device."""
        self.barrier = barrier

    def set_locks(self, lock_hash):
        """Assigns a hash map of location-based locks to the device."""
        self.lock_hash = lock_hash

    def setup_devices(self, devices):
        """Sets up shared synchronization primitives for all devices.
        
        This method is intended to be called by a single "master" device
        (the one with the minimum ID) to create and distribute the barrier
        and locks.
        """
        ids_list = [dev.device_id for dev in devices]

        if self.device_id == min(ids_list):
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
        """Assigns a script to the device for the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A `None` script signals that all scripts for the time step have been assigned.
            self.none_script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a device."""

    def __init__(self, device):
        """Initializes the control thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # This semaphore limits the number of concurrently executing script threads for this device.
        self.semaphore = Semaphore(value=8)

    def run(self):
        """The main execution loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break  # End of simulation

            # Wait for the signal that all scripts for the time step have been received.
            self.device.none_script_received.wait()
            self.device.none_script_received.clear()

            thread_list = []

            # Spawn a worker thread for each script.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, neighbours, script, location,
                    self.semaphore)
                thread.start()
                thread_list.append(thread)

            # Wait for all worker threads for this time step to complete.
            for i in xrange(len(thread_list)):
                thread_list[i].join()

            # Synchronize with all other devices before starting the next time step.
            self.device.barrier.wait()

class MyThread(Thread):
    """A worker thread responsible for executing a single script."""

    def __init__(self, device, neighbours, script, location, semaphore):
        """Initializes a script-executing worker thread."""
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.semaphore = semaphore

    def run(self):
        """The execution logic for the worker thread."""
        # Acquire the semaphore to limit concurrency.
        self.semaphore.acquire()

        # Acquire the lock for the specific data location to ensure data consistency.
        self.device.lock_hash[self.location].acquire()

        script_data = []

        # Gather data from neighboring devices.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        # Gather data from the local device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Run the script on the collected data.
            result = self.script.run(script_data)

            # Distribute the result to all relevant devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)

        # Release the location-specific lock.
        self.device.lock_hash[self.location].release()

        # Release the semaphore to allow another worker to run.
        self.semaphore.release()
