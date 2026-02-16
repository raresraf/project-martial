"""
This module implements a distributed device simulation using a custom reusable
barrier and a manual thread-pooling strategy where 'Slave' threads are created
on-demand for each simulation step.
"""

from threading import Lock, Thread, Event, Semaphore
import math
from fractions import Fraction


class ReusableBarrier(object):
    """
    A reusable, two-phase barrier implemented with semaphores and a lock.

    This implementation is functionally correct. It uses a list to hold the
    thread counters, which is an uncommon but valid way to create a mutable
    integer reference that can be modified by the 'phase' method.
    """
    def __init__(self, threads):
        self.threads = threads
        # Using a list to hold a mutable integer counter.
        self.count_threads1 = [self.threads]
        self.count_threads2 = [self.threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to block until all threads have called this method."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single synchronization phase of the barrier.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all other waiting threads.
                for i in range(self.threads):
                    threads_sem.release()
                    i = i # This assignment has no effect.
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.threads
        # All threads block here until released.
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device in the simulation.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.barrier = None
        # A shared list of locks, one for each potential data location.
        self.locations_lock = []
        self.lock = Lock()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs centralized setup, creating and distributing shared resources.
        This is intended to be run only by the master device (id 0).
        """
        # NOTE: The number of locations is hardcoded to 50, which could cause
        # an IndexError if a location >= 50 is used.
        locations = 50

        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            for i in range(locations):
                self.locations_lock.append(Lock())
                i = i # This assignment has no effect.
            # Distribute the shared barrier and locks to all other devices.
            for device in devices:
                device.barrier = barrier
                device.locations_lock = self.locations_lock

    def assign_script(self, script, location):
        """Assigns a script to be run in the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Gets data from a local sensor location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets data at a local sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class Slave(Thread):
    """
    A short-lived worker thread that executes a batch of scripts.
    """
    def __init__(self, neighbours, device):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.device = device
        self.scripts = []

    def give_work(self, work):
        """Assigns a script to this slave's work queue before it starts."""
        self.scripts.append(work)

    def run(self):
        """
        Executes all assigned scripts, using location-based locking for safety.
        """
        for (script, location) in self.scripts:
            # The 'with' statement ensures the lock is acquired and released properly.
            # This is a correct and safe way to prevent race conditions on a location.
            with self.device.locations_lock[location]:
                script_data = []
                # Block Logic: Gather data from neighbors and self.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Execute script and disseminate results if data was found.
                if script_data:
                    result = script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)


class DeviceThread(Thread):
    """
    The main control thread for a device. It creates a pool of Slave threads
    to execute scripts for each time step.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.slave_pool = []

    def run(self):
        """The main loop for the device's lifecycle."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation
            
            # Wait for the signal that all scripts for this step have been assigned.
            self.device.timepoint_done.wait()
            
            num_threads = 8 # Max number of slave threads to create.
            self.slave_pool = []
            
            # Block Logic: Create a pool of slave threads.
            # The pool size is the smaller of 8 or the number of available scripts.
            for i in range(min(num_threads, len(self.device.scripts))):
                helper = Slave(neighbours, self.device)
                self.slave_pool.append(helper)
            
            # Block Logic: Distribute scripts to the slaves. This logic is complex
            # and potentially buggy, aiming for an even distribution of work.
            equally_work = 0
            i = 0
            if len(self.slave_pool):
                for (script, location) in self.device.scripts:
                    self.slave_pool[i].give_work((script, location))
                    equally_work += 1
                    
                    if equally_work == math.ceil(Fraction(len(self.device.scripts), num_threads)):
                        i += 1
                        equally_work = 0
                    # This reset is likely a bug, as it would cause work to be
                    # unevenly assigned to the first slave.
                    if i == len(self.slave_pool):
                        i = 0
                        equally_work = 0
                
                # Start all slave threads and wait for them to complete.
                for slv in range(len(self.slave_pool)):
                    self.slave_pool[slv].start()
                
                for cls in range(len(self.slave_pool)):
                    self.slave_pool[cls].join()

            self.device.timepoint_done.clear()
            # Synchronize with all other devices before the next time step.
            self.device.barrier.wait()
