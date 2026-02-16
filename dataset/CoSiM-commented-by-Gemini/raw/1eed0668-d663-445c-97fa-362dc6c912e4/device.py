"""
This module implements a highly complex, custom-architected simulation of a
distributed device network.

Each `Device` has a main `DeviceThread` which in turn manages its own private,
persistent pool of `WorkerThread`s. The `DeviceThread` acts as a dispatcher,
coordinating with its workers and the main script-assignment process using a
complex system of three different Semaphores and a deque. While functionally
sophisticated, this coordination logic is difficult to follow compared to
standard library solutions like `queue.Queue`.
"""

from threading import Event, Thread, Lock, Semaphore
from collections import deque

class ReusableBarrierSem(object):
    """
    A correct, two-phase reusable barrier for thread synchronization.
    This is included here as it is imported by the original code.
    """
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
    Represents a device in the simulation, which owns and manages a DeviceThread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.setup_done = Event() 
        # Semaphore released by `assign_script` to signal a new script is ready.
        self.script_semaphore = Semaphore(0) 
        self.location_locks = [] 
        # A queue to signal the total number of scripts for a time step.
        self.queue = deque() 
        self.thread = None
        self.barrier = None

    def __str__(self):
        return "Device %d" % self.device_id

    def start_thread(self, barrier, locks):
        """Creates and starts the main DeviceThread after setup is complete."""
        self.thread = DeviceThread(self)
        self.barrier = barrier
        self.location_locks = locks
        self.thread.start()
        self.setup_done.set() 

    def setup_devices(self, devices):
        """
        Master-driven setup (device 0) to create and distribute shared resources.
        """
        if self.device_id == 0:
            barrier = ReusableBarrierSem(len(devices))
            # Identifies all unique locations to create a shared list of locks.
            locks = []
            for device in devices:
                for location in device.sensor_data:
                    # This check is not thread-safe but works if locations are unique
                    if location not in [loc for loc, _ in locks]:
                        locks.append((location, Lock()))
            
            for device in devices:
                device.start_thread(barrier, locks)

    def assign_script(self, script, location):
        """
        Adds a script to the list and signals the dispatcher thread.
        A 'None' script signals the end of assignments for the time step.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Enqueue the total number of scripts as the end-of-batch signal.
            self.queue.append(len(self.scripts))
        self.script_semaphore.release()

    def get_data(self, location):
        """Retrieves data from the local sensor dictionary."""
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        """Sets data in the local sensor dictionary."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.setup_done.wait()
        self.thread.join()


class WorkerThread(Thread):
    """
    A persistent worker thread that executes tasks given by its parent DeviceThread.
    """
    def __init__(self, device_thread):
        Thread.__init__(self)
        self.device_thread = device_thread

    def run(self):
        """Continuously fetches and executes tasks."""
        while True:
            # Blocks until the dispatcher releases the semaphore, signaling a new task.
            self.device_thread.threads_semaphore.acquire()
            
            script = None
            location = None
            # The dispatcher's queue is not thread-safe; this relies on the
            # semaphore ensuring only one worker accesses it at a time.
            if len(self.device_thread.scripts_queue) > 0:
                (script, location) = self.device_thread.scripts_queue.popleft()
            
            if location is None:
                break # Shutdown signal

            # Inefficiently finds the lock for the location by linear search.
            lock = next(l for (x, l) in self.device_thread.device.location_locks
                if x == location)

            lock.acquire()
            # --- Start of Critical Section ---
            script_data = []
            for device in self.device_thread.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = script.run(script_data)
                for device in self.device_thread.neighbours:
                    device.set_data(location, result)
                self.device_thread.device.set_data(location, result)
            # --- End of Critical Section ---
            lock.release()
            
            # Signal to the dispatcher that one task has been completed.
            self.device_thread.worker_semaphore.release()


class DeviceThread(Thread):
    """
    The main dispatcher thread for a device. Manages a pool of worker threads
    and coordinates a complex, multi-semaphore workflow.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Semaphore for workers to wait for tasks.
        self.threads_semaphore = Semaphore(0)
        self.scripts_queue = deque()
        self.worker_threads = [] 
        self.neighbours = [] 
        # Semaphore for the dispatcher to wait for workers to finish.
        self.worker_semaphore = Semaphore(0)
        self.nr_threads = 8 

        # Create and start the persistent pool of worker threads.
        for _ in xrange(self.nr_threads):
            thread = WorkerThread(self)
            self.worker_threads.append(thread)
            thread.start()

    def run(self):
        """The main, complex dispatcher loop."""
        index = 0
        while True:
            # Block until all workers from the *previous* time step have finished.
            for _ in xrange(index):
                self.worker_semaphore.acquire()
            
            self.device.barrier.wait()
            index = 0
            stop = None
            
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break # End of simulation.

            # Inner loop to dispatch all tasks for the current time step.
            while True:
                # If we've run out of known scripts, wait for a signal from assign_script.
                if not len(self.device.scripts) > index:
                    self.device.script_semaphore.acquire()
                
                # Check if the end-of-batch signal has been received.
                if stop is None:
                    if len(self.device.queue) > 0:
                        stop = self.device.queue.popleft()
                
                # If all scripts for this batch have been dispatched, exit inner loop.
                if stop is not None and stop == index:
                    break
                
                if stop is None and not len(self.device.scripts) > index:
                    continue

                (script, location) = self.device.scripts[index]
                
                self.scripts_queue.append((script, location))
                self.threads_semaphore.release() # Wake up one worker.
                index += 1
        
        # Shutdown: Wake up all workers so they can exit their loops.
        for _ in xrange(len(self.worker_threads)):
            self.threads_semaphore.release()
        for thread in self.worker_threads:
            thread.join()
