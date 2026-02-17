"""
Models a distributed network of devices that execute scripts on sensor data.

This script provides an alternative implementation of a device network simulation.
Key concurrency patterns used here include:
- A classic producer-consumer model using Semaphores to manage a task queue between
  the main device thread and its workers.
- A two-phase reusable barrier to ensure robust synchronization between all devices
  at the end of each time step.
- An array of Locks for location-specific resource protection.
"""

from threading import Event, Thread, Lock, Semaphore
from reusable_barrier_semaphore import ReusableBarrier
import multiprocessing
import Queue


class Device(object):
    """
    Represents a single device in the simulated network.

    Each device manages its own sensor data and a pool of worker threads.
    Device 0 acts as a 'root' node, holding shared synchronization primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): The unique identifier for the device.
            sensor_data (dict): A dictionary of the device's local sensor data.
            supervisor (Supervisor): The network management object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.max = 0
        self.supervisor = supervisor
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.root = None
        self.barrier = None
        self.loc_lock = []
        
        # Determine the maximum location key to size the lock array.
        for key in self.sensor_data.keys():
            if self.max < key:
                self.max = key

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes network-wide settings and synchronization objects.

        This method identifies the root device (ID 0) and has it create the
        shared barrier and location-specific locks.

        Args:
            devices (list): A list of all Device objects in the network.
        """
        self.devices = devices
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
        for dev in devices:
            if dev.device_id == 0:
                self.root = dev
            if self.max < dev.max:
                self.max = dev.max
        # The root device creates a lock for each possible data location.
        if self.device_id == 0:
            for i in range(self.max + 1):
                self.loc_lock.append(Lock())

    def assign_script(self, script, location):
        """
        Assigns a script to the device for a specific location.

        A 'None' script signals that all scripts for the current timepoint
        have been assigned, triggering the `timepoint_done` event.

        Args:
            script: The script object to execute.
            location: The location key relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Returns:
            The data for the location, or None if not present.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device, acting as a "producer".

    It waits for scripts to be assigned for a timepoint, then "produces"
    script execution tasks and puts them onto a shared queue for the worker
    threads.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = []
        self.nr_workers = multiprocessing.cpu_count()
        self.workers = range(self.nr_workers)
        self.neighbours = None
        # Semaphores for a bounded-buffer producer-consumer queue.
        self.queue_empty = Semaphore(150) # Controls space in the queue.
        self.queue_full = Semaphore(0)   # Controls items in the queue.
        self.end_scripts = Event() # Signals that all workers have finished.
        self.count = 0
        self.count_lock = Lock()

    def run(self):
        """The main execution loop for the device thread."""
        for i in range(self.nr_workers):
            self.workers[i] = WorkerThread(self)
            self.workers[i].start()

        while True:
            # Fetches neighbors for the current timepoint.
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                # A 'None' neighbor list is the shutdown signal.
                break
            
            # Waits until all scripts for the timepoint are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            if len(self.device.scripts) > 0:
                # Sets up a counter for the number of scripts to be executed.
                self.count_lock.acquire()
                self.count = len(self.device.scripts)
                self.count_lock.release()

                # Produces tasks by adding scripts to the queue for workers.
                for (script, location) in self.device.scripts:
                    self.queue_empty.acquire()
                    self.queue.append((script, location))
                    self.queue_full.release()
                
                # Waits for a signal from the last worker that all tasks are done.
                self.end_scripts.wait()
                self.end_scripts.clear()
            
            # Invariant: All devices synchronize at the barrier before the next timepoint.
            self.device.root.barrier.wait()

        # Shutdown logic: send termination signal (None) to worker threads.
        for i in range(self.nr_workers):
            self.queue_empty.acquire()
            self.queue.append((None, None))
            self.queue_full.release()
        
        for i in range(self.nr_workers):
            self.workers[i].join()


class WorkerThread(Thread):
    """
    A worker thread that acts as a "consumer" of script execution tasks.
    """

    def __init__(self, master):
        """Initializes the worker."""
        Thread.__init__(self)
        self.master = master

    def run(self):
        """The main loop for the worker, consuming and executing tasks."""
        while True:
            # Waits for a task to be available in the queue.
            self.master.queue_full.acquire()
            (script, location) = self.master.queue.pop(0)
            self.master.queue_empty.release()
            
            if self.master.neighbours is None:
                # Shutdown signal received.
                break
            else:
                script_data = []
                # Invariant: Acquires a location-specific lock to ensure atomic
                # data access and updates across the entire network for this location.
                self.master.device.root.loc_lock[location].acquire()
                
                # Gathers data from neighbors and the local device.
                for device in self.master.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.master.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                # Executes the script and distributes the results.
                if script_data:
                    result = script.run(script_data)
                    self.master.device.set_data(location, result)
                    for device in self.master.neighbours:
                        device.set_data(location, result)
                self.master.device.root.loc_lock[location].release()
                
                # Logic to signal the master thread when all tasks for the
                # current timepoint are complete.
                self.master.count_lock.acquire()
                self.master.count -= 1
                if self.master.count == 0:
                    self.master.end_scripts.set()
                self.master.count_lock.release()

class ReusableBarrier():
    """
    A two-phase reusable barrier implementation using semaphores.

    This ensures that threads from one cycle of waits do not overlap with threads
    from the next, preventing race conditions.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters for each phase, stored in a list to be mutable.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        # Semaphores to block threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)
    
    def wait(self):
        """Makes the calling thread wait until all threads have reached the barrier."""
        # The barrier consists of two distinct phases to ensure reusability.
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
    
    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier.

        Args:
            count_threads (list): The counter for the current phase.
            threads_sem (Semaphore): The semaphore for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                # Resets the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        threads_sem.acquire() # All threads block here until released.


class MyThread(Thread):
    """
    An example thread class to demonstrate the usage of the ReusableBarrier. 
    
    This class is not part of the main device simulation logic.
    """
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
    
    def run(self):
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",