"""
This module provides a framework for simulating a distributed system of devices.

It uses a two-level threading model where a main thread per device dispatches
tasks to a pool of worker threads. Synchronization between devices is handled
by a centralized root device (device 0) that manages a global barrier and locks.
"""

from threading import Event, Thread, Lock, Semaphore
from reusable_barrier_semaphore import ReusableBarrier
import multiprocessing
import Queue


class Device(object):
    """
    Represents a single device in the distributed simulation.

    Each device has a master thread (`DeviceThread`) that manages a pool of
    worker threads. It holds sensor data and communicates with a supervisor
    to determine its neighbors.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of sensor data, keyed by location.
            supervisor: The supervisor object that provides neighborhood info.
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
        
        for key in self.sensor_data.keys():
            if self.max < key:
                self.max = key

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the simulation environment and shared synchronization objects.

        Device 0 is designated as the 'root' and is responsible for creating
        the main synchronization barrier and location-based locks that all
        other devices will use.

        Args:
            devices (list): A list of all devices participating in the simulation.
        """
        self.devices = devices
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
        for dev in devices:
            if dev.device_id == 0:
                self.root = dev
            if self.max < dev.max:
                self.max = dev.max
        if self.device_id == 0:
            for i in range(self.max + 1):
                self.loc_lock.append(Lock())

    def assign_script(self, script, location):
        """
        Assigns a script to be executed on this device.

        A 'None' script is a signal that all scripts for the current
        timepoint have been assigned.

        Args:
            script: The script object to execute.
            location (int): The location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data for a specific location if it exists on this device.

        Args:
            location (int): The location of the sensor data.

        Returns:
            The data if the location exists, otherwise None.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets the data for a specific location if it exists on this device.

        Args:
            location (int): The location of the sensor data.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its master thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    A master thread for a single Device, dispatching tasks to worker threads.

    This thread manages a producer-consumer queue where it acts as the producer,
    placing scripts into a queue for the worker threads to consume. It also handles
    synchronization between timepoints.
    """

    def __init__(self, device):
        """
        Initializes the master DeviceThread.

        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = []
        self.nr_workers = multiprocessing.cpu_count()
        self.workers = range(self.nr_workers)
        self.neighbours = None
        # Semaphores to manage the bounded buffer (worker queue).
        self.queue_empty = Semaphore(150)
        self.queue_full = Semaphore(0)
        self.end_scripts = Event() # Signals that all workers have finished.
        self.count = 0
        self.count_lock = Lock()

    def run(self):
        """Main execution loop for the master thread."""
        for i in range(self.nr_workers):
            self.workers[i] = WorkerThread(self)
        for i in range(self.nr_workers):
            self.workers[i].start()
        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None: # Supervisor signals shutdown.
                break
            
            # Wait until all scripts for the current timepoint are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            if len(self.device.scripts) > 0:
                
                # Produce tasks for the worker threads.
                with self.count_lock:
                    self.count = len(self.device.scripts)
                for (script, location) in self.device.scripts:
                    self.queue_empty.acquire()
                    self.queue.append((script, location))
                    self.queue_full.release()
                
                # Wait for all workers to complete their tasks for this timepoint.
                self.end_scripts.wait()
                self.end_scripts.clear()
            
            # Synchronize with all other devices before proceeding to the next timepoint.
            self.device.root.barrier.wait()

        
        # Shutdown sequence: send sentinel values to workers.
        for i in range(self.nr_workers):
            self.queue_empty.acquire()
            self.queue.append((None, None))
            self.queue_full.release()
        
        for i in range(self.nr_workers):
            self.workers[i].join()


class WorkerThread(Thread):
    """
    A worker thread that executes scripts.

    It consumes scripts from a queue managed by its master `DeviceThread`,
    executes them, and updates data on its local device and neighbors.
    """

    def __init__(self, master):
        """
        Initializes the WorkerThread.

        Args:
            master (DeviceThread): The master thread that manages this worker.
        """
        Thread.__init__(self)
        self.master = master

    def run(self):
        """Main execution loop for the worker thread."""
        while True:
            
            # Consume a task from the queue.
            self.master.queue_full.acquire()
            (script, location) = self.master.queue.pop()
            self.master.queue_empty.release()
            
            if self.master.neighbours is None: # Shutdown signal.
                break
            else:
                script_data = []
                # Acquire a global lock for the specific location.
                self.master.device.root.loc_lock[location].acquire()
                
                # Gather data from neighbors and the local device.
                for device in self.master.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                data = self.master.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                # Execute script and update data on all relevant devices.
                if script_data:
                    result = script.run(script_data)
                    self.master.device.set_data(location, result)
                    for device in self.master.neighbours:
                        device.set_data(location, result)
                self.master.device.root.loc_lock[location].release()
                
                # Atomically decrement the counter of active scripts.
                with self.master.count_lock:
                    self.master.count -= 1
                    # If this is the last script, signal the master thread.
                    if self.master.count == 0:
                        self.master.end_scripts.set()

# NOTE: The following is a duplicate implementation of a ReusableBarrier.
# It seems to be unused test or remnant code. The main simulation appears
# to use the version imported from 'reusable_barrier_semaphore'.
from threading import *

class ReusableBarrier():
    """A reusable barrier for thread synchronization. Duplicated in this file."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        """Causes a thread to wait at the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
    
    def phase(self, count_threads, threads_sem):
        """An internal phase of the barrier wait."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            


                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    
                                                 

class MyThread(Thread):
    """An example thread class, not used in the main simulation."""
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
    
    def run(self):
        """Example run method."""
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",