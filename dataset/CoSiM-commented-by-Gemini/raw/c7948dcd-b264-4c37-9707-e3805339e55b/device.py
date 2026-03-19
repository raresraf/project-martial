"""
@file raw/c7948dcd-b264-4c37-9707-e3805339e55b/device.py
@brief Implements a distributed device simulation using an intra-device
       producer-consumer model and centralized locking.

This module simulates a network of devices using a "process-then-synchronize"
model. Its concurrency architecture is notable for two features:
1.  **Intra-Device Producer-Consumer:** Each `Device` has a main `DeviceThread`
    that acts as a producer, placing tasks into a bounded buffer (a list
    protected by semaphores). A pool of `WorkerThread`s acts as consumers,
    executing these tasks in parallel.
2.  **Centralized Locking:** A single "root" device (id=0) creates and owns an
    array of `Lock` objects for all possible data locations. All worker threads,
    regardless of their parent device, must access `device.root.loc_lock` to
    acquire the global lock for a location before processing data.
"""

from threading import Event, Thread, Lock, Semaphore
# This external module is expected to provide a correct ReusableBarrier.
from reusable_barrier_semaphore import ReusableBarrier
import multiprocessing
import Queue


class Device(object):
    """
    Represents a device node. The device with id=0 acts as the "root" node,
    managing the global barrier and the centralized lock array.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.max = 0 # The highest location ID this device knows about initially.
        self.supervisor = supervisor
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.root = None
        self.barrier = None
        self.loc_lock = [] # Only populated and used by the root device.
        
        # Pre-calculate the max location ID for this device.
        for key in self.sensor_data.keys():
            if self.max < key:
                self.max = key

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects. The root
        device creates the barrier and the central lock array. All other
        devices get a reference to the root device to access them.
        """
        self.devices = devices
        if self.device_id == 0:
            # As root, create the shared barrier.
            self.barrier = ReusableBarrier(len(devices))
        
        # All devices discover the root and the global max location ID.
        for dev in devices:
            if dev.device_id == 0:
                self.root = dev
            if self.max < dev.max:
                self.max = dev.max

        if self.device_id == 0:
            # As root, create the centralized array of locks.
            for i in range(self.max + 1):
                self.loc_lock.append(Lock())

    def assign_script(self, script, location):
        """Assigns a script to be run for a specific location."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

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
    """
    The main control thread for a Device. Acts as a Producer that feeds tasks
    to a pool of WorkerThreads.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # The bounded buffer is implemented as a list protected by two semaphores.
        self.queue = []
        self.nr_workers = multiprocessing.cpu_count()
        self.workers = range(self.nr_workers)
        self.neighbours = None
        self.queue_empty = Semaphore(150) # Counts empty slots in the buffer.
        self.queue_full = Semaphore(0)    # Counts filled slots in the buffer.
        self.end_scripts = Event() # Signaled by the last worker when the batch is done.
        self.count = 0 # Number of scripts to be processed in the current step.
        self.count_lock = Lock()

    def run(self):
        """Main simulation loop."""
        # Create and start the pool of worker (consumer) threads.
        for i in range(self.nr_workers):
            self.workers[i] = WorkerThread(self)
            self.workers[i].start()
            
        while True:
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break # End of simulation.
            
            # 1. Wait for scripts for the current time step to be assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            if self.device.scripts:
                # 2. Produce tasks: add all scripts to the bounded buffer for workers.
                with self.count_lock:
                    self.count = len(self.device.scripts)
                for (script, location) in self.device.scripts:
                    self.queue_empty.acquire() # Wait for an empty slot.
                    self.queue.append((script, location))
                    self.queue_full.release()  # Signal that a slot is now full.
                
                # 3. Wait for workers to signal that all scripts are processed.
                self.end_scripts.wait()
                self.end_scripts.clear()
            
            # 4. Synchronize with all other devices at the global barrier.
            self.device.root.barrier.wait()

        # --- Shutdown ---
        # Signal worker threads to terminate.
        for i in range(self.nr_workers):
            self.queue_empty.acquire()
            self.queue.append((None, None))
            self.queue_full.release()
        
        for i in range(self.nr_workers):
            self.workers[i].join()


class WorkerThread(Thread):
    """
    A long-running consumer thread that executes tasks from the bounded buffer.
    """

    def __init__(self, master):
        Thread.__init__(self)
        self.master = master # The parent DeviceThread.

    def run(self):
        while True:
            # 1. Consume a task from the bounded buffer.
            self.master.queue_full.acquire() # Wait for a full slot.
            
            # This check is needed to prevent a crash on shutdown.
            if self.master.neighbours is None:
                break

            (script, location) = self.master.queue.pop(0)
            self.master.queue_empty.release() # Signal that a slot is now empty.
            
            # 2. Process the task.
            script_data = []
            # Acquire the global, centralized lock for this location.
            self.master.device.root.loc_lock[location].acquire()
            
            # Aggregate data from neighbors and the local device.
            for device in self.master.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            data = self.master.device.get_data(location)
            if data is not None:
                script_data.append(data)
            
            # Execute script and broadcast results.
            if script_data:
                result = script.run(script_data)
                self.master.device.set_data(location, result)
                for device in self.master.neighbours:
                    device.set_data(location, result)

            self.master.device.root.loc_lock[location].release()
            
            # 3. Decrement the work counter and signal the master if all work is done.
            with self.master.count_lock:
                self.master.count -= 1
                if self.master.count == 0:
                    self.master.end_scripts.set()


# --- The following classes appear to be unused/dead code from development ---

class ReusableBarrier():
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
    
    def phase(self, count_threads, threads_sem):
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()                    
                                                 
class MyThread(Thread):
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
    
    def run(self):
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "
",
