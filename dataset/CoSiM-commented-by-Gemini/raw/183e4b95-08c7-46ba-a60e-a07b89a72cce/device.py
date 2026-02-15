"""
This module defines a multi-threaded framework for a distributed device simulation.

It includes a Device class that manages a thread pool and a controller thread,
a custom ThreadPool and Worker implementation, and a custom ReusableBarrier
built with semaphores. The architecture coordinates script execution across
multiple simulated devices in synchronized time steps.
"""

from threading import Event, Thread, Lock , Condition
# The following classes are defined later in this same file.
# from queue import Worker, ThreadPool
# from reusable_barrier_semaphore import ReusableBarrier

class Device(object):
    """
    Represents a device node in the simulation, managing its own data,
    a controller thread, and a pool of worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device, its thread pool, and controller thread.
        
        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): The local data store for this device.
            supervisor (Supervisor): The central object managing the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        # Events for synchronizing the main device thread with script execution.
        self.script_received = Event()
        self.wait_neighbours = Event() # Signals workers that the neighbor list is ready.
        
        self.scripts = []
        self.neighbours = []
        self.allDevices = []
        self.locks = [] # A list of locks for location-based synchronization.
        self.pool = ThreadPool(8) # A pool of worker threads.
        self.lock = Lock()
        self.thread = DeviceThread(self) # The main controller thread for this device.
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the global synchronization objects for the simulation.
        
        This is intended to be run by a master device. It creates and distributes
        a global barrier and a list of location-based locks.
        """
        self.allDevices = devices
        self.barrier = ReusableBarrier(len(devices))

        # Initialize a fixed-size list of locks. This assumes locations are integer indices.
        for i in range(0, 50):
            self.locks.append(Lock())
        pass

    def assign_script(self, script, location):
        """
        Assigns a script to the device's thread pool for execution.
        
        Args:
            script (Script): The script to execute.
            location (any): The data location the script operates on. If script is None,
                            it signals the end of assignments for the current step.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Add the script execution task to the thread pool.
            self.pool.add_task(self.executeScript, script, location)
        else:
            # Signal that all scripts for the time step have been assigned.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves data from the device's local sensor data."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets data in the device's local sensor data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main controller thread to terminate."""
        self.thread.join()

    def executeScript(self, script, location):
        """
        The target function for worker threads. Executes a single script.
        
        This function gathers data from neighbors, runs the script, and writes
        the results back. It uses a location-based locking mechanism.
        """
        # Wait until the main thread has fetched the neighbor list for this step.
        self.wait_neighbours.wait()
        script_data = []

        # Gather data from neighbors, locking each one sequentially.
        if self.neighbours is not None:
            for device in self.neighbours:
                device.locks[location].acquire()
                data = device.get_data(location)
                device.locks[location].release()

                if data is not None:
                    script_data.append(data)

        # Gather data from the local device.
        self.locks[location].acquire()
        data = self.get_data(location)
        self.locks[location].release()

        if data is not None:
            script_data.append(data)

        # If data was gathered, run the script and write back the results.
        if script_data:
            result = script.run(script_data)

            # Write result to neighbors, locking each one sequentially.
            if self.neighbours is not None:
                for device in self.neighbours:
                    device.locks[location].acquire()
                    device.set_data(location, result)
                    device.locks[location].release()

            # Write result to local device.
            self.locks[location].acquire()
            self.set_data(location, result)
            self.locks[location].release()


class DeviceThread(Thread):
    """
    The main controller thread for a single device, orchestrating the
    simulation through time steps.
    """

    def __init__(self, device):
        """Initializes the controller thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main time-step loop for the device."""
        while True:
            # Reset events for the new time step.
            self.device.script_received.clear()
            self.device.wait_neighbours.clear()
            self.device.neighbours = []

            # Get the list of neighbors for the current step from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            # Unblock any worker threads waiting for the neighbor list.
            self.device.wait_neighbours.set()

            if self.device.neighbours is None:
                # Supervisor signaled end of simulation.
                self.device.pool.wait_completion()
                self.device.pool.terminateWorkers()
                self.device.pool.threadJoin()
                return

            # Re-queue all assigned scripts for the current step.
            # Note: This may be redundant if scripts are already queued in assign_script.
            for (script, location) in self.device.scripts:
                self.device.pool.add_task(self.device.executeScript, script, location)

            # Wait for the supervisor to signal that all scripts have been assigned.
            self.device.script_received.wait()
            # Wait for all tasks in the thread pool to complete for this step.
            self.device.pool.wait_completion()

            # Barrier synchronization: wait for all other devices to finish this step.
            for dev in self.device.allDevices:
                dev.barrier.wait()



from Queue import Queue
# Redefinition of Thread from threading module
# from threading import Thread

class Worker(Thread):
    """A worker thread that consumes tasks from a queue."""
    def __init__(self, tasks):
        Thread.__init__(self)
        self.tasks = tasks
        self.daemon = True
        self.terminate_worker = False
        self.start()

    def run(self):
        """The main loop for the worker, processing tasks from the queue."""
        while True:
            func, args, kargs = self.tasks.get()
            # A None function is a sentinel value to terminate the worker.
            if func is None:
                self.tasks.task_done()
                break
            try:
                func(*args, **kargs)
            except Exception as e:
                print(e)
            self.tasks.task_done()


class ThreadPool:
    """A simple thread pool implementation."""
    def __init__(self, num_threads):
        """Initializes the thread pool and creates worker threads."""
        self.tasks = Queue(99999)
        self.workers = []
        for _ in range(num_threads):
            self.workers.append(Worker(self.tasks))

    def add_task(self, func, *args, **kargs):
        """Adds a task to the queue for a worker to execute."""
        self.tasks.put((func, args, kargs))

    def wait_completion(self):
        """Blocks until all tasks in the queue are processed."""
        self.tasks.join()

    def terminateWorkers(self):
        """Sends a termination signal to all worker threads."""
        for _ in self.workers:
            self.tasks.put([None, None, None])
        for worker in self.workers:
            worker.terminate_worker = True

    def threadJoin(self):
        """Waits for all worker threads to complete."""
        for worker in self.workers:
            worker.join()

# Redefinition of threading primitives
# from threading import *

class ReusableBarrier():
    """
    A custom reusable barrier implemented using locks and semaphores.
    
    This implementation uses a two-phase approach to allow the barrier to be
    reused multiple times without race conditions.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)
    
    def wait(self):
        """
        Blocks the calling thread until all threads have reached the barrier.
        This is done in two phases to ensure reusability.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
    
    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the two-phase barrier.
        
        The last thread to enter the phase unlocks the semaphore for all
        other threads and resets the count for the next use of this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()
                                                 
# This MyThread class appears to be an unused example and is not part of the
# main Device simulation framework defined above.
class MyThread(Thread):
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
    
    def run(self):
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",