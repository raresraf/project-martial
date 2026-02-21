"""
A multi-threaded, distributed device simulation framework.

This module contains several classes that together form a simulation of
devices operating in a network. It features a thread pool worker pattern
(`MyQueue`), a custom two-phase reusable barrier (`ReusableBarrier`), and a
`Device` class that acts as the main actor, driven by a `DeviceThread`.

NOTE: This file appears to contain multiple classes that were likely intended
to be in separate modules, as suggested by the local `from MyQueue import MyQueue`
import. It is documented here as a single module.
"""

from Queue import Queue 
from threading import Thread

class MyQueue():
    """A worker pool that processes tasks from a shared queue.

    This class creates a fixed number of worker threads that continuously pull
    tasks from a `Queue.Queue` and execute them. It is designed to process
    computational scripts for a device in parallel.
    """
    def __init__(self, num_threads):
        """Initializes the worker pool with a set number of threads."""
        self.queue = Queue(num_threads)
        self.threads = []
        self.device = None

        # Create and start the worker threads.
        for _ in xrange(num_threads):
            thread = Thread(target=self.run)
            self.threads.append(thread)
        
        for thread in self.threads:
            thread.start()
    
    def run(self):
        """The target function for worker threads.
        
        Continuously fetches tasks from the queue. A task consists of a script
        to run and the context (neighbors, location) for the script. A sentinel
        value of (None, None, None) signals the thread to terminate.
        """
        while True:
            # Blocks until a task is available.
            neighbours, script, location = self.queue.get()

            # Sentinel value check for graceful shutdown.
            if neighbours is None and script is None:
                self.queue.task_done()
                return
        
            script_data = []
            # Gather data from neighboring devices.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            # Include the device's own data.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Execute the script on the aggregated data.
                result = script.run(script_data)
                
                # Disseminate the result back to the neighborhood.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                self.device.set_data(location, result)
            
            self.queue.task_done()
    
    def finish(self):
        """Shuts down the worker pool gracefully.

        Waits for all tasks in the queue to be processed, then puts sentinel
        values in the queue to terminate each worker thread, and finally joins
        all threads.
        """
        # Wait for all enqueued tasks to be completed.
        self.queue.join()

        # Send a sentinel value for each thread to signal termination.
        for _ in xrange(len(self.threads)):
           self.queue.put((None, None, None))

        # Wait for all worker threads to finish.
        for thread in self.threads:
            thread.join()


from threading import Thread, Event, Lock, Semaphore
# This local import suggests the file was intended to be split.
from MyQueue import MyQueue

class ReusableBarrier():
    """A custom, reusable barrier for thread synchronization.

    This implementation uses a two-phase signaling protocol with semaphores to
    ensure that no thread can start a new phase of a barrier wait before all
    threads have completed the previous phase. This prevents race conditions
    in cyclic or iterative algorithms.
    """
    def __init__(self, num_threads):
        """Initializes the barrier for a fixed number of threads."""
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
 
    def wait(self):
        """Causes a thread to block until all `num_threads` have called wait.
        
        The wait is implemented in two distinct phases to make the barrier safely
        reusable across multiple synchronization points.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """Executes one phase of the two-phase barrier protocol."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0: # The last thread to arrive
                # Release all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()        
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads  
        # All threads block here until the last thread releases the semaphore.
        threads_sem.acquire()

class Device(object):
    """Represents a device actor in the simulation.
    
    This version of the Device class manages its own lifecycle via a
    `DeviceThread` and uses a simplified master-slave setup for initialization.
    
    WARNING: The locking mechanism in `get_data` and `set_data` is critically
    flawed. `get_data` acquires a lock that it never releases, and `set_data`
    releases a lock that it never acquired. This will lead to deadlocks or
    unexpected behavior.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()


        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        # Creates a dictionary of locks, one for each data location.
        self.location_locks = {location: Lock() for location in self.sensor_data}
        self.scripts_available = False
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Configures the device group, with device 0 acting as master.
        
        The master device (hardcoded as device_id 0) creates a ReusableBarrier
        and distributes it to all other devices in the group.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to be executed in the next time step."""
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Acquires a lock and returns data from a location.
        
        WARNING: This method acquires a lock but does not release it, which
        is a bug that will cause deadlocks. It relies on a corresponding
        `set_data` call to release the lock.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]     
        else:
            return None

    def set_data(self, location, data):
        """Sets data at a location and releases a lock.
        
        WARNING: This method releases a lock it did not acquire, which
        is a bug. It is intended to be paired with `get_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
        else:
            return None

    def shutdown(self):
        """Gracefully shuts down the device."""
        self.thread.join()

class DeviceThread(Thread):
    """The main control thread for a Device instance.

    This thread manages a `MyQueue` worker pool to execute scripts and
    synchronizes with other devices using a global barrier at the end of
    each time step.
    """
    def __init__(self, device):
        """Initializes the thread and its local worker pool."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = MyQueue(8) # Each device has its own 8-thread worker pool.

    def run(self):
        """The main, complex lifecycle loop for the device."""
        self.queue.device = self.device
        while True:
            # Get neighbors for the current time step.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Simulation exit condition.

            # This inner loop waits for scripts or a timepoint completion signal.
            while True:
                if self.device.scripts_available or self.device.timepoint_done.wait():
                    if self.device.scripts_available:
                        self.device.scripts_available = False

                        # Dispatch all available scripts to the worker queue.
                        for (script, location) in self.device.scripts:
                            self.queue.queue.put((neighbours, script, location))
            
                    else:
                        # This branch is taken when timepoint_done is set.
                        self.device.timepoint_done.clear()
                        self.device.scripts_available = True
                        break # Exit inner loop to proceed to barrier sync.
            
            # Wait for all local script executions to complete.
            self.queue.queue.join()
            # Synchronize with all other devices before the next time step.
            self.device.barrier.wait()

        # Cleanly shut down the local worker pool.
        self.queue.finish()
