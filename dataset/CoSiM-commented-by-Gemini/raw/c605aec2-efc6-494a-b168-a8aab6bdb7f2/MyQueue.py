"""
@file raw/c605aec2-efc6-494a-b168-a8aab6bdb7f2/MyQueue.py
@brief Implements a flawed simulation of a distributed, synchronous device network.

This module defines a system for simulating a network of devices that process
data in parallel and synchronize at the end of each time step. Each device uses
its own thread pool (`MyQueue`) to process assigned scripts.

@warning
This implementation contains a critical design flaw in its locking mechanism.
The `Device.get_data` method acquires a lock but never releases it, and the
`Device.set_data` method releases a lock it does not own. This will inevitably
lead to deadlocks, preventing the simulation from running correctly. The
documentation describes the intended logic, but also notes these critical bugs.
"""
from Queue import Queue
from threading import Thread, Event, Lock, Semaphore

class MyQueue():
    """
    A multi-threaded worker pool for processing tasks from a queue.
    In this simulation, each Device has its own MyQueue instance to process
    its scripts in parallel.
    """
    def __init__(self, num_threads):
        """
        Initializes the queue and starts the worker threads.

        Args:
            num_threads (int): The number of worker threads in the pool.
        """
        self.queue = Queue(num_threads)
        self.threads = []
        # CRITICAL DESIGN FLAW: This `device` attribute is set externally,
        # creating a tight coupling with the DeviceThread class.
        self.device = None

        for _ in xrange(num_threads):
            thread = Thread(target=self.run)
            self.threads.append(thread)
        
        for thread in self.threads:
            thread.start()
    
    def run(self):
        """
        The target function for the worker threads.
        It continuously fetches tasks from the queue, processes them, and waits
        for the next task.
        """
        while True:
            # A task is a tuple of (neighbours, script, location)
            neighbours, script, location = self.queue.get()

            # A (None, None, None) tuple is a sentinel value to terminate the thread.
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
            
            # Gather data from the local device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Execute the script on the aggregated data.
                result = script.run(script_data)
                
                # Broadcast the result back to all neighbors.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                # Set the result on the local device.
                self.device.set_data(location, result)
            
            self.queue.task_done()
    
    def finish(self):
        """
        Shuts down the worker pool gracefully.
        Waits for all tasks to be completed, then sends a sentinel value to each
        thread to terminate it.
        """
        self.queue.join()

        for _ in xrange(len(self.threads)):
           self.queue.put((None, None, None))

        for thread in self.threads:
            thread.join()

class ReusableBarrier():
    """
    A reusable barrier implemented using semaphores and a lock.
    Allows a set of threads to wait for each other to reach a common point.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Using a list to hold the counter is a way to have a mutable integer
        # that can be modified across method calls passed as an argument.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)
 
    def wait(self):
        """Forces threads to wait until all have reached the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """A single phase of the two-phase barrier."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread has arrived, release all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of the barrier.
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """
    Represents a single device in the network simulation.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        # Creates a dictionary of locks, one for each sensor location.
        self.location_locks = {location: Lock() for location in self.sensor_data}
        self.scripts_available = False
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the global barrier and distributes it to all devices.
        Intended to be called from a single 'master' device (id 0).
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script for a given location."""
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_available = True
        else:
            # A `None` script signals the end of assignments for a time step.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.
        @warning FLAWED IMPLEMENTATION: This method acquires a lock but
                 never releases it, which will cause a deadlock.
        """
        if location in self.sensor_data:
            self.location_locks[location].acquire()
            return self.sensor_data[location]     
        else:
            return None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location.
        @warning FLAWED IMPLEMENTATION: This method releases a lock that it
                 never acquired. This breaks the locking protocol.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.location_locks[location].release()
        else:
            return None

    def shutdown(self):
        """Shuts down the device's main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """The main control thread for a device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Each device has its own private worker pool.
        self.queue = MyQueue(8)

    def run(self):
        """
        Main execution loop. Implements a "process-then-synchronize" model.
        """
        # Poor practice: Mutating an attribute of another object directly.
        self.queue.device = self.device
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.
            
            # This is a busy-wait loop to check for work.
            while True:
                if self.device.scripts_available or self.device.timepoint_done.wait():
                    if self.device.scripts_available:
                        self.device.scripts_available = False
                        # Dispatch all assigned scripts to the worker queue.
                        for (script, location) in self.device.scripts:
                            self.queue.queue.put((neighbours, script, location))
                    else:
                        # End of script assignments for this time step.
                        self.device.timepoint_done.clear()
                        self.device.scripts_available = True
                        break
            
            # Wait for the device's own worker pool to finish all tasks.
            self.queue.queue.join()
            # Wait at the global barrier for all other devices to finish their work.
            self.device.barrier.wait()

        # Cleanly shut down the worker pool.
        self.queue.finish()
