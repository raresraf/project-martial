"""
This module provides another implementation of a multi-threaded device simulation.

It defines a `Device` that runs scripts on sensor data and synchronizes with
other devices using a barrier. This version includes the barrier implementation
locally.

NOTE: This implementation contains several severe concurrency issues, including
a broken locking mechanism for resource location access, thread-unsafe data
get/set methods, and a potentially racy barrier setup. The threading model is
also highly inefficient, creating new threads for every task in every time step.
"""


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """
    A reusable barrier implementation using two semaphores.
    
    Allows a set number of threads to synchronize at a point in code. It uses
    a two-phase protocol to allow for reuse in loops.
    """
    
    def __init__(self, num_threads):
        """Initializes the barrier for a specific number of threads."""
        self.num_threads = num_threads
        # Using single-element lists to make integers mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.counter_lock = Lock()
        # The two semaphores representing the "turnstiles" for each phase.
        self.threads_sem1 = Semaphore(0) 
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        """Makes the calling thread wait at the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.counter_lock:
            count_threads[0] -= 1
            # The last thread to arrive resets the counter and releases all other threads.
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        # All threads block here until the last one has released the semaphore.
        threads_sem.acquire()

class Device(object):
    """
    Represents a device node in the simulation.
    
    This class has significant concurrency flaws in its locking and setup logic.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # NOTE: This appears unused.
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)

        # --- Shared and state attributes ---
        self.neighbours = []
        self.barrier = None
        self.threads = []
        # BUG: This list of locks is not properly shared between devices.
        self.locks = [None] * 100

        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for all devices.
        
        BUG: This setup is racy. If multiple devices call this concurrently,
        they may create different barrier objects, breaking synchronization.
        A single, non-racy setup call (e.g., from a main thread before
        devices are started) is required.
        """
        if self.barrier is None:
            barrier = ReusableBarrierSem(len(devices))
            self.barrier = barrier
            for d in devices:
                if d.barrier is None:
                    d.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device.
        
        BUG: This method creates a *new* Lock object for a location every time
        a script is assigned. It then tries to propagate this new lock to its
        neighbors. This is fundamentally incorrect. For mutual exclusion, all
        threads/devices needing access to a shared resource (the location) must
        share the *exact same* Lock object. This implementation fails to do so.
        """
        if script is not None:
            self.scripts.append((script, location))
            
            # This creates a new, unshared lock. It provides no protection against other devices.
            self.locks[location] = Lock()
            
            for device in self.neighbours:
                # This gives the neighbor its own separate lock. No synchronization is achieved.
                device.locks[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data.
        
        BUG: This method is not thread-safe. It has no lock, so it can race with
        `set_data`, leading to corrupt reads or other concurrency problems.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data.
        
        BUG: This method is not thread-safe. It has no lock, so it can race with
        `get_data` or other calls to `set_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

class MyThread(Thread):
    """A short-lived worker thread to execute one script."""
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """Acquires a lock, processes data, and releases the lock."""
        # This lock is likely not the same object as the locks on other devices,
        # providing no real mutual exclusion between them.
        self.device.locks[self.location].acquire()
        
        script_data = []
        
        # Gather data from neighbors.
        for device in self.neighbours:
            data = device.get_data(self.location) # Unsafe read
            if data is not None:
                script_data.append(data)
            
        # Gather data from self.
        data = self.device.get_data(self.location) # Unsafe read
        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = self.script.run(script_data)
            
            # Propagate results.
            for device in self.neighbours:
                device.set_data(self.location, result) # Unsafe write
            
            self.device.set_data(self.location, result) # Unsafe write
            
        self.device.locks[self.location].release()


class DeviceThread(Thread):
    """The main control thread for a device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal from supervisor.

            # Wait for the signal that all scripts for the time step have been assigned.
            self.device.timepoint_done.wait()

            self.device.neighbours = neighbours

            # --- Inefficient Threading Model ---
            # Create a new thread for each script. This is costly.
            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, location, script, neighbours)
                self.device.threads.append(thread)
            
            # Start and immediately join all threads. This makes script execution
            # parallel within the device, but the device as a whole waits for
            # all its scripts to finish before proceeding.
            for thread in self.device.threads:
                thread.start()
            for thread in self.device.threads:
                thread.join()
            
            # Clear the list of worker threads for the next time step.
            self.device.threads=[]
            
            # Clear the event and wait at the barrier for other devices.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
