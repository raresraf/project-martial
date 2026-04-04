"""
This module defines a distributed device simulation framework utilizing a
dynamic work-stealing model for its internal thread pool.

A coordinator device (ID 0) is responsible for setting up global synchronization
primitives. Each device uses a thread pool where workers pull tasks from a
shared script list, protected by a lock.
"""

from threading import Event, Thread, Semaphore, Lock

class Device(object):
    """
    Represents a device in the simulation with a pool of worker threads.

    This class manages a list of scripts for a timepoint, which are
    dynamically consumed by its worker threads. Device 0 acts as the
    coordinator for initializing shared locks and barriers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_scripts = []
        self.neighbours = []
        self.timepoint_done = Event()
        
        self.initialization = Event()
        
        self.threads = []
        # Creates a pool of 8 worker threads.
        for k in xrange(8):
            self.threads.append(DeviceThread(self, k))
        self.locations_lock = Lock()
        self.locked_locations = None # Shared dictionary of location-specific locks
        self.devices_barrier = None # Global barrier for all threads
        self.device_barrier = ReusableBarrier(len(self.threads)) # Internal device barrier

    def __str__(self):
        """String representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources via the coordinator device (ID 0).
        """
        
        if self.device_id == 0:
            
            # Coordinator creates the shared location locks and global barrier.
            self.locked_locations = {}

            
            self.devices_barrier = ReusableBarrier(len(devices)*len(self.threads))

            
            # Propagate shared objects to other devices and signal initialization.
            for device in devices:
                device.locked_locations = self.locked_locations
                device.devices_barrier = self.devices_barrier
                device.initialization.set()

        else:
            
            # Non-coordinator devices wait for initialization to complete.
            self.initialization.wait()

        # Start all worker threads in the pool.
        for thread in self.threads:
            thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A None script signals the end of a timepoint.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's worker threads."""
        
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread that dynamically pulls scripts from a shared list.
    """

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            
            # All threads in the simulation synchronize globally.
            self.device.devices_barrier.wait()

            
            # Thread 0 fetches neighbor data for the device.
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            
            # Threads in this device synchronize after neighbor data is fetched.
            self.device.device_barrier.wait()
            neighbours = self.device.neighbours
            # Supervisor signals shutdown
            if neighbours is None:
                break

            
            # Wait for all scripts for the timepoint to be assigned.
            self.device.timepoint_done.wait()
            # Thread 0 prepares the shared list of scripts for this timepoint.
            if self.thread_id == 0:
                self.device.timepoint_scripts = self.device.scripts[:]
            self.device.device_barrier.wait()
            # Work-stealing loop: threads pull scripts from the shared list.
            while True:
                
                self.device.locations_lock.acquire()
                if len(self.device.timepoint_scripts) == 0:
                    self.device.locations_lock.release()
                    break
                (script, location) = self.device.timepoint_scripts.pop()

                


                # Dynamically create location lock if it doesn't exist.
                if location not in self.device.locked_locations:
                    self.device.locked_locations[location] = Lock()

                self.device.locked_locations[location].acquire()
                self.device.locations_lock.release()

                script_data = []
                
                # Gather data from neighbors and self.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    # Run script and propagate results.
                    result = script.run(script_data)

                    
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                
                self.device.locked_locations[location].release()

            
            # Synchronize within the device before clearing timepoint data.
            self.device.device_barrier.wait()
            self.device.timepoint_done.clear()

class ReusableBarrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads.
    
    Uses a two-phase semaphore implementation.
    """
    
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for _ in range(self.num_threads):
                    
                    threads_sem.release()
                count_threads[0] = self.num_threads        
        
        threads_sem.acquire()                    
