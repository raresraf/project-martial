"""
This module provides a simulation framework for a network of distributed devices.

It features a robust two-phase semaphore-based reusable barrier for synchronization
and a dynamic, on-demand locking mechanism to ensure data consistency for
specific data locations. Each device operates in its own thread, processing
assigned scripts in synchronized time-steps.
"""


from threading import Event, Thread, RLock, Lock, Semaphore


class ReusableBarrier():
    """
    A reusable, two-phase thread barrier implemented with Semaphores.

    This barrier is designed to prevent race conditions where faster threads might
    lap slower ones and re-enter the barrier before the first wave of threads
    has fully exited. It uses a generalized 'phase' method for its two stages.
    """

    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        # Counters are wrapped in lists to be passed by reference.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)
 
    def wait(self):
        """Blocks the calling thread until all threads reach the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """A single phase of the two-phase barrier."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive opens the semaphore gate for all others.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # All threads wait here until the gate is opened by the last thread.
        threads_sem.acquire()
     
class Device(object):
    """
    Represents a single device in the simulation.
    
    It manages its own sensor data and a control thread that orchestrates the
    execution of scripts and synchronization with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Event to signal that all scripts for a time-step have been assigned.
        self.last_script = Event()
        self.thread = DeviceThread(self)
        # The shared barrier for synchronizing time-steps.
        self.timepoint_done = None
        # The shared dictionary for location-specific locks.
        self.loc_lock = None
        self.thread.start()
        
    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared synchronization objects."""
        # This non-leader-based setup could be racy if called concurrently.
        if self.timepoint_done is None:
            barrier = ReusableBarrier(len(devices))
            dic = {} # The shared dictionary for location-specific locks.
            for dev in devices:
                dev.timepoint_done = barrier
                dev.loc_lock = dic

    def assign_script(self, script, location):
        """Assigns a script to the device or signals the end of assignments."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.last_script.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
    
    def run_script(self, lock, neighbours, location, script):
        """
        Target function for worker threads that executes a single script.

        Implements a two-tier, on-demand locking scheme to protect shared data.
        """
        # Phase 1: Acquire a global lock to safely check/create the location-specific lock.
        lock.acquire();
        if not (self.device.loc_lock).has_key(location):
            self.device.loc_lock[location] = Lock()
        lock.release()
        
        # Phase 2: Acquire the location-specific lock to protect data access.
        self.device.loc_lock.get(location).acquire()
        
        script_data = []
        
        # Gather data from neighbors and self.
        for dev in neighbours:
            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)
               
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
                
        if script_data != []:
            # Run script and update data for all relevant devices.
            result = script.run(script_data)
            for dev in neighbours:
                dev.set_data(location, result)
            self.device.set_data(location, result)
        
        # Release the location-specific lock.
        (self.device.loc_lock.get(location)).release()
                

    def run(self):
        """
        The main operational loop for the device thread.
        
        It waits for scripts, spawns worker threads to execute them under a
        locking scheme, and then synchronizes with all other devices.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for the signal that all scripts for this time-step have been assigned.
            self.device.last_script.wait()
            
            # NOTE: This lock is re-created in every loop iteration, which is inefficient.
            # It should be created once outside the loop.
            lock = RLock()
            threads = []

            # Spawn a new worker thread for each assigned script.
            for (script, location) in self.device.scripts:
               	thread = Thread(target = self.run_script, args = (lock, neighbours, location, script))
                thread.start()
                threads.append(thread)
            
            # Wait for all worker threads for this device to complete.
            for thread in threads:
                thread.join()
 
            # Wait at the global barrier to synchronize the end of the time-step
            # with all other devices in the simulation.
            self.device.timepoint_done.wait()
            self.device.last_script.clear()