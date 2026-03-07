"""
This Python module implements a distributed device simulation with several
critical design flaws.

The framework uses a master-worker pattern where each device spawns worker
threads (`MyThread`) to execute scripts. It attempts to use fine-grained locking
for data locations and a reusable barrier for synchronization.

However, the implementation suffers from:
- Flawed lock distribution during setup, leading to incorrect state.
- A likely deadlock in the main `DeviceThread` loop, where it waits on a global
  barrier before its own worker threads have finished their tasks. The barrier
  wait should occur after the threads are joined.
"""

from threading import Event, Semaphore, Lock, Thread

class ReusableBarrierSem(object):
    """A reusable barrier implementation using a two-phase semaphore protocol."""
    

    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads


        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """Causes the calling thread to wait until all threads reach the barrier."""
        
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the barrier synchronization."""
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):


                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase to ensure all threads have passed phase 1 before reuse."""
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()


class MyThread(Thread):
    """A worker thread to execute a single script on a specific data location."""
    

    def __init__(self, device, location, neighbours, script):
        """Initializes the worker thread."""
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script

    def run(self):
        """
        Acquires a location-specific lock, gathers data, runs the script,
        updates data, and releases the lock.
        """
        
        self.device.locks[self.location].acquire()

        script_data = []
        
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        self.device.locks[self.location].release()

def get_locations(devices):
    """
    Calculates the number of unique data locations across all devices.

    This helper function iterates through all devices to find the maximum
    location ID to size the list of locks.
    """
    
    no_loc = 0

    for i in xrange(len(devices)):
        maxx = int(max(devices[i].sensor_data.keys()))
        if maxx > no_loc:
            no_loc = maxx
    return no_loc + 1 

class Device(object):
    """Represents a single device in the simulation."""
    

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device."""
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        
        self.locks = []
        
        self.barrier = None
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up and distributes shared synchronization primitives.

        NOTE: The logic for distributing locks is flawed. It repeatedly appends
        the same locks to each device's lock list instead of sharing a single list.
        """
        
        
        

        if self.device_id == 0:
            
            barrier = ReusableBarrierSem(len(devices))
            for i in xrange(len(devices)):
                devices[i].barrier = barrier

            
            no_loc = get_locations(devices)
            for i in xrange(no_loc):
                lock = Lock()              
                self.locks.append(lock)    

            
            # FLAWED LOGIC: This creates a much larger and incorrect lock list on each device.
            for i in xrange(no_loc):
                for j in xrange(len(devices)):
                    devices[j].locks.append(self.locks[i])

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device."""
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's thread."""
        
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device.

    NOTE: The synchronization logic in the run loop is critically flawed and
    will likely cause deadlocks.
    """
    

    def __init__(self, device):
        """Initializes the device thread."""
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            my_threads = []

            self.device.script_received.wait()
            self.device.script_received.clear()

            
            # Start worker threads for each script.
            for (script, location) in self.device.scripts:
                
                my_threads.append(MyThread(self.device, location, neighbours, script))
                my_threads[-1].start()

            
            # CRITICAL FLAW: The thread waits at the barrier BEFORE its worker
            # threads have completed. This will cause a deadlock, as workers
            # may need to access resources locked by other devices that are
            # also waiting at the barrier. The barrier.wait() call should
            # occur AFTER the join() loop.
            self.device.barrier.wait()

            for i in xrange(len(my_threads)):
                my_threads[i].join()

            
            
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
