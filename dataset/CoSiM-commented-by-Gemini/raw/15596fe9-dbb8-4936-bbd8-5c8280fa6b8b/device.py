"""
This module implements a device simulation framework for Python 2, featuring
a hierarchical master-worker threading model within each device.

Each 'Device' is managed by a 'DeviceThread' (acting as a master), which in
turn coordinates a pool of 'Worker' threads to execute tasks. This design
separates high-level timepoint orchestration from the parallel execution of
individual scripts.

Classes:
    ReusableBarrier: A two-phase, semaphore-based reusable barrier.
    Device: The main device class, containing its data and thread hierarchy.
    Worker: A worker thread that executes a partition of the device's scripts.
    DeviceThread: A master thread that manages a pool of Worker threads for a single device.
"""
from threading import Event, Thread, Lock, RLock, Semaphore
import multiprocessing


class ReusableBarrier(object):
    """A reusable, two-phase barrier implemented with semaphores."""
    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        # Counters are wrapped in lists to act as mutable integer references.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        
        self.count_lock = Lock()
        
        self.threads_sem1 = Semaphore(0)
        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to block until all threads reach the barrier."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            
            if count_threads[0] == 0:
                
                for i in range(self.num_threads):
                    threads_sem.release()
                    
                count_threads[0] = self.num_threads
                

        
        threads_sem.acquire()



class Device(object):
    """
    Represents a single device in the simulation network.

    Each device encapsulates its own master thread (`DeviceThread`), which
    manages a pool of worker threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

         
        self.barrier = None

        
        
        self.location_lock = {}

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier for inter-device synchronization.

        Note: This implementation has a fragile dependency on the device order,
        as it assumes `devices[0]` is the coordinator.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
        else:
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """
        Assigns a script to the device.

        A `None` script signals that all scripts for the timepoint are assigned.
        """
        if script is not None and location is not None:
            
            
            if location not in self.location_lock.keys():
                self.location_lock[location] = RLock()

            self.scripts.append((script, location))
        else:
            
            self.script_received.set()

    def get_data(self, location):
        """
        Acquires a reentrant lock and gets data for a location.

        Warning: This method acquires a lock that must be released by a
        corresponding call to `set_data`, creating a fragile API.
        """
        if location in self.location_lock.keys():
            self.location_lock[location].acquire()
        else:
            
            self.location_lock[location] = RLock()
            self.location_lock[location].acquire()

        return self.sensor_data[location] if location in self.sensor_data \
                                                    else None

    def set_data(self, location, data):
        """Sets data for a location and releases the corresponding lock."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

        
        
        if location in self.location_lock.keys():
            self.location_lock[location].release()

    def shutdown(self):
        """Shuts down the device by joining its master thread."""
        self.thread.join()


class Worker(Thread):
    """
    A worker thread that executes a static partition of scripts for a device.
    """
    def __init__(self, device, id, master):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id = id
        self.master = master
        self.neighbours = []

    def run(self):
        """The main execution loop for the worker thread."""
        lg = len(self.device.scripts) / self.master.no_threads + 1

        while True:
            
            self.master.update_neigh.wait()

            
            self.neighbours = self.master.neighbours

            
            if self.neighbours is None:
                break

            self.master.w_barrier.wait()

            
            self.master.start_worker.wait()

            
            for index in xrange(self.id * lg, (self.id + 1) * lg):
                
                
                if index < len(self.device.scripts):
                    script = self.device.scripts[index][0]


                    location = self.device.scripts[index][1]

                    script_data = []
                    
                    for device in self.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        
                        result = script.run(script_data)

                        for device in self.neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)

            
            
            self.master.w_barrier.wait()

            if self.id == 0:
                
                self.master.start_worker.clear()
                


                self.master.update_neigh.clear()
                
                self.master.end_worker.set()

            
            self.master.w_barrier.wait()


class DeviceThread(Thread):
    """
    A master thread that manages a pool of Worker threads for a single device.
    """
    def __init__(self, device):
        """Initializes the master thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.no_threads = max(8, multiprocessing.cpu_count())
        self.w_barrier = ReusableBarrier(self.no_threads)
        self.neighbours = []
        
        self.start_worker = Event()
        
        self.end_worker = Event()
        
        self.update_neigh = Event()

    def run(self):
        """The main loop for the master thread."""
        threads = []

        for i in xrange(0, self.no_threads):
            threads.append((Worker(self.device, i, self)))

        for thread in threads:
            thread.start()

        while True:
            
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                self.update_neigh.set()
                break



            self.update_neigh.set()

            
            self.device.script_received.wait()

            
            self.start_worker.set()

            
            self.end_worker.wait()

            
            
            self.device.barrier.wait()

            
            
            self.device.script_received.clear()
            self.end_worker.clear()

        
        for thread in threads:
            thread.join()
