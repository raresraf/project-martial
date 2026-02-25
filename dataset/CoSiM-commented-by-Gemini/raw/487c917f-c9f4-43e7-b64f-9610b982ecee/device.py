"""
Models a device in a distributed simulation using static work partitioning
among a fixed pool of worker threads.

This module simulates a network of devices where each device manages its own
pool of worker threads. Work (scripts) is distributed statically among the
threads based on their ID. A global, two-phase barrier synchronizes all
worker threads from all devices at the end of each time step.

Classes:
    ReusableBarrier: A standard, condition-based reusable thread barrier.
    Device: Represents a device, its data, and its pool of worker threads.
    DeviceThread: The worker thread for a device. There is no master thread;
                  all logic is contained within the workers.
"""

from threading import Lock, Event, Thread, Condition

class ReusableBarrier():
    """
    A standard implementation of a reusable barrier for thread synchronization.
    
    This barrier uses a `Condition` variable to block threads until the required
    number of threads have reached the barrier.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 

    def wait(self):
        """Blocks the calling thread until all participating threads have also called wait."""
        self.cond.acquire()                      
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()               
            self.count_threads = self.num_threads    
        else:
            self.cond.wait()                    
        self.cond.release()                     


class Device(object):
    """
    Represents a device and manages a pool of worker threads to execute scripts.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.gotneighbours = Event()
        self.zavor = Lock() # A lock to coordinate fetching neighbors.
        self.threads = []
        self.neighbours = []
        self.nthreads = 8
        self.barrier = ReusableBarrier(1) # Initially a dummy barrier.
        self.lockforlocation = {}
        self.num_locations = supervisor.supervisor.testcase.num_locations
        
        # Create and start a fixed pool of worker threads.
        for i in xrange(self.nthreads):
            self.threads.append(DeviceThread(self, i))


        for i in xrange(self.nthreads):
            self.threads[i].start()


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs a centralized setup for shared resources across all devices.
        
        This method creates a single barrier for all worker threads in the entire
        simulation and a set of shared locks for each data location.
        """
        barrier = ReusableBarrier(devices[0].nthreads*len(devices))
        lockforlocation = {}
        for i in xrange(0, devices[0].num_locations):
            lock = Lock()
            lockforlocation[i] = lock
        for i in xrange(0, len(devices)):
            devices[i].barrier = barrier
            devices[i].lockforlocation = lockforlocation


    def assign_script(self, script, location):
        """
        Assigns a script to the device. A None script signals the end of assignments.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates the sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its worker threads."""
        for i in xrange(self.nthreads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A worker thread that executes a statically partitioned subset of scripts.
    """
    def __init__(self, device, id_thread):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id_thread = id_thread

    def run(self):
        """The main simulation loop for the worker thread."""
        while True:
            
            # Block to ensure only one thread per device fetches neighbors.
            self.device.zavor.acquire()
            if self.device.gotneighbours.is_set() == False:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.gotneighbours.set()
            self.device.zavor.release()
            
            # If supervisor signals shutdown, break the loop.
            if self.device.neighbours is None:
                break

            # Wait for the device to receive all scripts for the timepoint.
            self.device.timepoint_done.wait()
            
            # --- Static Work Partitioning ---
            # Each worker thread gets a unique, non-overlapping subset of the scripts.
            # NOTE: The stride `self.device.nthreads + 1` appears to be a bug;
            # it should likely be `self.device.nthreads` to cover all scripts.
            myscripts = []
            for i in xrange(self.id_thread, len(self.device.scripts), self.device.nthreads + 1):
                myscripts.append(self.device.scripts[i])

            # Process the assigned partition of scripts.
            for (script, location) in myscripts:
                self.device.lockforlocation[location].acquire()
                script_data = []
                
                # Gather data from neighbors.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from the parent device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execute script and propagate results.
                if script_data != []:
                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
                self.device.lockforlocation[location].release()

            # --- Two-Phase Barrier Synchronization ---
            # 1. All worker threads (from all devices) wait here after finishing their work.
            self.device.barrier.wait()
            
            # 2. One thread per device resets the events for the next timepoint.
            if self.id_thread == 0:
                self.device.timepoint_done.clear()
                self.device.gotneighbours.clear()
            
            # 3. All threads wait again to ensure no thread starts the next timepoint
            #    before all others have finished the cleanup phase.
            self.device.barrier.wait()