"""
This module implements a device simulation framework for Python 2.

This version features a robust, if inefficient, custom lock manager (`LocationsLoc`)
for location-based resource locking, and uses a static partitioning scheme to
distribute work among a pool of threads within each device.

Classes:
    ReusableBarrierCond: A condition-based reusable barrier for synchronization.
    LocationsLoc: A custom manager for location-specific locks.
    Device: Represents a device, managing its threads and shared resources.
    DeviceThread: A worker thread that executes a partition of the device's scripts.
"""
from threading import Event, Thread, Condition, Lock

class ReusableBarrierCond(object):
    """A reusable barrier implemented with a Condition variable."""
    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition() 
                                
    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.cond.acquire()     
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all() 
            self.count_threads = self.num_threads
        else:
            self.cond.wait()   
        self.cond.release()    


class LocationsLoc(object):
    """
    A custom lock manager for handling location-based locks.

    It maintains a list of locations and a parallel list of Lock objects.

    Note: The use of list.index() for lookups can be inefficient for a
    large number of unique locations.
    """
    def __init__(self):
        """Initializes the lock manager."""
        self.locations_vect = []
        self.lock_vect = []

    def add_loc(self, location):
        """Adds a new location and a corresponding lock if not already present."""
        if location not in self.locations_vect:
            self.locations_vect.append(location)
            self.lock_vect.append(Lock())
            return True
        else:
            return False

    def acquire_loc(self, location):
        """Acquires the lock associated with a given location."""
        ind = self.locations_vect.index(location)
        self.lock_vect[ind].acquire()

    def release_loc(self, location):
        """Releases the lock associated with a given location."""
        ind = self.locations_vect.index(location)
        self.lock_vect[ind].release()


class Device(object):
    """
    Represents a single device in the simulation.

    Each device manages a pool of worker threads and coordinates with other
    devices using shared barrier and lock objects.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.setup_ready = Event()
        self.barrier = None
        self.threads_lista = []
        self.locs_loc = None
        self.locs_loc_lock = None
        self.neighbours = []
        self.take_neigh = 1
        self.setup_dev = Event()
        self.devices = []
        for i in xrange(8):
            self.threads_lista.append(DeviceThread(self, i, 8))
        for i in xrange(8):
            self.threads_lista[i].start()


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources like the barrier and lock manager.

        Device 0 acts as the coordinator.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(8 * len(devices))
            self.locs_loc = LocationsLoc()
            self.locs_loc_lock = Lock()
            for dev in devices:
                dev.barrier = self.barrier
                dev.locs_loc = self.locs_loc
                dev.locs_loc_lock = self.locs_loc_lock
                dev.setup_dev.set()

            for dev in devices:
                dev.setup_ready.set()
        
        self.setup_dev.wait()
        self.devices = devices


    def assign_script(self, script, location):
        """
        Assigns a script to the device and ensures a lock exists for its location.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()

            self.locs_loc_lock.acquire()
            if self.locs_loc.add_loc(location) == True:
                for dev in xrange(len(self.devices)):
                    self.devices[dev].locs_loc = self.locs_loc

            self.locs_loc_lock.release()

        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its threads."""
        for i in xrange(8):
            self.threads_lista[i].join()



class DeviceThread(Thread):
    """
    A worker thread that executes a statically assigned partition of scripts.
    """
    def __init__(self, device, i=-1, nr=0):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.online = 0
        self.thread_id = i
        self.number = nr

    def run(self):
        """The main execution loop for the worker thread."""
        self.device.setup_ready.wait()

        if self.thread_id == -1:
            print "ERROR: Wrong thread_id = -1"
            return -1
        else:
            online = 1



        while True:
            
            self.device.locs_loc_lock.acquire()
            if self.device.take_neigh == 1:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.take_neigh = 0
            self.device.locs_loc_lock.release()

            if self.device.neighbours is None:
                break

            self.device.timepoint_done.wait()

            
            for (script, location) in self.device.scripts[self.thread_id:len(self.device.scripts):self.number]:
                script_data = []
                self.device.locs_loc.acquire_loc(location)
                
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)



                if script_data != []:
                    
                    result = script.run(script_data)

                    

                    for device in self.device.neighbours:
                        if device.get_data(location) is not None:
                            device.set_data(location, result)
                    
                    if self.device.get_data(location) is not None:
                        self.device.set_data(location, result)

                self.device.locs_loc.release_loc(location)
            
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
            self.device.take_neigh = 1
            self.device.barrier.wait()
        
        if online == 1:
            return 0
        else:
            return -1
            
