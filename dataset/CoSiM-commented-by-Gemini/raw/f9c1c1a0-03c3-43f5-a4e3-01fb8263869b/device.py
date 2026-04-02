"""
This module implements a seventh variant of a distributed device simulation.

This architecture uses a master device (ID 0) to create and distribute shared
synchronization objects, including a global ReusableBarrier and a dictionary of
re-entrant locks for each data location. The simulation proceeds in
time-steps, synchronized by a barrier-work-barrier pattern. The work itself is
performed using an inefficient, unbounded "spawn-and-join" model, where a new
thread is created for each task in a time-step.

A significant design flaw exists in the locking strategy, where a worker thread
holds a global location lock while acquiring individual device locks, creating a
high risk of deadlocks.
"""

from threading import Event, Thread, Semaphore, Lock, RLock
from reusable_barrier import ReusableBarrier
import multiprocessing

class Device(object):
    """
    Represents a device node in the simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and its main control thread."""
        self.results = {}
        self.lock = None 
        # Dictionary for distributed, location-based locks.
        self.dislocksdict = None 
        self.barrier = None
        self.sem = Semaphore(1)
        # Semaphore used for orchestrating the initial setup of shared resources.
        self.sem2 = Semaphore(0)
        self.all_devices = []
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources using a master-device pattern.

        Device 0 creates the global barrier and location locks. A semaphore chain
        is used to sequence the distribution of these shared objects to other devices,
        which is a complex and potentially fragile synchronization pattern.
        """
        loc = []
        for d in devices:
            for l in d.sensor_data:
                loc.append(l) 
        all_devices = devices
        # Device 0 acts as the master for initialization.
        if self.device_id == 0:
            self.sem2.release()
            self.barrier = ReusableBarrier(len(devices))
            self.dislocksdict = {}
            for k in list(set(loc)):
                self.dislocksdict[k] = RLock()
            self.lock = Lock()

        # This semaphore acquire/release chain serializes the setup process.
        self.sem2.acquire()

        # Each device gets a reference to the globally created objects.
        for d in devices:
            if d.barrier == None:
                d.barrier = self.barrier 
                d.sem2.release() 
                d.dislocksdict = self.dislocksdict
                d.lock = Lock()

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
   
    def get_data(self, location):
        """Retrieves data from the device's sensor data."""
        data = -1
        if location in self.sensor_data:
            data = self.sensor_data[location]
            return data
        else:
            return None

    def set_data(self, location, data):
        """Sets data in the device's sensor data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class RunScript(Thread):
    """A worker thread that executes a single script."""
    def __init__(self, script, location, neighbours, device):
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device
    def run(self):
        """
        Executes the script.

        WARNING: This method implements a dangerous double-locking pattern that is
        highly prone to deadlocks. It acquires a global location lock and then,
        while holding it, acquires individual locks for each device it needs to
        access. If two threads acquire location locks in one order and then
        device locks in the opposite order, a deadlock will occur.
        """
        # Acquire a global lock for the specific data location.
        self.device.dislocksdict[self.location].acquire()
        script_data = []
        for device in self.neighbours:
            # Acquire a lock for the specific device being accessed.
            device.lock.acquire()
            data = device.get_data(self.location) 
            device.lock.release()

            if data is not None:
                script_data.append(data)
                
        self.device.lock.acquire()
        data = self.device.get_data(self.location)
        self.device.lock.release()
        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = self.script.run(script_data) 
            
            # Disseminate results, again using per-device locks.
            for device in self.neighbours:
                device.lock.acquire()
                device.set_data(self.location, result)
                device.lock.release()
            self.device.lock.acquire()
            self.device.set_data(self.location, result)
            self.device.lock.release()
        
        # Release the global location lock.
        self.device.dislocksdict[self.location].release()


class DeviceThread(Thread):
    """The main control thread for a device."""

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop, structured as a barrier-work-barrier pattern.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.
            
            self.device.timepoint_done.wait() 
            
            # First barrier: All devices synchronize before starting work.
            self.device.barrier.wait() 
            
            # Create, start, and join a new thread for each script. (Spawn-and-join)
            script_threads = []
            for (script, location) in self.device.scripts:
                script_threads.append(RunScript(script, location, neighbours, self.device))
            for t in script_threads:
                t.start() 
            for t in script_threads:
                t.join() 
            
            # Second barrier: All devices synchronize after finishing work.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
