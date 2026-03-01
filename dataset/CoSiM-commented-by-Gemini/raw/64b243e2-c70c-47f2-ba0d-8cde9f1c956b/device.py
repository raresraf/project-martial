"""
@file device.py
@brief A distributed device simulation with a dynamic, per-task threading model
       and a complex multi-level synchronization scheme.

This script models a network of devices. Its architecture is notable for being
highly inefficient: for each time step, a master thread on each device spawns a
new, separate thread for every single script it needs to run.

Synchronization is particularly complex, involving:
1. A two-stage global barrier synchronization per time step.
2. A semaphore-driven, sequential setup routine.
3. A two-tiered locking system for data access (a global location lock plus a
   per-device data lock).
"""

from threading import Event, Thread, Semaphore, Lock, RLock
from reusable_barrier import ReusableBarrier
import multiprocessing

class Device(object):
    """
    Represents a node in the network, managing its data and a large collection of
    synchronization primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.results = {}
        # A per-device lock, likely for its own data structure.
        self.lock = None 
        # A dictionary of global, re-entrant locks for each location.
        self.dislocksdict = None 
        # A shared barrier for all devices.
        self.barrier = None
        # Semaphores used for a sequential setup process.
        self.sem = Semaphore(1)
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
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (locks and barrier).
        This routine is complex, using semaphores to enforce a sequential
        setup across devices.
        """
        loc = []
        for d in devices:
            for l in d.sensor_data:
                loc.append(l) 
        all_devices = devices
        # Device 0 acts as the coordinator.
        if self.device_id == 0:
            self.sem2.release() # Start the chain.
            self.barrier = ReusableBarrier(len(devices))
            self.dislocksdict = {}
            for k in list(set(loc)):
                self.dislocksdict[k] = RLock()
            self.lock = Lock()

        self.sem2.acquire() # Each device waits its turn.

        # Pass the shared resources down the chain.
        for d in devices:
            if d.barrier == None:
                d.barrier = self.barrier 
                d.sem2.release() # Signal the next device.
                d.dislocksdict = self.dislocksdict
                d.lock = Lock()

    def assign_script(self, script, location):
        """Assigns a script to run in the next timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
   
    def get_data(self, location):
        """Retrieves sensor data. Locking is handled externally."""
        data = -1
        if location in self.sensor_data:
            data = self.sensor_data[location]
            return data
        else:
            return None

    def set_data(self, location, data):
        """Sets sensor data. Locking is handled externally."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

class RunScript(Thread):
    """A short-lived thread created to execute exactly one script."""
    def __init__(self, script, location, neighbours, device):
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """Executes the script using a two-tiered locking scheme."""
        # Tier 1 Lock: Acquire the global lock for this location.
        self.device.dislocksdict[self.location].acquire()
        
        script_data = []
        # Aggregate data from neighbors.
        for device in self.neighbours:
            # Tier 2 Lock: Acquire the per-device lock. This is likely redundant.
            device.lock.acquire()
            data = device.get_data(self.location) 
            device.lock.release()
            if data is not None:
                script_data.append(data)
        
        # Aggregate data from self.
        self.device.lock.acquire()
        data = self.device.get_data(self.location)
        self.device.lock.release()
        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = self.script.run(script_data)
            
            # Disseminate results to neighbors and self, using per-device locks.
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
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main time-stepping loop with a two-stage barrier synchronization."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown
            
            # Wait for supervisor to finish assigning scripts.
            self.device.timepoint_done.wait() 

            # ----- TIMEPOINT START -----
            # Stage 1: All devices synchronize before starting work.
            self.device.barrier.wait() 

            # Inefficiently create and start a new thread for every single script.
            script_threads = []
            for (script, location) in self.device.scripts:
                script_threads.append(RunScript(script, location, neighbours, self.device))
            for t in script_threads:
                t.start() 
            # Wait for all script threads for this timepoint to complete.
            for t in script_threads:
                t.join() 

            # Stage 2: All devices synchronize after finishing work.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
