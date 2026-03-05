"""
This module implements a device simulation using a two-phase barrier
synchronization and a mixture of correct and incorrect locking strategies.

This version uses a "leader" device (device 0) to initialize shared
resources, including a dictionary of re-entrant locks (`RLock`) for each sensor
location. The main `DeviceThread` correctly uses a two-barrier approach to
synchronize the main work phase of the simulation.

NOTE: This implementation contains multiple severe concurrency bugs:
1. The setup logic in `setup_devices` is overly complex and fragile, using
   semaphores in a confusing way to propagate shared state.
2. The data access locking is fundamentally broken. While there is a correct
   shared lock per *location*, the `get_data` and `set_data` methods are
   protected by a *per-device*, unshared lock, which provides no protection
   against race conditions between devices. This makes all data access unsafe.
"""

from threading import Event, Thread, Semaphore, Lock, RLock
# Assumes the presence of a 'reusable_barrier.py' file.
from reusable_barrier import ReusableBarrier
import multiprocessing

class Device(object):
    """
    Represents a device in the simulation.
    
    This class attempts to manage shared state (locks, barrier) initialized
    by a leader device (device_id 0).
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.results = {}
        # BUG: This lock is unshared. Each device gets its own instance.
        # It provides no mutual exclusion between different devices.
        self.lock = None 
        # This dictionary of locks IS intended to be shared.
        self.dislocksdict = None 
        self.barrier = None # The main shared barrier.
        self.sem = Semaphore(1) # Unused semaphore.
        self.sem2 = Semaphore(0) # Used in the fragile setup logic.
        self.all_devices = []
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Unused event.
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources.
        
        BUG: The semaphore-based propagation logic is convoluted and fragile. A
        simple loop in the main thread would be clearer and more robust.
        """
        loc = [l for d in devices for l in d.sensor_data]
        
        # Device 0 acts as the leader to create shared resources.
        if self.device_id == 0:
            self.sem2.release()
            self.barrier = ReusableBarrier(len(devices))
            self.dislocksdict = {k: RLock() for k in set(loc)}
            self.lock = Lock() # Leader creates its own unshared lock.

        # This mechanism attempts to make setup sequential.
        self.sem2.acquire()

        # Propagate shared resources to other devices.
        for d in devices:
            if d.barrier is None:
                d.barrier = self.barrier 
                d.sem2.release() # Signal the next device in line.
                d.dislocksdict = self.dislocksdict
                d.lock = Lock() # Each device gets a new, unshared lock.

    def assign_script(self, script, location):
        """Assigns a script to be executed in the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()
   
    def get_data(self, location):
        """Retrieves sensor data. This method itself is not thread-safe."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data. This method itself is not thread-safe."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

class RunScript(Thread):
    """A worker thread to execute a single script."""
    def __init__(self, script, location, neighbours, device):
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        # Acquire the shared lock for this specific location. This correctly
        # prevents other scripts for the same location from running concurrently.
        self.device.dislocksdict[self.location].acquire()
        try:
            script_data = []
            # --- BUG: Unsafe Data Access ---
            # The code acquires `device.lock`, but this is the device's *own*
            # private lock, offering no protection from other threads accessing
            # its data. The `get_data` call is racy.
            for device in self.neighbours:  
                with device.lock: # This lock is not shared, so this is ineffective.
                    data = device.get_data(self.location) 
                if data is not None:
                    script_data.append(data)
                    
            with self.device.lock: # Ineffective lock.
                data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data:
                result = self.script.run(script_data) 
                
                # Propagate results with the same flawed locking.
                for device in self.neighbours:
                    with device.lock:
                        device.set_data(self.location, result)
                with self.device.lock:
                    self.device.set_data(self.location, result)
        finally:
            # Release the shared location lock.
            self.device.dislocksdict[self.location].release()


class DeviceThread(Thread):
    """The main control thread for a device."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main simulation loop using a two-barrier synchronization pattern."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal.
            
            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait() 

            # --- Synchronization Point 1 ---
            # All devices wait here before starting their work for the time step.
            self.device.barrier.wait() 

            # --- Work Phase ---
            # Inefficiently create, start, and join new threads for every script.
            script_threads = []
            for (script, location) in self.device.scripts:
                script_threads.append(RunScript(script, location, neighbours, self.device))
            for t in script_threads:
                t.start() 
            for t in script_threads:
                t.join() 
            
            # --- Synchronization Point 2 ---
            # All devices wait here after finishing their work, before the next step.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
            self.device.scripts = [] # Clear scripts for the next round.
