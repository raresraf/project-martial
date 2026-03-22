"""
This module implements a distributed device simulation with per-device thread pools.

Key architectural features:
- A "master" device (device 0) creates and distributes two global barriers and a
  global list of locks.
- Each device has its own pool of worker threads (`ParallelScript`).
- The main control threads (`DeviceThread`) use the global barriers to synchronize
  the start and end of time steps.
- The supervisor, which calls `assign_script`, also participates in one of the
  barriers, creating a complex synchronization scheme.
- A critical race condition exists in the implementation of the task queue
  (`to_procces`), which is a standard list modified without a lock.

Note: This script depends on a local `barrier.py` and uses Python 2 syntax.
"""

from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a device node with its own pool of worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)
        self.thread.start()
        
        # --- Globally Shared Objects ---
        self.time_bar = None
        self.script_bar = None
        self.devloc = [] # A list of location-specific locks.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared resources using a master pattern."""
        if self.device_id == 0:
            # Device 0 creates and distributes two separate global barriers.
            self.time_bar = ReusableBarrierSem(len(devices))
            self.script_bar = ReusableBarrierSem(len(devices))
            for device in devices:
                device.time_bar = self.time_bar
                device.script_bar = self.script_bar

            # Create a global list of locks, one for each location.
            maxim = 0
            for device in devices:
                if device.sensor_data:
                    loc_list = sorted(device.sensor_data.keys())
                    if loc_list[-1] > maxim:
                        maxim = loc_list[-1]
            self.devloc = [Lock() for _ in range(maxim + 1)]

            for device in devices:
                device.devloc = self.devloc

    def assign_script(self, script, location):
        """
        Assigns a script. When all scripts are assigned (script is None), the
        calling (supervisor) thread blocks on a barrier.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()
            # The supervisor now waits until all DeviceThreads have posted their work.
            self.script_bar.wait()

    def get_data(self, location):
        """Non-thread-safe method to get data."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Non-thread-safe method to set data."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class ParallelScript(Thread):
    """
    A worker thread that executes a single script task.
    """
    def __init__(self, device_thread):
        Thread.__init__(self)
        self.device_thread = device_thread

    def run(self):
        while True:
            self.device_thread.sem_scripts.acquire() # Wait for a task.

            # CRITICAL FLAW: This is a race condition. Two threads could acquire
            # the semaphore, and both could enter this block before either has
            # deleted the item, causing them to process the same task twice
            # and leading to an index error on the `del` for the second thread.
            # A lock is needed to make this pop operation atomic.
            nod = self.device_thread.to_procces.pop(0)
            
            if nod is None: # Shutdown signal
                break
            
            neighbours, script, location = nod

            # Acquire the specific lock for the location being processed.
            with self.device_thread.device.devloc[location]:
                script_data = []
                # Gather data from neighbors.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                # Gather data from the local device.
                data = self.device_thread.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Execute script and broadcast results.
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device_thread.device.set_data(location, result)


class DeviceThread(Thread):
    """
    The main control thread for a device, managing a pool of worker threads.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.sem_scripts = Semaphore(0) # Semaphore to signal work to the pool.
        self.numar_procesoare = 8 # "numar_procesoare" is Romanian for "number of processors"
        self.to_procces = [] # Unsafe list used as a task queue.
        self.pool = self.create_pool(self)

    def create_pool(self, device_thread):
        """Creates and starts the pool of worker threads for this device."""
        pool = []
        for _ in xrange(self.numar_procesoare):
            aux_t = ParallelScript(device_thread)
            pool.append(aux_t)
            aux_t.start()
        return pool

    def run(self):
        """The main execution loop, synchronized with other DeviceThreads."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # --- Shutdown sequence ---
                for _ in range(self.numar_procesoare):
                    self.to_procces.append(None) # Post sentinel values.
                    self.sem_scripts.release() # Wake up workers to exit.
                for item in self.pool:
                    item.join()
                break
            
            # Wait for supervisor to assign all scripts.
            self.device.script_received.wait()
            
            # Post all tasks to the unsafe queue and signal the workers.
            for (script, location) in self.device.scripts:
                nod = (neighbours, script, location)
                self.to_procces.append(nod)
                self.sem_scripts.release()

            # BARRIER 1: Sync with other devices after all work has been queued.
            self.device.script_bar.wait()

            # BARRIER 2: Sync again to mark the end of the time step. The control
            # thread does not wait for its workers to finish, which is a confusing
            # design, relying on the barriers to enforce the overall timing.
            self.device.time_bar.wait()
            
            self.device.script_received.clear()
