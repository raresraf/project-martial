"""
This module implements a device simulation using a persistent worker thread pool
for each device and a centralized, leader-based setup for shared resources.

Each `Device` object creates a fixed pool of `DeviceThread` workers upon
initialization. A designated "root" device is responsible for creating a
shared barrier and a shared dictionary of locks for all devices. Work (scripts)
is distributed statically among the worker threads.

NOTE: This implementation contains multiple, severe concurrency bugs:
1. The `ReusableBarrierCond` class is not a correctly implemented reusable
   barrier and will fail to synchronize threads properly across multiple cycles.
2. The `get_data` and `set_data` methods are not thread-safe. They are called
   without any locks, creating a critical data race condition that compromises
   the correctness of the entire simulation.
3. The synchronization logic within the `DeviceThread` run loop is overly
   complex and uses clumsy locking patterns for initialization and finalization.
"""

from threading import Event, Thread, Condition, Lock

class ReusableBarrierCond(object):
    """
    An implementation of a barrier using a Condition variable.
    
    BUG: This is not a correct implementation of a *reusable* barrier. It is
    prone to race conditions where threads from one cycle (or "generation") can
    get mixed with threads from the next, as there is no second phase to
    prevent this.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread notifies everyone and resets the counter for the next wave.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Wait to be notified by the last thread.
            self.cond.wait()
        self.cond.release()


class Device(object):
    """Represents a device with a persistent pool of worker threads."""

    def __init__(self, device_id, sensor_data, supervisor):
        self.root_device = 0
        self.init_lock = Lock()
        self.finalize_lock = Lock()
        self.max_threads = 8 # Number of worker threads per device.
        # A barrier for synchronizing the worker threads *within* this single device.
        self.device_barrier = ReusableBarrierCond(self.max_threads)

        # --- Shared Resources (to be provided by the root device) ---
        self.neighbours = None
        self.barrier = None # The main barrier for *all* threads from *all* devices.
        self.locks = None # Shared dictionary mapping locations to locks.
        self.dict_lock = None # A lock to protect the shared `locks` dictionary itself.

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # NOTE: Appears unused.
        self.scripts = []
        self.timepoint_done = Event()
        # Create and start the persistent worker thread pool.
        self.threads = [DeviceThread(self, i) for i in range(self.max_threads)]
        for thread in self.threads:
            thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes shared resources via a leader-election pattern."""
        if self.device_id == self.root_device:
            # This is the root device; it creates the shared resources.
            self.locks = {} # Shared dictionary for location locks.
            self.dict_lock = Lock() # Lock to protect the dictionary.
            # Global barrier for all worker threads across all devices.
            self.barrier = ReusableBarrierCond(self.max_threads * len(devices))
            
            # Distribute the shared objects to other devices.
            for device in devices:
                if device.device_id != self.root_device:
                    device.dict_lock = self.dict_lock
                    device.locks = self.locks
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """Assigns a script to be executed."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Signal that all scripts for the time step have been received.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data. BUG: Not thread-safe.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates sensor data. BUG: Not thread-safe.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """A persistent worker thread within a device's thread pool."""

    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device %d-Thread %d" % (device.device_id, thread_id))
        self.device = device
        self.thread_id = thread_id # Unique ID (0-7) within the device.

    def run(self):
        """Main loop for the worker thread."""
        while True:
            # --- Initialization for the time step ---
            # Use a lock to ensure only one worker thread fetches the neighbor list.
            with self.device.init_lock:
                if self.device.neighbours is None:
                    self.device.neighbours = self.device.supervisor.get_neighbours()
            neighbours = self.device.neighbours

            if neighbours is None:
                break # Shutdown signal.

            # All workers wait until the device signals that scripts are ready.
            self.device.timepoint_done.wait()

            # --- Work Phase ---
            # Statically distribute scripts among the workers in the pool.
            # Each thread takes every `max_threads`-th script.
            for i in range(self.thread_id, len(self.device.scripts), self.device.max_threads):
                (script, location) = self.device.scripts[i]

                # Safely check for and create a location lock if it doesn't exist.
                with self.device.dict_lock:
                    if location not in self.device.locks:
                        self.device.locks[location] = Lock()

                # Acquire the lock for this specific location.
                self.device.locks[location].acquire()
                try:
                    script_data = []
                    # BUG: get_data is not thread-safe. This block is racy.
                    for device in neighbours:
                        script_data.append(device.get_data(location))
                    script_data.append(self.device.get_data(location))
                    
                    # Filter out None values from devices that don't have the location.
                    script_data = [d for d in script_data if d is not None]

                    if script_data:
                        result = script.run(script_data)
                        # BUG: set_data is not thread-safe. This block is racy.
                        for device in neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)
                finally:
                    # Ensure the location lock is always released.
                    self.device.locks[location].release()

            # --- Synchronization Phase ---
            # 1. All threads *within this device* synchronize at a local barrier.
            self.device.device_barrier.wait()

            # 2. One thread resets the device state for the next time step.
            with self.device.finalize_lock:
                if self.device.neighbours is not None:
                    self.device.scripts = [] # Clear scripts for next round
                    self.device.neighbours = None
                    self.device.timepoint_done.clear()

            # 3. All threads *from all devices* synchronize at the global barrier.
            #    BUG: This barrier is not implemented correctly and will fail.
            self.device.barrier.wait()
