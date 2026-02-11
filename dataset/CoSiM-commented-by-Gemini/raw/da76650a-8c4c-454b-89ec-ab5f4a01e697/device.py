"""
This module implements a complex device simulation using a multi-threaded
architecture within each device.

Each `Device` manages its own internal pool of worker threads. The system uses
a two-level barrier synchronization scheme: an internal barrier for workers
within a single device, and a global barrier for all workers across all devices.
"""

from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrier


class Device(object):
    """
    Represents a device that manages an internal pool of worker threads.

    Each device creates a fixed set of 8 `DeviceThread` workers upon
    initialization to process scripts in parallel.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # A list of events, one for each internal worker thread.
        self.script_received = []
        for _ in xrange(8):
            self.script_received.append(Event())
        self.scripts = []
        self.devices = None
        self.barrier = None # This will hold the global, system-wide barrier.
        self.lock = Lock()
        self.locks = {} # Holds location-specific locks.
        self.neighbours = None

        # Each device has its own internal barrier for its 8 worker threads.
        thread_barrier = ReusableBarrier(8)
        self.threads = []
        for i in xrange(8):
            self.threads.append(DeviceThread(self, i, thread_barrier))
        for i in xrange(8):
            self.threads[i].start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes a global barrier for all worker threads in the system.

        The device with ID 0 is responsible for creating a single barrier that
        is shared across all worker threads of all devices.
        """
        
        self.devices = devices
        if self.device_id == 0:
            
            # The global barrier's size is the total number of worker threads.
            barrier = ReusableBarrier(8 * len(devices))
            lock = Lock()
            for device in devices:
                device.barrier = barrier
                device.lock = lock

    def assign_script(self, script, location):
        """
        Assigns a script and sets up a shared lock for its location.

        If a lock for the given location doesn't exist, it is created and
        distributed to all devices to ensure they all share the same lock
        instance for that location.
        """
        if script is not None:
            self.scripts.append((script, location))
            
            # Create and distribute a shared lock for the script's location.
            if location not in self.locks:
                with self.lock:
                    auxlock = Lock()
                    for device in self.devices:
                        device.locks[location] = auxlock
        else:
            # When script is None, it signals the end of assignments for a timepoint.
            # Wake up all 8 internal worker threads.
            for i in xrange(8):


                self.script_received[i].set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all internal worker threads to terminate."""
        for i in xrange(8):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    An internal worker thread for a single device.

    Each device runs 8 of these threads. They partition the device's script
    list and execute them in parallel, coordinating with both an internal and a
    global barrier.
    """

    def __init__(self, device, thread_id, barrier):
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id
        self.barrier = barrier # This is the device's internal barrier.

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            
            # The first worker thread (id 0) is responsible for fetching neighbors.
            if self.thread_id == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()

            
            # 1. Internal Barrier: Synchronize all 8 workers within this device.
            self.barrier.wait()
            if self.device.neighbours is None:
                break

            
            # 2. Wait for this specific thread's signal to start processing scripts.
            self.device.script_received[self.thread_id].wait()
            
            # Statically partition the work: each thread processes a subset of scripts.
            # e.g., thread 0 takes scripts 0, 8, 16...; thread 1 takes 1, 9, 17...
            for i in xrange(self.thread_id, len(self.device.scripts), 8):
                (script, location) = self.device.scripts[i]
                # Use the globally shared lock for this location.
                with self.device.locks[location]:
                    script_data = []
                    
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
                            device.set_data(location, result)
                        
                        self.device.set_data(location, result)

            
            self.device.script_received[self.thread_id].clear()
            
            # 3. Global Barrier: Wait for ALL threads from ALL devices to finish.
            self.device.barrier.wait()
