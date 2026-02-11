
"""
This module defines a simulated device for a concurrent processing network.

This implementation is characterized by its use of class-level (static)
synchronization primitives, including a globally shared ReusableBarrier. It also
employs a BoundedSemaphore to throttle the number of active worker threads per
device and uses a static dictionary for location-based locks.
"""

from threading import Event, Thread, Lock, BoundedSemaphore
from barrier import ReusableBarrier


class Device(object):
    """
    Represents a single device in the network.

    It uses a static barrier for synchronization across all device instances and
    an instance-level semaphore to limit its own concurrency.
    """

    # Class-level variables shared across all instances of Device.
    timepoint_barrier = None
    barrier_lock = Lock()

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        # A semaphore to limit the number of concurrent worker threads per device to 8.
        self.max_threads_semaphore = BoundedSemaphore(8)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    @staticmethod
    def setup_devices(devices):
        """
        Initializes the shared timepoint barrier for all devices.

        This method uses a static variable and a lock (in a double-checked
        locking pattern) to ensure the ReusableBarrier is initialized only
        once for the entire class.

        Args:
            devices (list): The list of devices to be set up.
        """
        
        
        

        
        # Use a double-checked locking pattern to initialize the barrier
        # as a singleton for the class.
        if Device.timepoint_barrier is None:
            Device.barrier_lock.acquire()
            if Device.timepoint_barrier is None:
                Device.timepoint_barrier = ReusableBarrier(len(devices))
            Device.barrier_lock.release()

    def assign_script(self, script, location):
        """Assigns a script to the device for execution."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for the device's lifecycle."""

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop.
        
        It spawns worker threads for each script, but throttles them using a
        semaphore to not exceed a fixed number of concurrent workers.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            threads = []
            


            for (script, location) in self.device.scripts:
                # Acquire the semaphore. This will block if the device already
                # has 8 active worker threads.
                self.device.max_threads_semaphore.acquire()

                
                worker_thread = ScriptWorkerThread(self.device, neighbours, location, script)
                threads.append(worker_thread)
                worker_thread.start()

            
            for thread in threads:
                thread.join()

            
            self.device.timepoint_done.clear()

            
            # Wait at the global barrier for all other devices to finish.
            Device.timepoint_barrier.wait()


class ScriptWorkerThread(Thread):
    """
    A worker thread that executes a single script.

    It uses a static (class-level) dictionary of locks to ensure that only
    one thread can operate on a specific location at a time, across the entire
    system.
    """

    

    # A class-level dictionary holding locks for each location.
    # This is shared across all instances of ScriptWorkerThread.
    locations_lock = {}

    def __init__(self, device, neighbours, location, script):


        super(ScriptWorkerThread, self).__init__()
        self.device = device
        self.neighbours = neighbours
        self.location = location
        self.script = script

        
        # Lazily initialize the lock for this location if it doesn't exist.
        if location not in ScriptWorkerThread.locations_lock:
            ScriptWorkerThread.locations_lock[location] = Lock()

    def run(self):
        """
        Executes the script logic with location-based locking and semaphore release.
        """
        
        # Acquire the lock for this specific location.
        ScriptWorkerThread.locations_lock[self.location].acquire()

        script_data = []
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            
            result = self.script.run(script_data)

            
            for device in self.neighbours:


                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        ScriptWorkerThread.locations_lock[self.location].release()

        
        # Release the semaphore, allowing another worker thread to be started
        # by the parent device.
        self.device.max_threads_semaphore.release()
