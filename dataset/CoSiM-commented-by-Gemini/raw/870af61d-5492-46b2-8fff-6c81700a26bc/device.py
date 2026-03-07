"""
This Python 2 module provides a complex and flawed implementation of a
distributed device simulation.

The framework attempts to coordinate devices using a shared barrier and a complex,
custom locking mechanism for data locations. A 'root' device, chosen via `max()`,
is responsible for setting up synchronization primitives.

However, the design contains several critical flaws:
- The locking mechanism is not sound: locks are acquired in the main device
  thread and released in worker threads, which is a dangerous practice.
- Worker threads are started and immediately joined, resulting in sequential
  execution instead of the intended parallelism.
- The master election via `max(devices)` is fragile and relies on undefined
  comparison behavior for the Device class.
- The overall control flow within the DeviceThread is convoluted.
"""

from threading import Event, Thread, Lock
# This import assumes a 'reusable_barrier.py' file exists with a ReusableBarrier class.
from reusable_barrier import ReusableBarrier


class Device(object):
    """
    Represents a device in the simulation.

    This class contains a convoluted and unsafe locking mechanism.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device."""
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = None
        self.locations_locks = None
        self.lock = None
        self.devices = None

    def __str__(self):
        """Returns a string representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes synchronization objects.

        NOTE: Master election using `max(devices)` is fragile as `Device`
        objects have no natural ordering.
        """
        
        

        root = max(devices)
        if self == root:
            map_locks = {}
            lock = Lock()
            barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.set_barrier(barrier)
                device.set_locations_locks(map_locks)
                device.set_lock(lock)
        self.devices = devices

    def assign_script(self, script, location):
        """Assigns a script to the device for execution."""
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        

        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's thread to terminate."""
        

        self.thread.join()

    def set_barrier(self, barrier):
        """
        Sets the barrier and starts the main device thread.

        NOTE: Starting the thread here is an unusual design choice.
        """
        
        self.thread = DeviceThread(self, barrier)
        self.thread.start()

    def set_locations_locks(self, locations_locks):
        """Sets the shared dictionary for location-specific locks."""
        
        self.locations_locks = locations_locks

    def set_lock(self, lock):
        """Sets the shared global lock."""
        
        self.lock = lock

    def acquire_location(self, location):
        """
        Lazily initializes and acquires a lock for a specific location.
        
        This uses a double-locking pattern (global lock, then location lock).
        """
        
        location = str(location)
        self.lock.acquire()
        if (location in self.locations_locks) is False:
            self.locations_locks[location] = Lock()
        self.locations_locks[location].acquire()
        self.lock.release()

    def release_location(self, location):
        """Releases the lock for a specific location."""
        
        location = str(location)


        self.locations_locks[location].release()


class DeviceThread(Thread):
    """
    The main control thread for a device.

    NOTE: The logic in this thread is flawed. It acquires locks in this
    thread and passes them to worker threads to be released. It also starts
    and immediately joins worker threads, defeating the purpose of parallelism.
    """
    

    def __init__(self, device, barrier):
        """Initializes the device thread."""
        


        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
    def run(self):
        """The main simulation loop."""

        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.barrier.wait() # Synchronize at the start of the step.
            self.device.timepoint_done.wait() # Wait for all scripts to be assigned.
            self.device.timepoint_done.clear()
            threads = []

            
            for (script, location) in self.device.scripts:

                script_data = []

                # CRITICAL FLAW: Lock is acquired here in the main thread.
                self.device.acquire_location(location)

                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # A worker thread is created to run the script.
                    thread = ScriptThread(script, script_data, self.device, neighbours, location)
                    thread.start()
                    threads.append(thread)
                else:
                    # If there's no data, the lock must be released here.
                    self.device.release_location(location)

            # CRITICAL FLAW: Joining threads here makes execution sequential.
            for thread in threads:
                thread.join()

class ScriptThread(Thread):
    """
    A worker thread to execute a script.

    NOTE: This thread releases a lock that was acquired by a different thread
    (the parent DeviceThread), which is a dangerous and incorrect practice.
    """
    
    def __init__(self, script, data, device, neighbours, location):
        """Initializes the worker thread."""
        
        Thread.__init__(self, name="Script Thread %d" % device.device_id)
        self.script = script
        self.data = data
        self.device = device
        self.neighbours = neighbours
        self.location = location
    def run(self):
        """Runs the script, updates data, and releases the lock."""
        result = self.script.run(self.data)

        
        # Update data on neighbors and the local device.
        for device in self.neighbours:
            device.set_data(self.location, result)

        
        self.device.set_data(self.location, result)

        # CRITICAL FLAW: Lock is released here in the worker thread.
        self.device.release_location(self.location)

