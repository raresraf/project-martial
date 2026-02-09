"""
@file device.py
@brief Defines a device model with a flawed and deadlock-prone synchronization scheme.

@warning This implementation is critically flawed. The mechanism for sharing locks
         in `assign_script` is convoluted and fragile. More importantly, the worker
         thread (`InsideDeviceThread`) contains a guaranteed deadlock, as it acquires
         a lock and then waits on an event that will never be set for the current
         timepoint's execution.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    """
    Represents a device in the simulation. This class contains unconventional
    and buggy synchronization logic.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.lock_for_data = [None] * 100
        self.inside_threads = [] 
        self.stored_devices = [] 
        self.barrier = None 

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes locks and a shared barrier.

        @note Each device creates its own list of 100 locks. The subsequent attempt
              to share these locks in `assign_script` is broken.
        """
        for i in range(100):
            self.lock_for_data[i] = Lock()
        
        barrier = ReusableBarrierSem(len(devices))
        
        for device in devices:
            device.barrier = barrier
            self.stored_devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script and attempts to share locks in a flawed manner.
        
        @warning The lock sharing logic here is broken. It causes all devices
                 to ultimately share the lock list of the last device in the
                 `stored_devices` list, overwriting their own locks.
        """
        if script is not None:
            self.scripts.append((script, location))
            
            # This loop overwrites the device's lock list repeatedly, a buggy
            # attempt at sharing locks.
            for device in self.stored_devices:
                self.lock_for_data = device.lock_for_data
            
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. This read is not synchronized."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data. This write is not synchronized."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a device.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()
            
            # Block Logic: Create and start a worker thread for each script.
            for (script, location) in self.device.scripts:
                inside_thread = InsideDeviceThread(self.device, script, location, neighbours)
                self.device.inside_threads.append(inside_thread)
                inside_thread.start()
            
            # Wait for all (deadlocked) worker threads to complete.
            for inside_thread in self.device.inside_threads:
                inside_thread.join()

            del self.device.inside_threads[:]
            self.device.timepoint_done.clear()
            
            # Wait at the barrier for all other devices to finish the timepoint.
            self.device.barrier.wait()


class InsideDeviceThread(Thread):
    """
    A worker thread that contains a deadlock.
    """
    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self, name="Inside Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script after acquiring a lock, but deadlocks while waiting
        for an event.
        """
        self.device.lock_for_data[self.location].acquire()

        # DEADLOCK: The thread acquires a lock and then waits for an event that
        # is only set when a *new* script is assigned. Since no new scripts will
        # be assigned for the current timepoint, this thread will wait forever,
        # holding the lock and preventing any other thread from using that location.
        self.device.script_received.wait()

        script_data = []
        
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = self.script.run(self.location, result)
            
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        self.device.lock_for_data[self.location].release()