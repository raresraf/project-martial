"""
This module implements a device simulation for a concurrent system.

It uses an inefficient threading model where new threads are created for each
script in every timepoint. Its most critical issue is a Time-of-check to
time-of-use (TOCTOU) race condition in the worker thread's logic, where data is
read without a lock and then written with a lock, making it fundamentally unsafe.
"""

from threading import Event, Thread, Lock, Condition

class Device(object):
    """
    Represents a single device in the simulation.
    
    Each device has its own local lock and participates in a global barrier
    set up by a leader device.
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
        self.bariera = ReusableBarrier(0)
        # This is a device-local lock, not shared with other devices.
        self.lacat_date = Lock()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes a shared barrier for all devices.
        
        The device with ID 0 acts as a leader to create and distribute the barrier.
        """


        if self.device_id == 0:
            barria = ReusableBarrier(len(devices))
            for device in devices:
                device.bariera = barria

    def assign_script(self, script, location):
        """Assigns a script to the device for execution."""

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location. This read is not locked.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        """Sets sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()



class DeviceThread(Thread):
    """
    The main control thread for a device.
    
    This thread implements an inefficient model where it creates a new pool of
    worker threads (`MiniT`) for each timepoint.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_list = list()

    def run(self):
        """The main execution loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            # Inefficiently create a new thread for every script.
            self.thread_list = list()
            for (script, location) in self.device.scripts:
                minithrd = MiniT(neighbours, self.device, location, script)
                self.thread_list.append(minithrd)

            for i in range(len(self.thread_list)):
                self.thread_list[i].start()

            for i in range(len(self.thread_list)):
                self.thread_list[i].join()

            self.device.timepoint_done.clear()
            self.device.bariera.wait()

class MiniT(Thread):
    """A worker thread that executes a single script."""
    def __init__(self, neighbours, device, location, script):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script logic.
        
        This method contains a critical Time-of-check to time-of-use (TOCTOU)
        race condition because it reads data without a lock and then writes
        data with a lock.
        """
        script_data = []
        # --- START OF CRITICAL FLAW ---
        # The `get_data` calls are performed without any lock. Multiple threads
        # can interleave their reads here.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)



        if script_data != []:
            result = self.script.run(script_data)

            # The locks are only acquired here, after the data has already been
            # read. Another thread could have modified the data in the meantime,
            # meaning this thread is now operating on stale data and will
            # overwrite the intermediate change.
            for device in self.neighbours:
                device.lacat_date.acquire()
                device.set_data(self.location, result)
                device.lacat_date.release()

            self.device.lacat_date.acquire()
            self.device.set_data(self.location, result)
            self.device.lacat_date.release()
        # --- END OF CRITICAL FLAW ---

class ReusableBarrier(object):
    """
    A simple, non-reentrant barrier implementation using a Condition variable.

    This barrier is not a proper two-phase reusable barrier and may be prone to
    race conditions if threads enter the `wait` cycle at different speeds.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        
        self.count_threads = self.num_threads
        
        self.cond = Condition()
        

    def wait(self):
        """Blocks until all threads have reached the barrier."""
        
        
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            
            self.cond.notify_all()
            
            self.count_threads = self.num_threads
        else:
            
            self.cond.wait()

        
        self.cond.release()
