from threading import Event, Thread, Lock
# Assumes a 'barrier.py' module with a ReusableBarrierSem class.
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a device in a simulation that uses a bounded, "sliding window"
    pool of worker threads.

    Architectural Role: This model uses a leader (device 0) to set up shared
    synchronization objects. Its most distinct feature is a two-level locking
    strategy and a unique concurrency pattern in its main thread.

    Warning: The locking implementation is highly inefficient. A single global
    lock protects all basic `get_data` and `set_data` operations, which
    serializes all data access across the entire system and creates a major
    performance bottleneck.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

        self.timepoint_done = Event()
        self.script_received = Event()
        self.barrier = None
        # A dictionary of fine-grained locks, one for each data location.
        self.location_lock = None
        # A single global lock that protects all low-level data access.
        self.lock = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a global lock, location-specific locks,
        and a shared barrier using a leader-follower pattern.
        """
        if self.device_id == 0: # Designates device 0 as the leader.
            self.lock = Lock()
            self.barrier = ReusableBarrierSem(len(devices))
            self.location_lock = {}
            for device in devices:
                device.location_lock = self.location_lock
                # Populate the shared dictionary with location-specific locks.
                for location in device.sensor_data:
                    self.location_lock[location] = Lock()
                # Assign shared objects to follower devices.
                if device.device_id != 0:
                    device.barrier = self.barrier
                    device.lock = self.lock

    def assign_script(self, script, location):
        """Assigns a script to the device or triggers the start of a time-step."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data from the local store under a global lock.
        Note: This global lock serializes all `get_data` calls across all devices.
        """
        with self.lock:
            res = self.sensor_data.get(location)
        return res

    def set_data(self, location, data):
        """
        Updates data in the local store under a global lock.
        Note: This global lock serializes all `set_data` calls across all devices.
        """
        with self.lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()

    def run_script(self, script, location, neighbours):
        """
        The logic for executing a single script, using nested locking.
        """
        # Acquire the fine-grained lock for this specific location.
        self.location_lock[location].acquire()
        script_data = []
        
        # Block Logic: Data gathering phase. Each call to `get_data` will
        # acquire and release the single global lock.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        data = self.get_data(location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Script execution and data propagation.
        if script_data:
            result = script.run(script_data)
            for device in neighbours:
                device.set_data(location, result)
            self.set_data(location, result)
        
        # Release the fine-grained location lock.
        self.location_lock[location].release()

class DeviceThread(Thread):
    """
    The main control thread that manages a bounded-size pool of worker threads.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
    def run(self):
        """The main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Supervisor signals shutdown.
            
            # Wait for the supervisor to signal the start of a time-step.
            self.device.timepoint_done.wait()
            
            thread_list = []

            # Block Logic: Bounded Concurrency via a "Sliding Window" pool.
            # This loop ensures that no more than 8 scripts from this device are
            # executing concurrently.
            for (script, location) in self.device.scripts:
                if len(thread_list) < 8:
                    # If the pool is not full, create and start a new thread.
                    t = Thread(target=self.device.run_script, args=(script, location, neighbours))
                    t.start()
                    thread_list.append(t)
                else:
                    # If the pool is full, wait for the oldest thread to finish
                    # before continuing to the next script.
                    out_thread = thread_list.pop(0)
                    out_thread.join()
                    # A new thread for the current script will be added on the next loop iteration.

            # Wait for all remaining threads in the pool to complete.
            for thread in thread_list:
                thread.join()
                
            # Synchronize with all other devices at the end of the time-step.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()