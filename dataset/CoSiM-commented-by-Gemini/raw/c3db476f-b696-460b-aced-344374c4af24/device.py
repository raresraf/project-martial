from threading import Lock, Thread, Event, Condition

# A hardcoded constant for the number of location-specific locks.
# This is not robust as it assumes a fixed, known number of data locations.
max_size = 100

class MyThread(Thread):
    """
    A worker thread responsible for executing a single script safely.
    
    Functional Utility: Each instance of this class handles one unit of work.
    It ensures thread safety by acquiring a location-specific lock before
    accessing any data and releasing it upon completion.
    """
    def __init__(self, dev_thread, neighbors, location, script):
        Thread.__init__(self)
        self.dev_thread = dev_thread
        self.neighbors = neighbors
        self.location = location
        self.script = script

    def run(self):
        # Acquire the specific lock for the target location to prevent data races.
        self.dev_thread.device.location_lock[self.location].acquire()
        script_data = []
        
        # Data gathering phase (protected by the location lock).
        for device in self.neighbors:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.dev_thread.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Script execution and data propagation phase (protected by the lock).
        if script_data:
            result = self.script.run(script_data)
            
            for device in self.neighbors:
                device.set_data(self.location, result)
            
            self.dev_thread.device.set_data(self.location, result)
            
        # Release the location-specific lock.
        self.dev_thread.device.location_lock[self.location].release()

class ReusableBarrier():
    """
    An attempted implementation of a reusable barrier using a Condition variable.

    Warning: This implementation is critically flawed and NOT safely reusable.
    It suffers from a race condition where a fast thread can loop around and
    decrement the counter before all waiting threads have been woken up from the
    previous barrier wait, which will cause the simulation to deadlock.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Causes a thread to wait. This method is not safe for reuse."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # The last thread notifies all others and immediately resets the counter.
            # This is a race condition.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a device in a simulation that uses a flawed, `Condition`-based
    barrier for synchronization.

    Architectural Role: This model uses a leader (device 0) to initialize a
    shared (but broken) barrier and a list of location-based locks. It then
    dispatches one worker thread per assigned script for parallel execution.
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
        self.cond_barrier = None
        self.location_lock = None

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes a shared barrier and a list of locks."""
        if self.device_id == devices[0].device_id: # Designates device 0 as leader.
            self.cond_barrier = ReusableBarrier(len(devices))
            self.location_lock = []
            # Create a hardcoded number of locks.
            for i in range(0, max_size):
                self.location_lock.append(Lock())
            # Distribute the shared objects to all devices.
            for dev in devices:
                dev.cond_barrier = self.cond_barrier
                dev.location_lock = self.location_lock

    def assign_script(self, script, location):
        """Assigns a script or triggers the start of a time-step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A `None` script signals that scripts are ready and the time-step can proceed.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from the device's local data store."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates data in the device's local data store."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread, dispatching one new worker thread per script.
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
            
            # Wait for the supervisor to signal that scripts are ready.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Block Logic: Spawn one worker thread for each assigned script.
            thread_list = []
            for (script, location) in self.device.scripts:
                thread_list.append(MyThread(self, neighbours, location, script))

            # Start and wait for all worker threads to complete.
            for thr in thread_list:
                thr.start()
            for thr in thread_list:
                thr.join()

            # Block Logic: End-of-step synchronization.
            # The thread waits on the flawed barrier, which will likely cause
            # the simulation to deadlock after the first time-step.
            self.device.cond_barrier.wait()
            self.device.timepoint_done.wait()