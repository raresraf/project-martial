from threading import Event, Thread, Condition, Lock


class ReusableBarrier():
    """
    An attempted implementation of a reusable barrier using a Condition variable.

    Warning: This implementation is not safely reusable and is prone to race
    conditions and deadlocks. It suffers from the "lost wakeup" problem. A fast
    thread that completes the barrier can loop around and decrement the counter
    before all threads from the previous `wait` call have woken up, leading to
    a system hang on the next iteration. A correct reusable barrier requires
    at least two phases or a generation counter.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
        
    def wait(self):
        """
        Causes a thread to wait. This method is not safe for reuse in a loop.
        """
        self.cond.acquire()
        
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread arrives, notifies all waiting threads.
            self.cond.notify_all()
            # It then immediately resets the counter. This is the race condition.
            self.count_threads = self.num_threads
        else:
            # Other threads wait to be notified.
            self.cond.wait()
            
        self.cond.release()

class Device(object):
    """
    Represents a device in a simulation featuring a flawed barrier and an
    unconventional "lock-on-write" data access pattern.

    Architectural Role: This model uses a simple leader-follower pattern where
    device 0 creates a shared (but broken) barrier. Each device uses a single lock
    to protect its `set_data` method, but `get_data` remains unprotected,
    creating significant potential for race conditions.
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
        self.barrier = None
        # Each device has a single lock to protect its own data on writes.
        self.lock = Lock()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes the shared barrier in a simple leader-follower model."""
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
        else:
            # Followers get a reference to the leader's barrier.
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """Assigns a script to the device or triggers the start of a time-step."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data. Warning: This operation is NOT thread-safe.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates data. This operation is intended to be thread-safe."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()

class MyThread(Thread):
    """
    A worker thread for a single script, using a "lock-on-write" pattern.
    
    Warning: The logic here contains a race condition. Data is read from neighbors
    using `get_data` without any locks, but then written using `set_data` with locks.
    The data could be stale by the time it is used, and the pattern of a thread
    acquiring another object's lock is unconventional and can be risky.
    """
    def __init__(self, neighbours, device, location, script):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.device = device
        self.location = location
        self.script = script

    def run(self):
        script_data = []
        # Data gathering phase: unprotected reads create a race condition.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Data propagation phase: protected writes.
        if script_data:
            result = self.script.run(script_data)
            # Acquire and release the lock for each neighbor individually.
            for device in self.neighbours:
                device.lock.acquire()
                device.set_data(self.location, result)
                device.lock.release()
            
            # Acquire and release the lock for the local device.
            self.device.lock.acquire()
            self.device.set_data(self.location, result)
            self.device.lock.release()

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

            # Wait for the supervisor to signal the start of a time-step.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            threads = []
            # Block Logic: Spawn one worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                t = MyThread(neighbours, self.device, location, script)
                t.start()
                threads.append(t)

            # Wait for all worker threads for this time-step to complete.
            for t in threads:
                t.join()
            
            # Wait at the (broken) barrier for all other devices to finish.
            self.device.barrier.wait()