"""A distributed device simulation with flawed concurrent-update logic.

This module implements a simulation of devices that operate in synchronized
time steps. It uses a custom `ReusableBarrierSem` for synchronization.

WARNING: This implementation contains multiple severe concurrency flaws,
including a racy setup protocol, a deadlock-prone locking strategy for
updating neighbor devices, and a confusing double-barrier synchronization
in the main loop that is also likely to cause deadlocks.
"""

from threading import Event, Semaphore, Lock, Thread



class ReusableBarrierSem(object):
    """A custom, reusable barrier for synchronizing a fixed number of threads.

    This barrier uses a two-phase signaling protocol with two semaphores to
    ensure it can be safely used multiple times in a loop without race conditions.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)
    
    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """Executes the first phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()
    
    def phase2(self):
        """Executes the second phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """Represents a single device in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        
        self.my_lock = Lock()
        self.barrier = ReusableBarrierSem(0)
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up a shared barrier between devices.

        WARNING: This implementation is racy. It assumes device 0 will act as
        the master and sets its barrier first. Other devices then copy the
        reference from device 0. If `setup_devices` is not called on device 0
        first, other devices will copy a `None` or old barrier.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """Adds a script to the workload for the current time step."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of assignments for this step.
            self.script_received.set()
            self.timepoint_done.set()


    def get_data(self, location):
        """Gets data from a specific location (not thread-safe)."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Sets data at a specific location (not thread-safe)."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Gracefully shuts down the device."""
        self.thread.join()



class MyScriptThread(Thread):
    """A worker thread to execute one script; its locking is deadlock-prone."""

    def __init__(self, script, location, device, neighbours):
        """Initializes the worker thread."""
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """Executes the script.
        
        WARNING: This method's locking strategy is a textbook example of a
        deadlock risk. It iterates through neighbors and acquires each device's
        lock one by one. If two threads attempt to do this on the same set of
        devices but in a different order, they will deadlock.
        """
        script_data = []

        # Data is read without locks, which is a race condition.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = self.script.run(script_data)

            # This loop is highly likely to cause deadlocks.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()

class DeviceThread(Thread):
    """The main control thread for a device's lifecycle."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop.
        
        WARNING: The synchronization logic is suspect. It waits at a barrier,
        then waits for events, then does work, then waits for another event,
        then waits at a barrier again. For this to work, all devices must
        be able to trigger each other's events in a coordinated way to avoid
        deadlocking at one of the barriers or events.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # 1. First barrier synchronization.
            self.device.barrier.wait()

            # 2. Waits for script assignment to be complete.
            self.device.script_received.wait()
            script_threads = []
            
            # 3. Spawns and runs worker threads for the scripts.
            for (script, location) in self.device.scripts:
                script_threads.append(MyScriptThread(script,
                    location, self.device, neighbours))
            for thread in script_threads:
                thread.start()
            for thread in script_threads:
                thread.join()
            
            # 4. Waits for a "timepoint done" signal.
            self.device.timepoint_done.wait()
            
            # 5. Second barrier synchronization.
            self.device.barrier.wait()
            self.device.script_received.clear()
