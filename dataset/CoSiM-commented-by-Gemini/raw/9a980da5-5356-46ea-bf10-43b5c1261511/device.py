
"""
Models a distributed network of devices that process sensor data concurrently.

This script implements a device simulation where each device dynamically spawns a
new thread for every script execution within a timepoint.

NOTE: This implementation is inefficient due to the constant creation and
destruction of threads in each time step. It also contains race conditions in
the initialization of shared synchronization primitives.
"""

from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """A reusable barrier for thread synchronization using semaphores.

    This barrier uses a two-phase protocol to ensure that a specified number of
    threads can repeatedly synchronize at a point in an iterative algorithm.
    The counters are stored in single-element lists to simulate mutable integers.
    """
    
    def __init__(self, num_threads):
        """Initializes the ReusableBarrier."""
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads arrive."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()

class Device(object):
    """Represents a single device in the distributed sensor network.

    This implementation uses a racy, cooperative method to initialize shared
    synchronization objects and creates a new thread for each script.

    Attributes:
        device_id (int): A unique identifier for the device.
        thread (DeviceThread): The main orchestration thread for this device.
        barrier (ReusableBarrier): A shared barrier for timepoint synchronization.
        location_lock (list): A list used to store location-based locks. The
                              initialization is prone to race conditions.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barrier = None
        self.list_thread = []
        self.thread.start()
        self.location_lock = [None] * 100

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and shares the barrier among all devices.

        NOTE: This method has a race condition. Multiple devices could check
        `if self.barrier is None` simultaneously and create multiple barriers.
        """
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        for device in devices:
            if device is not None:
                self.devices.append(device)


    def assign_script(self, script, location):
        """Assigns a script and manages the lock for its execution location.

        NOTE: The lock initialization logic is cooperative and has a race condition.
        """
        flag = 0

        if script is not None:
            self.scripts.append((script, location))
            if self.location_lock[location] is None:
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        flag = 1
                        break
                if flag == 0:
                    self.location_lock[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to terminate."""
        self.thread.join()


class MyThread(Thread):
    """A worker thread to execute a single script task.

    An instance of this class is created for each script in each time step.
    """
    def __init__(self, device, location, script, neighbours):
        """Initializes the worker thread."""
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """The main execution logic for the thread.

        Acquires a lock, gathers data, runs the script, propagates the result,
        and releases the lock.
        """
        self.device.location_lock[self.location].acquire()
        script_data = []
        
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
        self.device.location_lock[self.location].release()


class DeviceThread(Thread):
    """The main orchestration thread for a single Device.

    This thread's main responsibility is to spawn worker threads (`MyThread`)
    for each script and manage synchronization across timepoints.
    """
    def __init__(self, device):
        """Initializes the main DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main control loop for the device.

        In each timepoint, it waits for scripts to be assigned, creates and runs
        a new thread for each one, waits for them to complete, and then
        synchronizes with other devices at a global barrier.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()

            for (script, location) in self.device.scripts:
                thread = MyThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            for thread_elem in self.device.list_thread:
                thread_elem.start()
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            self.device.list_thread = []

            self.device.timepoint_done.clear()
            self.device.barrier.wait()
