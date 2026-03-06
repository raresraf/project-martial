"""
This module implements a device simulation using a thread-per-script model, but
with a hard-coded limit on the number of threads per timepoint.

The `DeviceThread` controller creates a new `MyThread` worker for each assigned
script. However, it contains a major logical flaw where it will only process up
to 8 scripts per timepoint, silently ignoring any additional scripts.

The simulation also features fragile, non-thread-safe methods for initializing
shared resources (barriers and locks) on-the-fly, creating potential race
conditions.

The core worker logic in `MyThread` correctly uses location-specific locks to
ensure data consistency during a single script's execution.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """A standard two-phase reusable barrier using semaphores."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        self.phase1()
        self.phase2()

    def phase1(self):
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class MyThread(Thread):
    """A short-lived worker thread designed to execute one script."""
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """Executes a single script, protected by a location-specific lock."""
        # Acquire the lock for this specific location.
        self.device.locks[self.location].acquire()
        
        script_data = []
        # Safely gather data as the location lock is held.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = self.script.run(script_data)
            # Safely write data back.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
        
        self.device.locks[self.location].release()

class DeviceThread(Thread):
    """The main controller thread for a device."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()
            self.device.neighbours = neighbours

            # --- Worker Management ---
            # BUG: This implementation has a hard limit of 8 threads. If more than 8
            # scripts are assigned, the extras will be ignored.
            count = 0
            for (script, location) in self.device.scripts:
                if count >= self.device.threads_number:
                    break
                count += 1
                thread = MyThread(self.device, location, script, neighbours)
                self.device.threads.append(thread)

            # Start and join all created worker threads.
            for thread in self.device.threads:
                thread.start()
            for thread in self.device.threads:
                thread.join()
            
            # Reset state for the next timepoint.
            self.device.threads = []
            self.device.scripts = []
            self.device.timepoint_done.clear()
            
            # Synchronize with all other devices.
            self.device.barrier.wait()

class Device(object):
    """Represents a device and its resources."""
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.neighbours = []
        self.alldevices = []
        self.barrier = None
        self.threads = []
        self.threads_number = 8
        self.locks = [None] * 100
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Fragile, order-dependent setup method for initializing shared resources.
        """
        if self.barrier is None:
            barrier = ReusableBarrierSem(len(devices))
            self.barrier = barrier
            for d in devices:
                if d.barrier is None:
                    d.barrier = barrier
        
        for device in devices:
            if device is not None:
                self.alldevices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script and performs a racy, on-the-fly initialization of locks.
        """
        no_lock_for_location = 0
        if script is not None:
            self.scripts.append((script, location))
            # RACE CONDITION: This on-the-fly initialization of a shared resource
            # is not thread-safe.
            for device in self.alldevices:
                if device.locks[location] is not None:
                    self.locks[location] = device.locks[location]
                    no_lock_for_location = 1
                    break
            if no_lock_for_location == 0:
                self.locks[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. Not thread-safe on its own."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data. Not thread-safe on its own."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()
