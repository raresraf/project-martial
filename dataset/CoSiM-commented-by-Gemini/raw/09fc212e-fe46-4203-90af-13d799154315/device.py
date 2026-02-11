"""
This module simulates a distributed device network using a master thread per
device that spawns worker threads for each task. Synchronization is managed
by a global barrier and a complex, flawed system for sharing locks.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """
    A correct, two-phase reusable barrier for synchronizing multiple threads.
    It uses a pair of semaphores to prevent race conditions.
    """
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


class Device(object):
    """
    Represents a device in the simulation.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.threads = []
        self.cores_no = 8
        self.neighbours = []
        self.alldevices = []
        
        # --- Synchronization Primitives ---
        self.timepoint_done = Event()
        self.barrier = None
        # Pre-allocates a list for locks, to be populated later.
        self.locks = [None] * 100

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources. The first device to call this creates the
        global barrier.
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
        Assigns a script and attempts to create/share a lock for its location.

        CRITICAL FLAW: The lock initialization logic is complex and has a race
        condition. If two devices are assigned a script for the same *new*
        location at the same time, they could both create a new Lock object
        instead of sharing one.
        """
        if script is not None:
            self.scripts.append((script, location))
            
            # Attempt to find a pre-existing lock for this location from other devices.
            lock_found = False
            for device in self.alldevices:
                if device.locks[location] is not None:
                    self.locks[location] = device.locks[location]
                    lock_found = True
                    break
            # If no lock was found, create a new one.
            if not lock_found:
                self.locks[location] = Lock()
            
            self.script_received.set()
        else:
            # End of assignments for this time step.
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class MyThread(Thread):
    """A worker thread to execute a single script."""
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """Acquires a lock, runs the script, and releases the lock."""
        self.device.locks[self.location].acquire()
        try:
            script_data = []
            # Gather data...
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Run script and propagate results...
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)
        finally:
            self.device.locks[self.location].release()


class DeviceThread(Thread):
    """The master thread for a device."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main time-stepped loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # End of simulation.

            # 1. Wait for supervisor to signal that scripts are assigned.
            self.device.timepoint_done.wait()
            self.device.neighbours = neighbours

            # --- Thread Spawning ---
            # CRITICAL FLAW: This loop will only ever spawn up to 8 threads,
            # regardless of how many scripts are assigned. If more than 8
            # scripts exist, they will be ignored for this time step.
            count = 0
            for (script, location) in self.device.scripts:
                if count >= self.device.threads_number:
                    break
                count += 1
                thread = MyThread(self.device, location, script, neighbours)
                self.device.threads.append(thread)

            # 2. Start and wait for all worker threads to complete.
            for thread in self.device.threads:
                thread.start()
            for thread in self.device.threads:
                thread.join()
            self.device.threads = [] # Clear the list for the next step.

            # 3. Reset event and wait at the global barrier.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
