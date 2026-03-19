"""
This module provides an advanced framework for simulating a distributed network of devices.

It features a robust two-phase semaphore-based reusable barrier and a custom,
fine-grained distributed locking mechanism. This is designed to manage concurrent
access to sensor data in a multi-threaded environment, ensuring data consistency
during synchronized, time-stepped script executions.
"""


from threading import Event, Thread, RLock, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    A reusable, two-phase thread barrier implemented with Semaphores.

    This barrier ensures that threads from a new "wave" of operations cannot
    begin until all threads from the previous wave have completed, preventing race
    conditions common in simpler barrier implementations.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0) # Gate for the first phase
        self.threads_sem2 = Semaphore(0) # Gate for the second phase

    def wait(self):
        """Blocks the calling thread until all threads reach the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase of the barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread arrives, opens the gate for all waiting threads.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next use
        self.threads_sem1.acquire()

    def phase2(self):
        """Second synchronization phase to prevent premature re-entry."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread arrives, opens the second gate.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for next use
        self.threads_sem2.acquire()

class MyLock(object):
    """
    A custom lock that associates an RLock with a device and data zone.

    This allows for a fine-grained locking strategy where specific data zones
    can be locked per device.
    """
    def __init__(self, deviceId, zone):
        self.lock = RLock()
        self.dev = deviceId
        self.zone = zone

    def acquire(self):
        self.lock.acquire()

    def release(self):
        self.lock.release()

def get_leader(devices):
    """Utility function to find the device with the lowest ID."""
    leader = devices[0].device_id
    for dev in devices:
        if dev.device_id < leader:
            leader = dev.device_id
    return leader

class Device(object):
    """
    Represents a single device in the simulation, managing state, data, and threads.
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
        # Shared synchronization objects, initialized by setup_devices.
        self.barrier = None
        self.global_lock = None
        self.gl1 = None
        self.lock_list = None


    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects (barrier, locks).

        This is orchestrated by a single "leader" device.
        """
        leader = get_leader(devices)
        if self.device_id == leader:
            global_lock = RLock()
            gl1 = RLock()
            lock_list = []
            barrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                dev.barrier = barrier
                dev.global_lock = global_lock
                dev.gl1 = gl1
                dev.lock_list = lock_list

    def assign_script(self, script, location):
        """
        Assigns a script or signals the end of assignments for a time-step.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts for this step have been assigned.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class MyThread(Thread):
    """A worker thread that executes scripts on sensor data."""

    def __init__(self, device, scripts, neighbours):
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):
        """Gathers data, runs script, and updates data on self and neighbors."""
        dev = self.device
        scripts = self.scripts
        neighbours = self.neighbours

        for (script, location) in scripts:
            script_data = []

            # Gather data from neighbors and self.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # Run the script and update data on all relevant devices.
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                dev.set_data(location, result)

def contains(my_list, searched):
    """Helper to check if an element is in a list."""
    for elem in my_list:
        if elem == searched:
            return 1
    return 0

class DeviceThread(Thread):
    """Main control thread for a device, managing the execution lifecycle."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def existent_lock(self, new_lock):
        """Checks if a specific device/zone lock already exists in the shared list."""
        for lock in self.device.lock_list:
            if new_lock.dev == lock.dev:
                if new_lock.zone == lock.zone:
                    return 1
        return 0

    def get_index(self, dev, zone):
        """Gets the index of a specific device/zone lock from the shared list."""
        my_list = self.device.lock_list
        for i in range(len(my_list)):
            if dev == my_list[i].dev:
                if zone == my_list[i].zone:
                    return i
        return -1

    def run(self):
        """
        The main operational loop for the device.
        
        This loop co-ordinates the multi-phase process of:
        1. Lock discovery: Identifying all data zones that will be affected.
        2. Lock acquisition: Acquiring locks for those zones.
        3. Execution: Spawning worker threads to run scripts.
        4. Lock release: Releasing the locks.
        5. Synchronization: Waiting for all devices to complete the time-step.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal

            # Wait for signal that all scripts for the timestep have been assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Phase 1: Lock Discovery/Registration
            self.device.global_lock.acquire()
            my_list = []
            for (script, location) in self.device.scripts:
                # Register locks for the device and its neighbors for the script's location.
                new_lock = MyLock(self.device.device_id, location)
                if self.existent_lock(new_lock) == 0:
                    self.device.lock_list.append(new_lock)
                
                for device in neighbours:
                    new_lock = MyLock(device.device_id, location)
                    if self.existent_lock(new_lock) == 0:
                        self.device.lock_list.append(new_lock)
                    index = self.get_index(device.device_id, location)
                    if contains(my_list, index) == 0:
                        my_list.append(index)
            self.device.global_lock.release()

            # Phase 2: Lock Acquisition
            self.device.gl1.acquire()
            for index in my_list:
                self.device.lock_list[index].acquire()
            self.device.gl1.release()

            # Phase 3: Execution
            # Spawn worker threads to run the assigned scripts.
            length = len(self.device.scripts)
            if length == 1:
                trd = MyThread(self.device, self.device.scripts, neighbours)
                trd.start()
                trd.join()
            else:
                tlist = []
                for i in range(length):
                    lst = [self.device.scripts[i]]
                    trd = MyThread(self.device, lst, neighbours)
                    trd.start()
                    tlist.append(trd)
                for i in range(length):
                    tlist[i].join()

            # Phase 4: Lock Release
            for index in my_list:
                self.device.lock_list[index].release()

            # Phase 5: Global Synchronization
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()