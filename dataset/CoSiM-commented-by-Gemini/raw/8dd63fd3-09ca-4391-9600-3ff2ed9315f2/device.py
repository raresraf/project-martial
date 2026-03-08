# -*- coding: utf-8 -*-
"""
Models a distributed system of concurrent devices using a 'thread-per-script' model.

This script simulates a network of devices where each device's main control thread
spawns a new, short-lived worker thread (`MiniDeviceThread`) for every single script
it needs to execute in a given time step. Synchronization across the entire system
for specific data locations is handled by a globally shared array of locks.
"""

from threading import Thread, Semaphore, Event, Lock


class ReusableBarrierSem(object):
    """A reusable barrier for thread synchronization using semaphores.

    This implementation uses a two-phase protocol to ensure that threads can wait
    at the barrier multiple times.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads reach the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Device(object):
    """Represents a single device in the distributed network simulation.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): The device's local sensor data, keyed by location.
        supervisor (object): A reference to the central simulation supervisor.
        scripts (list): A list of scripts to execute for the current time step.
        thread (DeviceThread): The main control thread for this device.
        barrier (ReusableBarrierSem): A barrier shared by all devices for global synchronization.
        locks (list): A globally shared list of locks, indexed by location.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.locks = []
        # Assumes location IDs are integers and finds the max ID to size the lock array.
        self.nrlocks = max(sensor_data) if sensor_data else -1

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared resources (barrier and locks) for all devices.

        Device 0 is designated as the leader to create the shared resources.
        """
        if self.device_id == 0:
            # Create a global barrier for synchronizing the main thread of each device.
            self.barrier = ReusableBarrierSem(len(devices))
            for _, device in enumerate(devices):
                device.barrier = self.barrier

        if self.device_id == 0:
            # Determine the total number of location-specific locks needed.
            listmaxim = [dev.nrlocks for _, dev in enumerate(devices)]
            number = max(listmaxim) if listmaxim else -1

            # Create a global list of locks, one for each possible location.
            for _ in range(number + 1):
                self.locks.append(Lock())
            
            # Distribute the reference to the global lock list to all devices.
            for _, device in enumerate(devices):
                device.locks = self.locks

    def assign_script(self, script, location):
        """Assigns a script to the device for a specific location."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. Not internally thread-safe."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data. Not internally thread-safe."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class MiniDeviceThread(Thread):
    """A short-lived worker thread responsible for executing a single script."""
    def __init__(self, device, script, location, neighbours):
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """Executes one script, ensuring exclusive access to the location."""
        # Acquire the global lock for this specific data location.
        self.device.locks[self.location].acquire()
        
        script_data = []
        # Aggregate data from neighbors and self.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            # Run the script and disseminate the results.
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
        
        # Release the lock, allowing another script to process this location.
        self.device.locks[self.location].release()


class DeviceThread(Thread):
    """The main control thread for a single Device."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.nr_iter = None

    def run(self):
        """The main simulation loop."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for the supervisor to finish assigning all scripts for this time step.
            self.device.timepoint_done.wait()

            # --- Thread-per-Script Execution ---
            # NOTE: The following logic for batching thread creation is overly complex
            # and could be simplified by creating all threads, starting all, then joining all.
            self.nr_iter = len(self.device.scripts) // 8
            
            if self.nr_iter == 0:
                # If fewer than 8 scripts, process them all at once.
                scriptthreads = [MiniDeviceThread(self.device, script, location, neighbours)
                                 for (script, location) in self.device.scripts]
                for thread in scriptthreads:
                    thread.start()
                for thread in scriptthreads:
                    thread.join()
            else:
                # Flawed batching logic for larger numbers of scripts.
                count = 0
                size = 8
                for _ in range(self.nr_iter):
                    scriptthreads = []
                    for idx in range(count, size):
                        script, location = self.device.scripts[idx]
                        scriptthreads.append(MiniDeviceThread(self.device, script, location, neighbours))
                    
                    for thread in scriptthreads:
                        thread.start()
                    for thread in scriptthreads:
                        thread.join()
                    
                    count += 8
                    if size + 8 > len(self.device.scripts):
                        size = len(self.device.scripts)
                    else:
                        size += 8
            
            # Wait at the global barrier to synchronize with all other devices.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
