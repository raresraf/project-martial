# -*- coding: utf-8 -*-
"""
Models a distributed system of concurrent devices for a sensor network simulation.

This script defines a simulation where each device has a main control thread that,
for each time step, spawns a pool of worker threads to process scripts.
Synchronization for data locations is managed by a globally shared dictionary of locks.
"""

from threading import Event, Thread, Lock, Semaphore


class Device(object):
    """Represents a single device in the distributed network simulation.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): The device's local sensor data.
        supervisor (object): A reference to the central supervisor.
        scripts (list): Scripts assigned for the current time step.
        thread (DeviceThread): The main control thread for this device.
        barrier (ReusableBarrierSem): Shared barrier for global time step synchronization.
        map_locations (dict): Shared dictionary mapping data locations to Lock objects.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance and starts its control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared resources (barrier and locks) for all devices.

        An inefficient O(N^2) leader election selects the device with the lowest ID
        to perform the setup.
        """
        flag = True
        device_number = len(devices)

        # Inefficient leader election: O(N^2)
        for dev in devices:
            if self.device_id > dev.device_id:
                flag = False

        if flag is True: # If this is the device with the lowest ID
            barrier = ReusableBarrierSem(device_number)
            map_locations = {}
            
            # Create a shared dictionary of locks, one for each unique data location.
            for dev in devices:
                dev.barrier = barrier
                # Find new locations from the current device's sensor data.
                new_locations = set(dev.sensor_data.keys()) - set(map_locations.keys())
                for i in new_locations:
                    map_locations[i] = Lock()
            
            # Distribute the shared lock map to all devices.
            for dev in devices:
                dev.map_locations = map_locations


    def assign_script(self, script, location):
        """Assigns a script to be executed by the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location. Not internally thread-safe."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location. Not internally thread-safe."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a device, orchestrating work for each time step."""
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device

    def run(self):
        """The main simulation loop."""
        while True:
            self.device.timepoint_done.clear()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            self.device.timepoint_done.wait()
            
            script_list = list(self.device.scripts)
            thread_list = []
            
            # --- FLAWED WORK DISTRIBUTION ---
            # The following loop creates a severe race condition. All 8 threads
            # are initialized with the same list and the same index (0). They will
            # all compete to pop the first element, leading to incorrect behavior
            # and likely IndexErrors. This does not distribute the work as intended.
            for _ in xrange(8):
                # The same index `0` is passed to all threads.
                thread = SingleDeviceThread(self.device, script_list, neighbours, 0)
                thread.start()
                thread_list.append(thread)

            for thread in thread_list:
                thread.join()
            
            # Synchronize with all other devices before starting the next time step.
            self.device.barrier.wait()


class SingleDeviceThread(Thread):
    """A worker thread intended to process a single script.
    
    NOTE: Due to the bug in `DeviceThread.run`, this thread participates in a
    race condition to grab a script from a shared list rather than processing
    an assigned piece of work.
    """
    def __init__(self, device, script_list, neighbours, index):
        Thread.__init__(self)
        self.device = device
        self.script_list = script_list
        self.neighbours = neighbours
        self.index = index

    def run(self):
        """Pops a script from the shared list and executes it."""
        try:
            # Race condition: Multiple threads are popping from the same list.
            if self.script_list:
                (script, location) = self.script_list.pop(self.index)
                self.compute(script, location)
        except IndexError:
            # This exception is likely to occur for most threads.
            pass

    def update(self, result, location):
        """Disseminates the script result to itself and its neighbors."""
        for device in self.neighbours:
            device.set_data(location, result)
        self.device.set_data(location, result)

    def collect(self, location, neighbours, script_data):
        """Gathers data for a specific location from all neighbors."""
        # Acquire the global lock for the location to ensure data consistency.
        self.device.map_locations[location].acquire()
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

    def compute(self, script, location):
        """Orchestrates data collection, script execution, and result dissemination."""
        script_data = []
        self.collect(location, self.neighbours, script_data)

        if script_data:
            result = script.run(script_data)
            self.update(result, location)

        # Release the global lock for the location.
        self.device.map_locations[location].release()


class ReusableBarrierSem():
    """A reusable barrier implementation for synchronizing multiple threads."""
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
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()
