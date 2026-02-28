# -*- coding: utf-8 -*-
"""
Models a distributed network of computational devices using a custom barrier and
a flawed, race-prone thread pool implementation.

This module's simulation is characterized by:
- A locally defined `ReusableBarrierSem` for two-phase synchronization.
- A centralized setup process where one "master" device (the one with the lowest ID)
  creates and distributes a shared barrier and a dictionary of locks.
- A flawed concurrency model where the main `DeviceThread` creates a fixed number
  of worker threads (`SingleDeviceThread`) and passes them all a reference to the
  same list of tasks. The workers then use the non-atomic `list.pop()` method,
  creating a severe race condition.
"""

from threading import Event, Thread, Lock, Semaphore


class Device(object):
    """
    Represents a single computational device in the distributed network.
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

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the setup of all devices.

        A single "master" device (the one with the lowest ID) creates a shared
        barrier and a dictionary of locks for all unique data locations, then
        distributes them to all other devices.
        """
        flag = True
        device_number = len(devices)

        # This is a convoluted way to elect the device with the lowest ID as master.
        for dev in devices:
            if self.device_id > dev.device_id:
                flag = False

        # Only the master device executes the setup logic.
        if flag == True:
            barrier = ReusableBarrierSem(device_number)
            map_locations = {}
            tmp = {}
            for dev in devices:
                dev.barrier = barrier
                # Find all unique locations from all devices and create a lock for each.
                tmp = list(set(dev.sensor_data) - set(map_locations))
                for i in tmp:
                    map_locations[i] = Lock()
                # Distribute the same dictionary of locks to all devices.
                dev.map_locations = map_locations
                tmp = {}

    def assign_script(self, script, location):
        """Assigns a script to the device or signals the end of a script batch."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device. Implements a flawed thread pool.
    """
    def __init__(self, device):
        Thread.__init__(self)
        self.device = device

    def run(self):
        """
        The main execution loop for the device thread.
        """
        while True:
            self.device.timepoint_done.clear()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for supervisor to signal that all scripts for the timepoint are assigned.
            self.device.timepoint_done.wait()
            
            script_list = []
            thread_list = []
            index = 0
            for script in self.device.scripts:
                script_list.append(script)

            # Create a fixed pool of 8 worker threads.
            for i in xrange(8):
                # WARNING: The same mutable `script_list` is passed to all threads.
                # This will cause a race condition in the worker threads.
                thread = SingleDeviceThread(self.device, script_list, neighbours, index)
                thread.start()
                thread_list.append(thread)
            
            # Wait for all threads in the pool to complete.
            for i in xrange(len(thread_list)):
                thread_list[i].join()
            
            # Synchronize with all other devices.
            self.device.barrier.wait()

class SingleDeviceThread(Thread):
    """
    A worker thread designed to process one script from a shared list.
    """
    def __init__(self, device, script_list, neighbours, index):
        Thread.__init__(self)
        self.device = device
        # This list is shared and modified by multiple threads without a lock.
        self.script_list = script_list
        self.neighbours = neighbours
        self.index = index

    def run(self):
        """
        Pops a script from the shared list and executes it.
        """
        # CRITICAL FLAW: `list.pop()` is not atomic. Multiple threads calling this
        # on the same list concurrently will lead to race conditions, causing
        # some scripts to be missed and others to raise errors. The `index` (always 0)
        # makes this even more problematic.
        if self.script_list != []:
            (script, location) = self.script_list.pop(self.index)
            self.compute(script, location)

    def update(self, result, location):
        """Broadcasts the script result to neighbors and the local device."""
        for device in self.neighbours:
            device.set_data(location, result)
        self.device.set_data(location, result)

    def collect(self, location, neighbours, script_data):
        """Gathers data for a given location, protected by a lock."""
        self.device.map_locations[location].acquire()
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

    def compute(self, script, location):
        """Orchestrates data collection, script execution, and result update."""
        script_data = []
        self.collect(location, self.neighbours, script_data)

        if script_data != []:
            result = script.run(script_data)
            self.update(result, location)
        
        # Release the lock acquired in the collect phase.
        self.device.map_locations[location].release()

class ReusableBarrierSem():
    """
    A custom, reusable barrier implemented with two semaphores for two-phase synchronization.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads

        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all threads reach the barrier, using two phases."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            # The last thread to arrive releases all others for this phase.
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """The second synchronization phase, preventing thread lapping."""
        with self.counter_lock:
            self.count_threads2 -= 1
            # The last thread to arrive releases all others for this phase.
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()
