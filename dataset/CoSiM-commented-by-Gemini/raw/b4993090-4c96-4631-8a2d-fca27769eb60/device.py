# -*- coding: utf-8 -*-
"""
This module implements a simulation of a distributed sensor network with a
more complex threading model. Each device is a multi-threaded entity,
and all threads across all devices synchronize using a global barrier.

Classes:
    ReusableBarrier: A custom implementation of a reusable barrier.
    Device: Represents a multi-threaded device in the network.
    DeviceThread: One of several execution threads belonging to a single Device.
"""

from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    A custom implementation of a reusable barrier for thread synchronization.

    This barrier allows a set of threads to wait for each other to reach a
    certain point before proceeding. It uses a two-phase protocol to allow for reuse.
    """

    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The number of threads that will use this barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait until all threads have called this method."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread to arrive releases all waiting threads for phase 1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread to arrive releases all waiting threads for phase 2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a multi-threaded device in the sensor network.

    Each device consists of a fixed number of DeviceThread instances that
    work in parallel to process scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor (Supervisor): The supervisor for this device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.timepoint_done = Event()
        self.scripts = []

        # Each device has its own barrier for its internal worker threads.
        self.barrier_worker = ReusableBarrier(8)
        self.setup_event = Event()
        self.devices = []
        self.locks = None
        self.neighbours = []
        # A global barrier for all threads of all devices.
        self.barrier = None
        self.threads = []

        # Each device spawns 8 worker threads.
        for i in range(8):
            self.threads.append(DeviceThread(self, i))

        for thr in self.threads:
            thr.start()

        self.location_lock = []

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources for all devices in the network.

        Device 0 is the coordinator and is responsible for creating the global
        barrier and the list of location locks.
        """
        if self.device_id == 0:
            # The global barrier synchronizes all threads from all devices.
            barrier = ReusableBarrier(len(devices)*8)
            self.barrier = barrier
            location_max = 0
            for device in devices:
                device.barrier = barrier
                for location, data in device.sensor_data.iteritems():
                    if location > location_max:
                        location_max = location
                device.setup_event.set()
            self.setup_event.set()

            # Create a list of locks, one for each location.
            self.location_lock = [None] * (location_max + 1)

            for device in devices:
                device.location_lock = self.location_lock
                device.setup_event.set()
            self.setup_event.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed.

        This method also handles the lazy initialization of locks for locations.
        """
        busy = 0
        if script is not None:
            self.scripts.append((script, location))
            # Lazy initialization of the lock for this location.
            if self.location_lock[location] is None:
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        busy = 1
                        break
                if busy == 0:
                    self.location_lock[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its threads."""
        for thr in self.threads:
            thr.join()


class DeviceThread(Thread):
    """
    An execution thread for a Device. Each device has multiple such threads.
    """

    def __init__(self, device, idd):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent device.
            idd (int): The ID of this thread within the device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.idd = idd

    def run(self):
        """
        The main loop of the worker thread.
        """
        self.device.setup_event.wait()

        while True:
            # Thread 0 is responsible for fetching neighbor information.
            if self.idd == 0:
                neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours = neighbours

            # Synchronize with other threads of the same device.
            self.device.barrier_worker.wait()

            if self.device.neighbours is None:
                break

            # Wait for the signal that all scripts have been assigned for the timepoint.
            self.device.timepoint_done.wait()
            self.device.barrier_worker.wait()

            i = 0
            # Each thread processes a subset of the scripts.
            for (script, location) in self.device.scripts:
                if i % 8 == self.idd:
                    with self.device.location_lock[location]:
                        script_data = []
                        
                        # Gather data from neighbors.
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        # Gather data from the current device.
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        if script_data != []:
                            result = script.run(script_data)

                            # Update data on neighbors.
                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            # Update data on the current device.
                            self.device.set_data(location, result)
                i = i + 1

            
            self.device.timepoint_done.clear()
            # Synchronize with all threads of all devices.
            self.device.barrier.wait()
