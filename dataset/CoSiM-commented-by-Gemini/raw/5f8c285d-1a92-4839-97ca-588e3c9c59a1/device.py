# -*- coding: utf-8 -*-
"""
Models a distributed network of computational devices with a flawed concurrency model
that will lead to deadlock.

This module's simulation is characterized by:
- A centralized setup process where Device 0 creates and starts the main thread
  for all other devices in the simulation.
- A custom, but correct, two-phase reusable barrier for synchronization.
- A critical deadlock flaw in the `DeviceThread`. Each thread attempts to acquire
  locks on all its neighbors before processing its scripts, a classic circular
  locking dependency that will cause the system to freeze.
"""

from threading import Thread, Event, Lock, Semaphore


class Device(object):
    """
    Represents a single computational device in the distributed network.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Each device has its own lock.
        self.lock = Lock()
        self.all_scripts_received = Event()
        self.barrier = None
        self.thread = None
        self.devices = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the setup of all devices.

        Note the unusual design: Device 0 is responsible for creating and starting
        the main `DeviceThread` for all other devices in the simulation.
        """
        if self.device_id is 0:
            self.devices = devices
            
            self.barrier = ReusableBarrier(len(devices))
            self.thread = DeviceThread(self, self.lock, self.barrier)
            # Device 0 creates and starts threads for all other devices.
            for dev in devices:
                if dev.device_id is not 0:
                    dev.barrier = self.barrier
                    dev.thread = DeviceThread(dev, dev.lock, self.barrier)
                
                dev.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to the device or signals the end of a script batch."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.all_scripts_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down all device threads, initiated by Device 0."""
        if self.device_id is 0:
            for dev in self.devices:
                dev.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device. Contains a critical deadlock flaw.
    """
    def __init__(self, device, lock, barrier):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.lock = lock
        self.barrier = barrier

    def run(self):
        """
        The main execution loop. This implementation will cause a deadlock.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for the signal that all scripts for the timepoint are assigned.
            self.device.all_scripts_received.wait()
            self.device.all_scripts_received.clear()

            # Process all assigned scripts for this timepoint.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # *** DEADLOCK FLAW ***
                # This loop attempts to acquire a lock on every neighboring device.
                # If two devices are neighbors, Device A will lock Device B, and
                # Device B will simultaneously try to lock Device A, resulting
                # in a classic circular-dependency deadlock. The system will freeze.
                for device in neighbours:
                    device.lock.acquire()
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    result = script.run(script_data)

                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # The lock release loop is also misplaced. It is outside the script
                # processing loop, meaning the thread holds all neighbor locks while
                // processing all its scripts. Even if fixed, the deadlock remains.
                for device in neighbours:
                    device.lock.release()

            # Wait at the barrier for all other (deadlocked) threads.
            self.barrier.wait()

class ReusableBarrier():
    """
    A correct, custom implementation of a reusable, two-phase barrier.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Use a list for mutable counter
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks until all threads reach the barrier, using two phases."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes a single phase of barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            
            # The last thread to arrive releases all other waiting threads.
            if count_threads[0] == 0:
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use.
                count_threads[0] = self.num_threads
        
        # All threads block here until released.
        threads_sem.acquire()
