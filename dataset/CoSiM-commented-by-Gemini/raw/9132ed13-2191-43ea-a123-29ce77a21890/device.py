# -*- coding: utf-8 -*-
"""
Models a distributed system of concurrent devices using a 'thread-per-script'
model, where each script execution is handled by a new, short-lived thread.

NOTE: This implementation contains a critical race condition in the lazy
initialization of location locks, which can break the mutual exclusion guarantee.
"""

from threading import Lock, Thread, Event, Semaphore


class Device(object):
    """
    Represents a device in the simulation. It creates a control thread that
    manages the execution of scripts for each time step.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device and its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.location_locks = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources (barrier, lock dictionary) for all devices.
        Device 0 acts as a leader to create the shared objects.
        """
        # This class attribute is set but never used.
        Device.devices_no = len(devices)

        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            self.location_locks = {}
        else:
            # Non-leader devices get a reference to the leader's objects.
            self.barrier = devices[0].barrier
            self.location_locks = devices[0].location_locks

    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else: # A None script is a sentinel for the end of a timepoint's assignments.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a device, which spawns worker threads."""
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_scripts(self, script, location, neighbours):
        """
        The target function for worker threads. Executes a single script.

        NOTE: This method contains a critical race condition.
        """
        # --- FLAW: Race Condition on Lock Creation ---
        # The following block to check for and create a new lock is not atomic.
        # Two threads could simultaneously find that a lock is None and both
        # attempt to create it, leading to one lock being overwritten.
        lock_location = self.device.location_locks.get(location)
        if lock_location is None and location is not None:
            self.device.location_locks[location] = Lock()
            lock_location = self.device.location_locks[location]
        
        lock_location.acquire()
        try:
            script_data = []
            # Aggregate data from neighbors and self.
            for device in neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Run the script and disseminate results.
                result = script.run(script_data)
                for device in neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
        finally:
            lock_location.release()

    def run(self):
        """The main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()
            
            # Spawn a new thread for each script assigned in this time step.
            tlist = []
            for (script, location) in self.device.scripts:
                thread = Thread(target=self.run_scripts, args=(script, location, neighbours))
                tlist.append(thread)
                thread.start()
            
            # Wait for all worker threads to complete.
            for thread in tlist:
                thread.join()
            
            # Reset for the next timepoint.
            self.device.timepoint_done.clear()
            self.device.scripts = []
            
            # Synchronize with all other devices.
            self.device.barrier.wait()


class ReusableBarrierSem():
    """A correct two-phase reusable barrier using semaphores."""
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
