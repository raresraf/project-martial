# -*- coding: utf-8 -*-
"""
Models a distributed network of computational devices using a custom two-phase
reusable barrier and a thread-per-script execution model.

This module defines a simulation framework where `Device` objects operate concurrently.
Its key features are:
- A custom `ReusableBarrier` that uses a two-phase semaphore-based protocol to ensure
  that threads from a previous barrier wait cycle cannot interfere with a new one.
- A `DeviceThread` that, for each computational step (timepoint), spawns a new
  `NewThread` for every script to be executed. This thread-per-task model is highly
  inefficient compared to a thread pool but is implemented here.
- A decentralized and complex mechanism for creating and sharing locks among devices.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier():
    """
    A custom implementation of a reusable, two-phase barrier.

    This barrier ensures that all threads have completed one phase before any thread
    can begin the next, preventing race conditions where fast threads could loop
    around and re-enter the barrier before slow threads have left. It uses two
    semaphores to manage the two phases of synchronization.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters for each phase, stored in a list to be mutable.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        # Semaphores for each phase of the barrier.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Blocks the calling thread until all threads have reached the barrier.
        This is achieved by passing through two distinct synchronization phases.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes a single phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the mutable counter for the phase.
            threads_sem (Semaphore): The semaphore for this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # The last thread to arrive is responsible for releasing all other threads.
            if count_threads[0] == 0:
                # Release num_threads permits, one for each waiting thread.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of the barrier.
                count_threads[0] = self.num_threads
        # All threads, including the last one, will block here until a permit is released.
        threads_sem.acquire()

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
        self.devices = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barrier = None
        # A list to hold the threads spawned for the current timepoint's scripts.
        self.list_thread = []
        self.thread.start()
        # An array to hold locks for different locations. The size is fixed.
        self.location_lock = [None] * 100

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier and device list for the simulation.
        Note: This setup is decentralized and can be initiated by any device,
        which is less robust than a centralized approach.
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
        """
        Assigns a script and manages lock creation/discovery for its location.
        Note: This on-demand lock discovery is complex and potentially racy. A
        centralized lock creation during setup would be more reliable.
        """
        ok = 0
        if script is not None:
            self.scripts.append((script, location))
            # If a lock for this location doesn't exist locally...
            if self.location_lock[location] is None:
                # ...try to find it on another device.
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        ok = 1
                        break
                # If no other device has the lock, create a new one.
                if ok == 0:
                    self.location_lock[location] = Lock()
            self.script_received.set()
        else:
            # A None script signals the end of a timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()

class NewThread(Thread):
    """
    A short-lived thread created to execute a single script.
    """
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        The execution logic for a single script.
        """
        script_data = []
        # Acquire the lock for the location to ensure exclusive data access.
        self.device.location_lock[self.location].acquire()
        
        # Invariant: Gather data from all neighbors for the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Invariant: Only execute the script if there is data to process.
        if script_data != []:
            result = self.script.run(script_data)
            
            # Broadcast the result to neighbors and the local device.
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)
        
        # Release the lock.
        self.device.location_lock[self.location].release()

class DeviceThread(Thread):
    """
    The main control thread for a Device. It uses a thread-per-script model.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop, orchestrating script execution and synchronization.
        """
        while True:
            # Get the current list of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # A None response signals simulation shutdown.
            if neighbours is None:
                break
            
            # Pre-condition: Wait until all scripts for the current timepoint are assigned.
            self.device.timepoint_done.wait()

            # Inefficient Model: Create a new thread for each script.
            for (script, location) in self.device.scripts:
                thread = NewThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            # Start all the newly created threads.
            for thread_elem in self.device.list_thread:
                thread_elem.start()
            # Wait for all of them to complete before proceeding.
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            self.device.list_thread = []

            # Reset the event and wait at the barrier for all other devices.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
