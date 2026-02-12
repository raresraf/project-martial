# -*- coding: utf-8 -*-
"""
This module defines a simulation of networked devices that process data in
synchronized time steps. This implementation uses a manual, fixed-size thread
pool within each device to handle concurrent script execution and a
semaphore-based reusable barrier for synchronization between devices.

Classes:
    Device: A node in the network.
    MyThread: A thread to execute a single data processing task.
    DeviceThread: The main control loop for a device, managing the thread pool.
    ReusableBarrierSem: A custom semaphore-based reusable barrier.
"""

from threading import Event, Thread, Lock, Semaphore

class Device(object):
    """
    Represents a device in the simulation.
    
    Each device manages its own sensor data and a pool of threads to execute
    assigned scripts. It coordinates with other devices via a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's local sensor data.
            supervisor (object): The central simulation coordinator.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.script_received = Event()
        self.timepoint_done = Event()
        self.lock = {}  # Dictionary of location-based locks, shared among all devices.
        self.barrier = None
        self.devices = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization objects for all devices.

        This method is intended to be called by a single external actor to ensure
        all devices share the same barrier and lock objects. It creates one lock
        for each unique sensor location across the entire system.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.devices = devices
        # All devices will share this single barrier instance.
        self.barrier = ReusableBarrierSem(len(self.devices))

        # Create a shared dictionary of locks, one for each location.
        # This assumes location names are unique across all devices.
        all_locations = set()
        for device in devices:
            all_locations.update(device.sensor_data.keys())
        for location in all_locations:
            self.lock[location] = Lock()

        # Distribute the shared barrier and lock dictionary to all devices.
        for device in self.devices:
            device.barrier = self.barrier
            device.lock = self.lock

    def assign_script(self, script, location):
        """
        Assigns a script to the device. If script is None, it signals that
        all scripts for the current timepoint have been received.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that script assignment is done for this timepoint.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class MyThread(Thread):
    """

    A thread to execute one script. It gathers data, runs the script,
    and disseminates the result.
    """

    def __init__(self, my_id, device, neighbours, lock, script, location):
        Thread.__init__(self, name="Thread %d from device %d" % (my_id, device.device_id))
        self.device = device
        self.my_id = my_id
        self.neighbours = neighbours
        self.lock = lock
        self.script = script
        self.location = location

    def run(self):
        """
        Executes the task in a thread-safe manner for the given location.
        """
        # Use a `with` statement to ensure the lock for the specific location
        # is acquired and released properly.
        with self.lock[self.location]:
            script_data = []
            
            # --- Data Gathering ---
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # --- Execution and Dissemination ---
            if script_data:
                result = self.script.run(script_data)
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """
    Main control loop for a device, managing a simple, fixed-size thread pool.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.numThreads = 0
        self.listThreads = []

    def run(self):
        """
        The main simulation loop, advancing in discrete timepoints.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Supervisor signals shutdown.
                break

            # Wait until all scripts for the timepoint have been assigned.
            self.device.script_received.wait()

            # --- Manual Thread Pool Management ---
            # This section implements a basic, fixed-size pool by "reaping"
            # dead threads and creating new ones.
            for (script, location) in self.device.scripts:
                if len(self.listThreads) < 8:
                    # If the pool is not full, create and start a new thread.
                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.append(thread)
                    thread.start()
                    self.numThreads += 1
                else:
                    # If the pool is full, find a completed thread to replace.
                    index = -1
                    for i in xrange(len(self.listThreads)):
                        if not self.listThreads[i].is_alive():
                            self.listThreads[i].join()
                            index = i
                            break # Found a finished thread.
                    
                    if index != -1:
                        # Replace the finished thread with a new one.
                        self.listThreads.pop(index)
                        thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                        self.listThreads.insert(index, thread)
                        thread.start()
                        self.numThreads += 1

            # Wait for all threads in the current batch to complete.
            for thread in self.listThreads:
                thread.join()
            self.listThreads = [] # Clear the list for the next timepoint.

            # Wait for a redundant signal (already ensured by joins).
            self.device.timepoint_done.wait()
            
            # Reset events for the next timepoint.
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            self.device.scripts = []
            
            # --- Synchronization Point ---
            # All devices wait here, ensuring the timepoint ends for everyone
            # before the next one begins.
            self.device.barrier.wait()


class ReusableBarrierSem():
    """
    A reusable barrier implemented using semaphores.

    This allows a group of threads to wait for each other at a certain point,
    and to do so repeatedly in a loop without race conditions.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0) # Gate for the first phase
        self.threads_sem2 = Semaphore(0) # Gate for the second phase

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0: # The last thread arrives
                # Release all waiting threads.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next use.
        self.threads_sem1.acquire() # All threads wait here.

    def phase2(self):
        """Second phase to prevent lapping in the next iteration."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0: # The last thread arrives
                # Release all waiting threads.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for next use.
        self.threads_sem2.acquire() # All threads wait here.