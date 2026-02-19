


"""
@file device.py
@brief A device simulation using a high-level thread pool for concurrency.

This module defines a simulation framework where each device uses a thread pool
from `multiprocessing.dummy` to process scripts concurrently. This approach
abstracts away much of the manual thread management, using `pool.map` to
distribute and execute tasks.
"""


from threading import Thread, Event, Lock
from multiprocessing.dummy import Pool


class ReusableBarrierSem(object):
    """
    A two-phase reusable barrier implemented with Semaphores.
    
    Ensures that a fixed number of threads can repeatedly synchronize at a barrier point.
    """
    
    def __init__(self, num_threads):
        
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Last thread to arrive releases all waiting threads for this phase.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next use.
        self.threads_sem1.acquire()

    def phase2(self):
        """The second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Last thread to arrive releases all waiting threads for this phase.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for next use.
        self.threads_sem2.acquire()


class Device(object):
    """
    Represents a device in the simulation, which owns and manages a `DeviceThread`.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.locks = {} # Will be populated and shared by device 0.
        self.barrier = None # Shared barrier for inter-device synchronization.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects (barrier and locks).

        Functional Utility: Device 0 acts as a coordinator. It creates a single barrier
        for all devices. It also discovers all unique data locations across all devices
        and creates a shared dictionary of locks, which it then distributes.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            # Discover all unique locations and create a lock for each.
            for current_device in devices:
                current_device.barrier = self.barrier
                for location in current_device.sensor_data:
                    if not self.locks.has_key(location):
                        self.locks[location] = Lock()
            # Distribute the completed lock dictionary to all devices.
            for current_device in devices:
                current_device.locks = self.locks

    def assign_script(self, script, location):
        """Assigns a script to the device for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's worker thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a device, managing a thread pool to execute scripts.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.num_threads = 8
        self.device = device
        # The pool provides a high-level abstraction for managing worker threads.
        self.threads_pool = Pool(self.num_threads)
        self.neighbours = None

    def calculate(self, raw_data):
        """
        The target function for worker threads in the pool.

        This method processes a single script, handling locking, data aggregation,
        execution, and result distribution.
        @param raw_data A tuple containing the script and its target location.
        """
        script, location = raw_data
        data_list = []
        # Acquire a lock for the data location to ensure thread-safe access.
        with self.device.locks[location]:
            for i in range(len(self.neighbours)):
                current_data = self.neighbours[i].get_data(location)
                if current_data is not None:
                    data_list.append(current_data)

            my_data = self.device.get_data(location)
            if my_data is not None:
                data_list.append(my_data)

            if data_list:
                new_data = script.run(data_list)
                # Distribute the result to all participating devices.
                for i in range(len(self.neighbours)):
                    self.neighbours[i].set_data(location, new_data)
                self.device.set_data(location, new_data)

    def run(self):
        """
        The main simulation loop for the device.
        """
        while True:
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                # End of simulation.
                break

            # Wait for the supervisor to signal that scripts for the timepoint are assigned.
            self.device.timepoint_done.wait()
            
            # Block Logic: Use pool.map to distribute scripts to the thread pool.
            # This is a blocking call; it will not return until all scripts are processed.
            self.threads_pool.map(self.calculate, self.device.scripts)
            
            # After all work is done, synchronize with other devices.
            self.device.barrier.wait()
            self.device.timepoint_done.clear()
        
        # Gracefully shut down the thread pool.
        self.threads_pool.close()
        self.threads_pool.join()

