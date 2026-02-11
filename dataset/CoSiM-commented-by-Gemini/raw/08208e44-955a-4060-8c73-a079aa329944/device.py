"""
This module defines a simulated distributed device network using a Bulk Synchronous Parallel (BSP) model.
It features a custom ReusableBarrier for thread synchronization, and Device objects that
execute computational scripts in parallel based on data from their neighbors.
"""

from threading import Event, Thread, Lock, Semaphore
from multiprocessing import cpu_count


class ReusableBarrier():
    """
    A custom, two-phase reusable barrier for synchronizing a fixed number of threads.

    This barrier ensures that all participating threads wait at the barrier point until
    every thread has arrived. It uses a two-phase mechanism (two semaphores) to allow
    the barrier to be safely reused in a loop without race conditions.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that will be synchronized by this barrier.
        """
        self.num_threads = num_threads
        # Counters for each phase, stored in a list to be mutable across instances.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        # Semaphores to block and release threads for each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)
 
    def wait(self):
        """
        Causes a thread to wait at the barrier. Consists of two distinct phases
        to ensure no thread proceeds until all have completed the previous phase.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier synchronization.

        The last thread to arrive resets the counter and releases the semaphores
        for all other waiting threads.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread releases all other threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of the barrier.
                count_threads[0] = self.num_threads
        # All threads, including the last one, block here until released.
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device in a distributed network simulation.

    Each device has its own data, can be assigned scripts to run, and communicates
    with its neighbors under the coordination of a supervisor.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data.
            supervisor: An object that manages the network topology (e.g., neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []

    def __str__(self):
        return "Device %d" % self.device_id

    def set_lock(self, lock1, lock2, barrier1, barrier2):
        """
        Assigns shared synchronization primitives to the device.
        """
        self.lock1 = lock1
        self.lock2 = lock2
        self.script_received = barrier1
        self.timepoint_done = barrier2

    def setup_devices(self, devices):
        """
        Initializes and starts the device network.

        A single device (the first in the list) acts as the primary, creating the
        shared locks and barriers and distributing them to all other devices.
        This method also starts the main thread for each device.

        Args:
            devices (list): A list of all Device objects in the network.
        """
        # Block Logic: Centralized setup of synchronization primitives.
        # The first device creates shared locks and barriers for all devices to use.
        if self.device_id == devices[0].device_id:
            lock1 = Lock()
            lock2 = Lock()
            barrier1 = ReusableBarrier(len(devices))
            barrier2 = ReusableBarrier(len(devices))
            for dev in devices:
                dev.set_lock(lock1, lock2, barrier1, barrier2)

        self.thread = DeviceThread(self)
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a computational script to be run by the device.
        """
        if script is not None:
            self.scripts.append((script, location))

    def get_data(self, location):
        """
        Retrieves data from a specific location in the device's sensor data.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates data at a specific location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Waits for the device's main thread to terminate.
        """
        self.thread.join()


class MyThread(Thread):
    """
    A worker thread to execute a single script and propagate its result.
    """
    def __init__(self, script, script_date, device, neighbours, location):
        Thread.__init__(self)
        self.script = script
        self.script_data = script_date
        self.result = None
        self.device = device
        self.neighbours = neighbours
        self.location = location

    def run(self):
        """
        Executes the script and writes the result to the local device and its neighbors.
        """
        result = self.script.run(self.script_data)

        # Block Logic: Propagate results under a lock to prevent race conditions.
        self.device.lock2.acquire()
        for device in self.neighbours:
            device.set_data(self.location, result)
        
        self.device.set_data(self.location, result)
        self.device.lock2.release()


class DeviceThread(Thread):
    """
    The main control-loop thread for a device, orchestrating its lifecycle.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Implements the main time-stepped execution loop for the device.
        Follows a Bulk Synchronous Parallel (BSP) pattern.
        """
        while True:
            # Safely get the list of neighbors for the current time step.
            self.device.lock1.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.lock1.release()

            # The supervisor signals termination by returning None.
            if neighbours is None:
                break
    
            # --- BARRIER 1: Script Reception ---
            # All devices wait here until they have all been assigned scripts for the timestep.
            self.device.script_received.wait()
            
            threads = []
            
            # --- Data Gathering ---
            # For each script, gather necessary data from this device and its neighbors.
            for (script, location) in self.device.scripts:
                script_data = []
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    threads.append(MyThread(script, script_data, self.device, neighbours, location))

            # --- Parallel Execution with Throttling ---
            # Run the script-executing threads in batches to avoid overwhelming the system.
            step = cpu_count() * 2
            for i in range(0, len(threads), step):
                # Start a batch of threads.
                for j in range(step):
                    if i + j < len(threads):
                        threads[i + j].start()
                # Wait for the current batch to complete before starting the next.
                for j in range(step):
                    if i + j < len(threads):
                        threads[i + j].join()

            # --- BARRIER 2: Timepoint Completion ---
            # All devices wait here until all computation for the current timestep is done.
            self.device.timepoint_done.wait()
