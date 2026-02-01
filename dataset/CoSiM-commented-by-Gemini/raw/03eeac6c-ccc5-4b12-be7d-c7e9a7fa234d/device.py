# -*- coding: utf-8 -*-
"""
/**
 * @file device.py
 * @brief This module simulates a network of distributed devices that execute scripts and synchronize their states.
 *
 * @details
 * The system is composed of `Device` objects, each running in its own `DeviceThread`.
 * These devices are managed by a central `Supervisor` (not defined in this file).
 * Devices can receive scripts and execute them on a pool of `ScriptThread` workers.
 *
 * Core functionalities include:
 * - Script execution: Devices receive scripts and execute them, using data from neighboring devices.
 * - Data synchronization: The result of a script execution is propagated back to the device and its neighbors.
 * - Concurrency control: `Lock` objects are used to manage access to shared data (sensor data and location-specific operations).
 * - Multi-threaded script processing: Each device maintains a thread pool to execute multiple scripts concurrently.
 * - Barrier synchronization: A `ReusableBarrier` is employed to synchronize all devices at specific timepoints, ensuring that all devices complete a phase before any device proceeds to the next one.
 *
 * The file also includes a definition for `ReusableBarrier` and a simple test case (`MyThread`)
 * demonstrating its usage, although these seem to be appended from another source and are not directly
 * integrated with the `Device` simulation logic.
 */
"""

from threading import Event, Thread, Lock
from reusable_barrier_semaphore import ReusableBarrier
import Queue
NUMBER_OF_THREADS = 8

class Device(object):
    """
    Represents a single device in a simulated distributed system.

    Each device operates autonomously in its own thread, processes scripts,
    interacts with its neighbors, and synchronizes with the entire system
    at designated timepoints using a barrier. It is managed by a central supervisor.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor readings,
                                keyed by location.
            supervisor (Supervisor): The central supervisor object that coordinates devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been received.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that the device has completed its operations for the current timepoint.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        # Lock to ensure thread-safe access to the device's sensor_data.
        self.data_lock = Lock()

    def __str__(self):
        """
        Returns a string representation of the device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and shares a reusable barrier among all devices in the system.

        This method ensures that all devices use the same barrier instance, allowing them
        to synchronize correctly.

        Args:
            devices (list): A list of all `Device` objects in the system.
        """
        
        barrier = ReusableBarrier(len(devices))

        # Functional Utility: Sets up a shared barrier for all devices to enable system-wide synchronization.
        if self.barrier is None:
            self.barrier = barrier

        for device in devices:

            if device.barrier is None:
                device.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a new script to be executed by the device.

        If a script is provided, it is added to the device's script list and the
        `script_received` event is set. If the script is `None`, it signals that
        the device has finished its tasks for the current timepoint.

        Args:
            script (Script): The script object to execute.
            location (any): The location context for the script execution.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (any): The key for the desired sensor data.

        Returns:
            The sensor data associated with the location, or `None` if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specific location.

        Args:
            location (any): The key for the sensor data to update.
            data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by waiting for its main thread to terminate.
        """
        self.thread.join()

class DeviceThread(Thread):
    """
    The main execution thread for a `Device`.

    This thread manages a pool of `ScriptThread` workers and a queue for incoming scripts.
    It orchestrates the device's lifecycle, including fetching neighbors from the supervisor,
    dispatching scripts for execution, and synchronizing with other devices.
    """
    
    # A class-level dictionary to hold locks for each location, ensuring that operations
    # on the same location are serialized across all devices.
    location_locks = {}

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent `Device` object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.threads = []
        self.scripts_queue = Queue.Queue()

    def run(self):
        """
        The main loop of the device thread.

        It continuously performs the following cycle:
        1. Retrieves the list of neighboring devices from the supervisor.
        2. Waits for scripts to be assigned for the current timepoint.
        3. Enqueues the scripts for execution by the worker threads.
        4. Waits at a barrier for all other devices to complete the current timepoint.
        This loop terminates when the supervisor signals a shutdown.
        """
        # Block Logic: Initializes a pool of worker threads to execute scripts concurrently.
        for _ in range(NUMBER_OF_THREADS):
            self.threads.append(ScriptThread(self.scripts_queue))

        for script_thread in self.threads:
            script_thread.start()

        # Block Logic: The main operational loop for the device.
        # Invariant: At the start of each iteration, the device is synchronized with all others from the previous timepoint.
        while True:
            
            # Fetches the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Pre-condition: Check if the supervisor has signaled a shutdown.
            if neighbours is None:
                # Send poison pills to terminate worker threads.
                for script_thread in self.threads:
                    self.scripts_queue.put(MyObjects(None, None, None, None,
                                                     False, None))
                break

            
            self.device.timepoint_done.wait()
            # Block Logic: Dispatches all assigned scripts for the current timepoint to the worker queue.
            for (script, location) in self.device.scripts:

                
                if location not in self.location_locks:
                    self.location_locks[location] = Lock()

                self.scripts_queue.put(MyObjects(self.device, location, script,
                                                 neighbours, True,
                                                 self.location_locks),
                                       block=True, timeout=None)
            self.device.timepoint_done.clear()

            
            # Functional Utility: Synchronizes with all other devices, ensuring all have completed the current timepoint.
            self.device.barrier.wait()

        # Wait for all worker threads to terminate cleanly.
        for script_thread in self.threads:
            script_thread.join()

class ScriptThread(Thread):
    """
    A worker thread that executes scripts from a shared queue.
    """

    def __init__(self, queue):
        """
        Initializes the ScriptThread.

        Args:
            queue (Queue.Queue): The queue from which to fetch script execution tasks.
        """
        Thread.__init__(self, name="Script Thread")
        self.queue = queue

    def run(self):
        """
        The main loop for the script execution thread.

        It continuously dequeues tasks (`MyObjects`) and executes the associated script.
        The loop terminates when it receives a poison pill (`my_objects.stop == False`).
        """
        while True:

            
            my_objects = self.queue.get(block=True, timeout=None)

            # Pre-condition: Check for the poison pill to terminate the thread.
            if my_objects.stop == False:
                break

            # Acquire a lock for the script's location to prevent race conditions.
            my_objects.location_locks[my_objects.location].acquire()

            script_data = []
            
            # Block Logic: Gathers data from all neighboring devices for the specified location.
            for device in my_objects.neighbours:
                data = device.get_data(my_objects.location)


                if data is not None:
                    script_data.append(data)

            
            # Gather data from the current device as well.
            my_objects.device.data_lock.acquire()
            data = my_objects.device.get_data(my_objects.location)
            my_objects.device.data_lock.release()

            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                # Functional Utility: Executes the script with the aggregated data from self and neighbors.
                result = my_objects.script.run(script_data)

                
                # Block Logic: Propagates the script's result to all neighboring devices.
                for device in my_objects.neighbours:
                    device.data_lock.acquire()
                    device.set_data(my_objects.location, result)
                    device.data_lock.release()

                
                # Update the local device's data with the result.
                my_objects.device.data_lock.acquire()
                my_objects.device.set_data(my_objects.location, result)
                my_objects.device.data_lock.release()

            my_objects.location_locks[my_objects.location].release()

class MyObjects():
    """
    A data-transfer object used to pass tasks and context to ScriptThreads via the queue.
    """

    def __init__(self, device, location, script, neighbours, stop, location_locks):
        """
        Initializes the task object.

        Args:
            device (Device): The device context for the task.
            location (any): The location context for the script.
            script (Script): The script to be executed.
            neighbours (list): A list of neighboring `Device` objects.
            stop (bool): A flag to signal thread termination (True to continue, False to stop).
            location_locks (dict): A reference to the shared dictionary of location locks.
        """
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours
        self.stop = stop
        self.location_locks = location_locks
# Note: The following ReusableBarrier implementation appears to be an external dependency or a draft
# included in the same file. It provides a classic two-phase barrier synchronization mechanism.
from threading import *

class ReusableBarrier():
    """
    Implements a reusable, two-phase synchronization barrier for a fixed number of threads.

    This barrier ensures that all participating threads wait at a synchronization point (`wait()` call)
    until every thread has reached it. Once all threads arrive, they are all released simultaneously.
    The barrier is "reusable" because it resets itself after each synchronization, allowing it to be
    used multiple times, for instance, in an iterative algorithm.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a specified number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        # Counters for each phase of the barrier. Using a list to make it mutable across method calls.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # Lock to protect access to the shared counters.
        self.count_lock = Lock()
        # Semaphores to block and release threads for each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)
    
    def wait(self):
        """
        Causes a thread to wait at the barrier. The thread will block until all `num_threads`
        have called this method. This is achieved through two internal phases to prevent
        race conditions where fast threads could loop around and re-enter the barrier
        before slow threads have exited the previous `wait()`.
        """
        self.phase(self.count_threads1, self.threads_sem1) # First synchronization phase
        self.phase(self.count_threads2, self.threads_sem2) # Second synchronization phase
    
    def phase(self, count_threads, threads_sem):
        """
        Executes a single phase of the barrier synchronization.

        Args:
            count_threads (list[int]): The counter for the current phase.
            threads_sem (Semaphore): The semaphore used to block threads in this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # Pre-condition: If this is the last thread to arrive...
            if count_threads[0] == 0:
                # ...release all waiting threads by signaling the semaphore `num_threads` times.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # Block the thread until it is released by the last arriving thread.
        threads_sem.acquire()
                                                 

class MyThread(Thread):
    """
    A simple example thread class to demonstrate the usage of `ReusableBarrier`.
    """
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
    
    def run(self):
        """
        The thread's execution logic, which synchronizes at the barrier in a loop.
        """
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",