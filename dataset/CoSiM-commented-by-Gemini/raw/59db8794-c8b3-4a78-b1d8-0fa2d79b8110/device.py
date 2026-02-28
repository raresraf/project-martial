# -*- coding: utf-8 -*-
"""
Models a distributed network of computational devices that collaboratively process sensor data.

This module defines a simulation framework where each `Device` operates concurrently,
executing assigned scripts on shared data locations. Synchronization is managed through
a combination of barriers, semaphores, and locks to ensure data consistency and
coordinated execution across the network.

Key Components:
- **Device**: Represents a node in the network, managing its own sensor data and scripts.
- **DeviceThread**: The main control loop for a `Device`, orchestrating script execution and synchronization.
- **WorkerThread**: A thread responsible for executing a single script, fetching data from neighbors,
  and updating the results.

The simulation uses a supervisor entity to manage network topology (neighbors) and a
reusable barrier to synchronize devices at each step of the computation.
"""

from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierSem
from collections import deque

class Device(object):
    """
    Represents a single computational device in the distributed network.

    Each device maintains its own sensor data, a list of scripts to execute,
    and communicates with a central supervisor. It uses a dedicated `DeviceThread`
    to manage its lifecycle and a pool of `WorkerThread` instances to execute scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local sensor readings,
                                keyed by location.
            supervisor: An object that manages the network topology and provides neighbor information.
        """
        self.device_id = device_id
        # Local data store for the device, keyed by location.
        self.sensor_data = sensor_data
        # A reference to the central supervisor for neighbor discovery.
        self.supervisor = supervisor
        # A list to hold scripts assigned to this device.
        self.scripts = []
        # Event to signal that the device's main thread has been set up.
        self.setup_done = Event() 
        # Semaphore to signal the arrival of a new script.
        self.script_semaphore = Semaphore(0) 
        # A list of tuples, each containing a location and its corresponding lock.
        self.location_locks = [] 
        # A queue to manage the sequence of script execution stops.
        self.queue = deque() 
        
        # The main thread that orchestrates the device's operations.
        self.thread = None
        # A reusable barrier for synchronizing all devices in the simulation.
        self.barrier = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def start_thread(self, barrier, locks):
        """
        Initializes and starts the main DeviceThread.

        This method sets up the synchronization primitives (barrier and locks)
        and starts the device's main execution thread.

        Args:
            barrier (ReusableBarrierSem): The barrier for synchronizing all devices.
            locks (list): A list of (location, Lock) tuples for all shared locations.
        """
        self.thread = DeviceThread(self)
        self.barrier = barrier

        # Assign the shared locks for data locations.
        self.location_locks = locks
        self.thread.start()
        # Signal that the setup is complete.
        self.setup_done.set() 

    def setup_devices(self, devices):
        """
        Coordinates the setup of all devices in the simulation.

        This method should only be called on one device (e.g., device_id 0), which acts
        as the master for setup. It creates the shared barrier and the set of locks
        for all data locations across all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Only the master device (device_id 0) should orchestrate the setup.
        if self.device_id == 0:
            # Create a barrier for all devices.
            barrier = ReusableBarrierSem(len(devices))
            
            # Create a unique lock for each sensor data location.
            locks = []
            for device in devices:
                for location in device.sensor_data:
                    # Check if a lock for this location already exists.
                    if location not in locks:
                        locks.append((location, Lock()))
            
            # Start the main thread for each device with the shared barrier and locks.
            for device in devices:
                device.start_thread(barrier, locks)

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        Args:
            script: The script object to be executed. If None, it signals a
                    synchronization point or end of a script batch.
            location: The data location the script operates on.
        """
        if script is not None:
            # Add the script and its target location to the scripts list.
            self.scripts.append((script, location))
        else:
            # A None script is a sentinel indicating a break point in execution.
            self.queue.append(len(self.scripts))
        
        # Signal the DeviceThread that a new script is available.
        self.script_semaphore.release()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location: The location for which to retrieve data.

        Returns:
            The data at the given location, or None if the location is not
            managed by this device.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        """

        Updates sensor data for a specific location.

        Args:
            location: The location for which to update data.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Gracefully shuts down the device's threads.
        """
        # Wait for the initial setup to complete before shutting down.
        self.setup_done.wait()
        self.thread.join()


class WorkerThread(Thread):
    """
    A thread that executes a single script on data from a specific location.

    Worker threads are part of a pool managed by a `DeviceThread`. They are
    responsible for acquiring locks, gathering data from the device and its
    neighbors, running the script, and broadcasting the result.
    """

    def __init__(self, device_thread):
        """
        Initializes a WorkerThread.

        Args:
            device_thread (DeviceThread): The parent DeviceThread that manages this worker.
        """
        Thread.__init__(self)
        self.device_thread = device_thread

    def run(self):
        """
        The main loop for the worker thread.

        It continuously waits for scripts to be placed on the queue. When a script
        is available, it processes it and waits for the next one. A None script
        serves as a sentinel to terminate the thread.
        """
        while True:
            # Wait for a script to be assigned by the DeviceThread.
            self.device_thread.threads_semaphore.acquire()
            
            script = None
            location = None
            # Pre-condition: Check if there are scripts in the queue to process.
            if len(self.device_thread.scripts_queue) > 0:
                (script, location) = self.device_thread.scripts_queue.popleft()
            
            # A None location is a sentinel for shutting down the worker thread.
            if location is None:
                break

            # Find and acquire the lock for the target data location.
            lock = next(l for (x, l) in self.device_thread.device.location_locks
                if x == location)

            lock.acquire()
            
            # Gather data from neighbors and the local device.
            script_data = []
            
            # Invariant: Collect data from all neighbors that have data for the location.
            for device in self.device_thread.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Collect data from the local device itself.
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Invariant: Only run the script if there is data to process.
            if script_data != []:
                # Execute the script with the collected data.
                result = script.run(script_data)

                # Broadcast the result back to all neighbors and the local device.
                for device in self.device_thread.neighbours:
                    device.set_data(location, result)
                
                self.device_thread.device.set_data(location, result)
            
            # Release the lock for the location.
            lock.release()
            
            # Signal the DeviceThread that this worker has completed its task.
            self.device_thread.worker_semaphore.release()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    This thread manages a pool of `WorkerThread`s, orchestrates synchronization
    with other devices using a barrier, and dispatches assigned scripts to its
    worker pool.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Semaphore to control the number of active worker threads.
        self.threads_semaphore = Semaphore(0)
        
        # A queue for scripts to be processed by worker threads.
        self.scripts_queue = deque() 
        # A pool of worker threads to execute scripts concurrently.
        self.worker_threads = [] 
        # A list of neighboring devices.
        self.neighbours = [] 
        # Semaphore to track the completion of worker tasks.
        self.worker_semaphore = Semaphore(0)
        
        self.nr_threads = 8 

        # Create and start the pool of worker threads.
        for _ in xrange(self.nr_threads):
            thread = WorkerThread(self)
            self.worker_threads.append(thread)
            thread.start()

    def run(self):
        """
        The main execution loop for the device.

        This loop coordinates the device's participation in the simulation's
        synchronization and computation steps.
        """
        
        index = 0
        while True:
            # Pre-condition: Wait for all previously dispatched worker tasks to complete.
            for _ in xrange(index):
                self.worker_semaphore.acquire()
            
            # Synchronize with all other devices at the barrier.
            self.device.barrier.wait()
            
            index = 0
            stop = None
            
            # Get the current list of neighbors from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()
            
            # If get_neighbours returns None, it's a signal to shut down.
            if self.neighbours is None:
                break

            # Invariant: Process all assigned scripts in a loop.
            while True:
                # If there are no more scripts in the local list, wait for a new one.
                if not len(self.device.scripts) > index:
                    self.device.script_semaphore.acquire()
                
                # Check for a stop signal from the queue.
                if stop is None:
                    if len(self.device.queue) > 0:
                        stop = self.device.queue.popleft()
                
                # If a stop index is reached, break the inner loop to re-synchronize.
                if stop is not None and stop == index:
                    break
                
                # If no stop signal and no more scripts, continue waiting.
                if stop is None and not len(self.device.scripts) > index:
                    continue

                # Get the next script to execute.
                (script, location) = self.device.scripts[index]
                
                # Add the script to the worker queue and signal a worker.
                self.scripts_queue.append((script, location))
                self.threads_semaphore.release() 
                
                index += 1
        
        # Shutdown sequence: signal all worker threads to terminate.
        for _ in xrange(len(self.worker_threads)):
            self.threads_semaphore.release()
        
        # Wait for all worker threads to join.
        for thread in self.worker_threads:
            thread.join()
