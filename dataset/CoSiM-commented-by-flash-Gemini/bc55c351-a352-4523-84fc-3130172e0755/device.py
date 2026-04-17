"""
@bc55c351-a352-4523-84fc-3130172e0755/device.py
@brief Concurrent distributed sensor network simulation with multi-threaded local nodes.
This module implements a sophisticated synchronization model where each device node 
manages a pool of internal threads. Threads across the entire network coordinate via 
global barriers and semaphores to execute data processing scripts while ensuring 
mutual exclusion for specific sensor locations.

Domain: Parallel Systems, Distributed Synchronization, High-Concurrency Simulations.
"""

from threading import Event, Thread, Semaphore
from barrier import ReusableBarrierSem
from Queue import Queue
from copy import deepcopy
from time import sleep
from random import random

class Device(object):
    """
    Coordinator class for a multi-threaded network node.
    Functional Utility: Manages internal worker threads, synchronization primitives, 
    and shared state across the local device group.
    """

    # Constant: Defines the degree of local parallelism.
    NR_THREADS = 8

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device with local data and spawns worker threads.
        @param device_id: Unique integer identifier.
        @param sensor_data: Initial state of sensors managed by this device.
        @param supervisor: Entity providing connectivity topology.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = []
        self.script_received = Event()
        self.scripts = []
        self.script_queue = Queue()
        # Semaphore for atomic access to local script reset state.
        self.loc_dev_semaphore = Semaphore(value=1)
        self.loc_info = {}
        self.script_reset = False
        # Global semaphore shared across devices for cross-node location locking.
        self.semaphore_devices = Semaphore()
        self.update_semaphore = Semaphore(value=1)
        self.current_script_state = {}
        self.has_neighbours = False
        self.neighbours_semaphore = Semaphore(value=1)
        self.neighbours = []
        self.barrier_devices = ReusableBarrierSem(0)
        # Local barrier to synchronize threads within this specific device.
        self.barrier_threads = ReusableBarrierSem(Device.NR_THREADS)
        self.queue_semaphore = Semaphore(value=1)
        self.queue_init_semaphore = Semaphore(value=1)
        self.threads = []

        # Functional Utility: Initializes the local worker pool.
        for count in range(0, Device.NR_THREADS):
            self.threads.append(DeviceThread(self))
            self.threads[count].start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Global initialization of synchronization resources.
        Logic: Distributes a shared barrier and semaphore to all devices in the group 
        to enable network-wide atomic operations.
        """
        # Barrier sized to accommodate all threads from all devices in the network.
        same_barrier = ReusableBarrierSem(len(devices) * Device.NR_THREADS)
        semaphore_devices = Semaphore(value=1)

        self.devices = devices

        for device in devices:
            device.barrier_devices = same_barrier
            device.semaphore_devices = semaphore_devices

    def assign_script(self, script, location):
        """
        Registers a new computational task and propagates the target location metadata.
        @param script: The computational logic to execute.
        @param location: The sensor identifier affected by the script.
        """
        if script is not None:
            self.scripts.append((script, location))

            # Block Logic: Ensures all devices are aware of the new location identifier.
            for device in self.devices:
                device.add_location(location)
        else:
            # Signals that no more scripts are pending for the current timepoint.
            self.script_received.set()

    def add_location(self, location):
        """Thread-safe registration of a sensor location into the local metadata store."""
        self.update_semaphore.acquire()
        if location in self.loc_info:
            self.loc_info[location] = False
        else:
            self.loc_info.update({location : False})
        self.update_semaphore.release()

    def check_location(self, location):
        """
        Implements a distributed mutex for a sensor location.
        Precondition: Must be called within a cross-device synchronized context.
        Functional Utility: Atomic 'Test-and-Set' operation across all devices.
        @return: True if the location was successfully locked, False otherwise.
        """
        self.semaphore_devices.acquire()
        res = False

        if self.current_script_state[location] == False:
            res = True
            # Invariant: Marks the location as busy across ALL devices in the network.
            for device in self.devices:
                device.current_script_state[location] = True

        self.semaphore_devices.release()
        return res

    def free_location(self, location):
        """Releases the distributed lock for a specific location."""
        self.semaphore_devices.acquire()
        for device in self.devices:
            device.current_script_state[location] = False
        self.semaphore_devices.release()

    def get_data(self, location):
        """Accesses sensor data value for the given location."""
        result = None
        if location in self.sensor_data:
            result = self.sensor_data[location]
        return result

    def set_data(self, location, data):
        """Updates the sensor data value for the given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def get_current_neighbours(self):
        """Retrieves and caches the neighborhood topology from the supervisor."""
        self.neighbours_semaphore.acquire()
        if self.has_neighbours == False:
            self.neighbours = self.supervisor.get_neighbours()
            self.has_neighbours = True
        self.neighbours_semaphore.release()
        return self.neighbours

    def reset_neigbours(self):
        """Invalidates the neighborhood cache for the next processing cycle."""
        self.has_neighbours = False

    def init_queue(self):
        """Populates the script execution queue from the pending script list."""
        self.queue_init_semaphore.acquire()
        if self.script_queue.empty() == True:
            for (script, location) in self.scripts:
                self.script_queue.put((script, location))
        self.queue_init_semaphore.release()

    def again(self):
        """Resets the script processing flag for the next cycle."""
        self.script_reset = False

    def reset_script_state(self):
        """
        Resets the distributed lock states for the current processing phase.
        Logic: Performs a deep copy of the location metadata to refresh the lock map.
        """
        self.loc_dev_semaphore.acquire()
        if self.script_reset == False:
            self.current_script_state = deepcopy(self.loc_info)
            self.script_reset = True
        self.loc_dev_semaphore.release()

    def shutdown(self):
        """Terminates all worker threads managed by this device."""
        for count in range(0, Device.NR_THREADS):
            self.threads[count].join()

class DeviceThread(Thread):
    """
    Worker thread implementation.
    Functional Utility: Executes scripts from a shared queue using a spin-lock 
    pattern with backoff for distributed resource contention.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for the worker thread.
        Logic: Coordinates with peer threads and peer devices via multiple barrier points.
        """
        while True:
            neighbours = self.device.get_current_neighbours()
            if neighbours is None:
                break

            # Barrier Point 1: Wait for supervisor to assign scripts.
            self.device.script_received.wait()

            # Barrier Point 2: Initialize task queue.
            self.device.init_queue()

            # Global Synchronization: Ensure all threads in the network have reached this point.
            self.device.barrier_devices.wait()

            # Barrier Point 3: Prepare distributed state for processing.
            self.device.reset_script_state()

            self.device.barrier_devices.wait()
            self.device.script_received.clear()

            # Block Logic: Task Processing Loop.
            # Threads consume tasks from the shared queue until it is empty.
            while True:
                self.device.queue_semaphore.acquire()

                if self.device.script_queue.empty():
                    self.device.queue_semaphore.release()
                    break

                (script, location) = self.device.script_queue.get()

                # Functional Utility: Concurrent locking with retry.
                # If a location is already being processed, the task is re-queued.
                if self.device.check_location(location) == False:
                    last_script = self.device.script_queue.empty()
                    self.device.script_queue.put((script, location))
                    self.device.queue_semaphore.release()

                    # Inline: Prevents high-frequency spinning on contention 
                    # by introducing a random delay if the task is likely the only one left.
                    if last_script:
                        sleep(random() * 0.3)
                    continue

                self.device.queue_semaphore.release()

                # Execution Logic: Neighborhood data aggregation.
                script_data = []
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execute the computation and propagate the results.
                if script_data != []:
                    result = script.run(script_data)
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the distributed lock for this location.
                self.device.free_location(location)

            # Global Synchronization: Consensus point after all tasks are processed.
            self.device.barrier_devices.wait()

            # Post-processing cleanup.
            self.device.reset_neigbours()
            self.device.again()

            # Local Synchronization: Ensure all local threads finish the cycle together.
            self.device.barrier_threads.wait()

            # Final Network-wide synchronization before the next simulation step.
            self.device.barrier_devices.wait()
