"""
@bc55c351-a352-4523-84fc-3130172e0755/device.py
@brief Distributed sensor simulation with thread-level parallelism and global barrier synchronization.
* Algorithm: Cooperative multi-threaded execution loop with shared state validation and location-based mutual exclusion.
* Functional Utility: Manages a pool of worker threads that concurrently process sensor scripts while ensuring consistent global state.
"""

from threading import Event, Thread, Semaphore
from barrier import ReusableBarrierSem
from Queue import Queue
from copy import deepcopy
from time import sleep
from random import random

class Device(object):
    """
    @brief Encapsulates a sensor device with its own data, worker threads, and synchronization logic.
    """
    
    # Domain: Resource scaling - Defines the number of concurrent execution contexts per device.
    NR_THREADS = 8

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes the device and starts its worker thread pool.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = []
        self.script_received = Event()
        self.scripts = []
        self.script_queue = Queue()
        self.loc_dev_semaphore = Semaphore(value=1)
        self.loc_info = {}
        self.script_reset = False
        self.semaphore_devices = Semaphore()
        self.update_semaphore = Semaphore(value=1)
        self.current_script_state = {}
        self.has_neighbours = False
        self.neighbours_semaphore = Semaphore(value=1)
        self.neighbours = []
        self.barrier_devices = ReusableBarrierSem(0)
        self.barrier_threads = ReusableBarrierSem(Device.NR_THREADS)
        self.queue_semaphore = Semaphore(value=1)
        self.queue_init_semaphore = Semaphore(value=1)
        self.threads = []

        # Block Logic: Bootstraps the thread pool for parallel task execution.
        for count in range(0, Device.NR_THREADS):
            self.threads.append(DeviceThread(self))
            self.threads[count].start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures global synchronization across all devices in the simulation.
        Logic: Establishes a common barrier and semaphore to coordinate collective actions.
        """
        # Architectural Intent: Unified synchronization across all threads of all devices.
        same_barrier = ReusableBarrierSem(len(devices) * Device.NR_THREADS)
        semaphore_devices = Semaphore(value=1)

        self.devices = devices

        # Logic: Propagates shared synchronization primitives to all participating device instances.
        for device in devices:
            device.barrier_devices = same_barrier
            device.semaphore_devices = semaphore_devices

    def assign_script(self, script, location):
        """
        @brief Injects a script into the simulation at a specific location.
        Functional Utility: Populates script queues and informs peer devices about active locations.
        """
        if script is not None:
            self.scripts.append((script, location))

            # Logic: Synchronizes location awareness across the entire device cluster.
            for device in self.devices:
                device.add_location(location)
        else:
            # Logic: Termination or phase-completion signal.
            self.script_received.set()

    def add_location(self, location):
        """
        @brief Tracks a new data location in the local registry.
        Invariant: Uses update_semaphore to prevent concurrent modification of loc_info.
        """
        self.update_semaphore.acquire()

        if location in self.loc_info:
            self.loc_info[location] = False
        else:
            self.loc_info.update({location : False})

        self.update_semaphore.release()

    def check_location(self, location):
        """
        @brief Validates if a location is currently eligible for processing by a script.
        Logic: Implements a distributed "compare-and-swap" style check to ensure exclusive access.
        """
        self.semaphore_devices.acquire()

        res = False

        # Logic: If the location is free, mark it as occupied across all devices.
        if self.current_script_state[location] == False:
            res = True
            for device in self.devices:
                device.current_script_state[location] = True

        self.semaphore_devices.release()

        return res

    def free_location(self, location):
        """
        @brief Releases the occupation status of a location.
        """
        self.semaphore_devices.acquire()
        for device in self.devices:
            device.current_script_state[location] = False

        self.semaphore_devices.release()

    def get_data(self, location):
        """
        @brief Standard getter for local sensor data.
        """
        result = None

        if location in self.sensor_data:
            result = self.sensor_data[location]

        return result

    def set_data(self, location, data):
        """
        @brief Standard setter for local sensor data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def get_current_neighbours(self):
        """
        @brief Lazy-loads the neighbor list from the supervisor.
        Invariant: neighbors_semaphore ensures thread-safe access to the has_neighbours flag.
        """
        self.neighbours_semaphore.acquire()
        if self.has_neighbours == False:
            self.neighbours = self.supervisor.get_neighbours()
            self.has_neighbours = True

        self.neighbours_semaphore.release()

        return self.neighbours

    def reset_neigbours(self):
        """
        @brief Invalidates neighbor cache to trigger a refresh in the next timepoint.
        """
        self.has_neighbours = False

    def init_queue(self):
        """
        @brief Initializes the script queue for the current timepoint.
        Logic: Transfers assigned scripts into a work queue for consumption by worker threads.
        """
        self.queue_init_semaphore.acquire()
        if self.script_queue.empty() == True:
            for (script, location) in self.scripts:
                self.script_queue.put((script, location))

        self.queue_init_semaphore.release()

    def again(self):
        """
        @brief Resets the script reset flag for the next simulation cycle.
        """
        self.script_reset = False

    def reset_script_state(self):
        """
        @brief Restores the initial "free" state for all locations at the start of a timepoint.
        Logic: Performs a deep copy of the static location info into the active state map.
        """
        self.loc_dev_semaphore.acquire()

        if self.script_reset == False:
            self.current_script_state = deepcopy(self.loc_info)
            self.script_reset = True

        self.loc_dev_semaphore.release()

    def shutdown(self):
        """
        @brief Waits for all worker threads to terminate.
        """
        for count in range(0, Device.NR_THREADS):
            self.threads[count].join()

class DeviceThread(Thread):
    """
    @brief Worker thread implementation that executes sensor scripts in a loop.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief Main execution lifecycle of a worker thread.
        Algorithm: Iterative task processing with multi-stage barrier synchronization.
        """
        while True:
            # Logic: Neighbor discovery and simulation exit condition.
            neighbours = self.device.get_current_neighbours()
            if neighbours is None:
                break

            # Block Logic: Synchronization phase - wait for scripts and initialize local state.
            self.device.script_received.wait()
            self.device.init_queue()
            self.device.barrier_devices.wait()
            self.device.reset_script_state()
            self.device.barrier_devices.wait()
            self.device.script_received.clear()

            # Task Execution Phase: Process all scripts in the shared queue.
            while True:
                self.device.queue_semaphore.acquire()

                if self.device.script_queue.empty():
                    self.device.queue_semaphore.release()
                    break

                (script, location) = self.device.script_queue.get()

                # Logic: Check availability of the target location.
                if self.device.check_location(location) == False:
                    # Logic: Location occupied; re-queue and introduce backoff if queue is small.
                    last_script = self.device.script_queue.empty()
                    self.device.script_queue.put((script, location))
                    self.device.queue_semaphore.release()

                    if last_script:
                        # Optimization: Prevents busy-waiting on a single remaining occupied location.
                        sleep(random() * 0.3)
                    continue

                self.device.queue_semaphore.release()

                # Distributed Data Processing: Aggregates data from neighbors and local node.
                script_data = []
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Functional Utility: Runs the core script logic and propagates state changes.
                if script_data != []:
                    result = script.run(script_data)

                    for device in neighbours:
                        device.set_data(location, result)

                    self.device.set_data(location, result)

                # Post-condition: Mark location as available for other threads/scripts.
                self.device.free_location(location)

            # Synchronization Phase: Ensure all threads/devices complete task execution.
            self.device.barrier_devices.wait()
            self.device.reset_neigbours()
            self.device.again()
            self.device.barrier_threads.wait()
            self.device.barrier_devices.wait()
