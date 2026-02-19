"""
@file device.py
@brief A simulation of a distributed device where each device is itself multi-threaded.
@details This module defines a device simulation where each `Device` instance creates and manages
its own pool of worker threads (`DeviceThread`). It uses a two-tiered synchronization system:
an internal barrier for its own threads and a global barrier for synchronizing with all other devices.
"""

from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    @brief A reusable barrier implementation using two semaphores for thread synchronization.
    @details This barrier synchronizes a fixed number of threads over two phases, ensuring that
    all threads reach the barrier before any are allowed to proceed. The two-phase (turnstile)
    design prevents race conditions where faster threads might loop and re-enter the barrier
    before slower threads have left the previous wait, making it safely reusable.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        # Counters for threads arriving at each phase.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        # Semaphores for each synchronization phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Causes a thread to wait at the barrier until all participating threads have arrived.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief Handles the first phase of barrier synchronization (arrival).
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Handles the second phase of barrier synchronization (reset).
        @details This ensures all threads have passed phase 1 before the barrier can be reused.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    """
    @brief Represents a single, multi-threaded device in the simulation.
    @details Each device has a fixed pool of 8 worker threads that share the workload.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.timepoint_done = Event()
        self.scripts = []

        # An internal barrier for the device's own 8 worker threads.
        self.barrier_worker = ReusableBarrier(8)
        self.setup_event = Event()
        self.devices = []
        self.locks = None
        self.neighbours = []
        # The global barrier for synchronizing all threads across all devices.
        self.barrier = None
        self.threads = []

        # Create and start the device's internal pool of worker threads.
        for i in range(8):
            self.threads.append(DeviceThread(self, i))
        for thr in self.threads:
            thr.start()

        # A list of locks, one for each data location, shared across all devices.
        self.location_lock = []

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (global barrier and shared lock list) for all devices.
        @details The device with ID 0 is responsible for creating these shared resources.
        """
        if self.device_id == 0:
            # Create the global barrier for all worker threads in the system.
            barrier = ReusableBarrier(len(devices)*8)
            self.barrier = barrier
            location_max = 0
            # Distribute the barrier and determine the required size for the shared lock list.
            for device in devices:
                device.barrier = barrier
                for location, data in device.sensor_data.iteritems():
                    if location > location_max:
                        location_max = location
                device.setup_event.set()
            self.setup_event.set()

            # Initialize the shared lock list.
            self.location_lock = [None] * (location_max + 1)
            
            # Distribute the shared lock list to all devices.
            for device in devices:
                device.location_lock = self.location_lock
                device.setup_event.set()
            self.setup_event.set()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device and handles lazy initialization of locks.
        @note The lock initialization logic is complex and potentially racy. It attempts to
        find a lock from other devices before creating a new one, which is not a standard
        or safe pattern for concurrent lazy initialization.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Lazy, complex initialization of the lock for the given location.
            if self.location_lock[location] is None:
                busy = 0
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        busy = 1
                        break
                if busy == 0:
                    self.location_lock[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """@brief Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """@brief Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """@brief Shuts down the device by joining all of its worker threads."""
        for thr in self.threads:
            thr.join()


class DeviceThread(Thread):
    """
    @brief A worker thread belonging to a Device's internal thread pool.
    """

    def __init__(self, device, idd):
        """
        @param device The parent Device object.
        @param idd The ID of this thread within the device's pool (0-7).
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.idd = idd

    def run(self):
        """@brief The main execution loop for the worker thread."""
        # Wait for the initial setup to complete before starting.
        self.device.setup_event.wait()

        while True:
            # Block Logic: The first worker thread (idd=0) is responsible for fetching the neighbor list.
            # All worker threads then synchronize using the internal barrier to ensure the list is visible to all.
            if self.idd == 0:
                neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours = neighbours
            self.device.barrier_worker.wait()

            # A None value for neighbors is the shutdown signal.
            if self.device.neighbours is None:
                break

            # Wait until all scripts for the timepoint are received and the 'timepoint_done' signal is sent.
            self.device.timepoint_done.wait()
            self.device.barrier_worker.wait() # Sync to ensure all threads see the done signal.
            
            i = 0
            # Block Logic: Distribute script execution among the worker threads.
            # Each thread processes a portion of the scripts based on its ID.
            for (script, location) in self.device.scripts:
                if i % 8 == self.idd:
                    # Use a 'with' statement for safe, automatic lock acquisition and release.
                    with self.device.location_lock[location]:
                        script_data = []
                        # Gather data from neighbors.
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            if data is not None:
                                script_data.append(data)
                        
                        # Gather data from the local device.
                        data = self.device.get_data(location)
                        if data is not None:
                            script_data.append(data)

                        if script_data != []:
                            result = script.run(script_data)

                            # Broadcast the result to neighbors and the local device.
                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            self.device.set_data(location, result)
                i = i + 1

            # After processing, clear the timepoint event and wait at the global barrier for all devices to sync.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
