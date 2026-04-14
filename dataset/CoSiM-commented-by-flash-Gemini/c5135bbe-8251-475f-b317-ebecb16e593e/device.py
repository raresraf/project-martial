"""
@file device.py
@brief Implements a simulated distributed device system with asynchronous script execution and data synchronization.

This module defines the core components for simulating a network of devices.
It features asynchronous script processing via a worker thread pool and a queue,
and utilizes a conditional reusable barrier for inter-device synchronization.
Data access is protected using explicit locks for each sensor data location.
"""

from threading import Event, Thread, Lock
from Queue import Queue
from barrier import ReusableBarrierCond


class Device(object):
    """
    @class Device
    @brief Represents an individual device in the simulated distributed system.

    Each device manages its sensor data, receives and queues scripts for execution,
    and interacts with a supervisor to get information about its neighbors.
    It uses explicit locks for data locations to ensure thread-safe access and
    a conditional reusable barrier for overall system synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary holding the device's sensor data (location -> value).
        @param supervisor The supervisor object responsible for managing device interactions.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Invariant: Event to signal when a new script has been received and queued.
        self.script_received = Event()
        # Invariant: List to store (script, location) tuples assigned to this device.
        self.scripts = []
        # Invariant: Event to signal that all scripts for the current timepoint have been assigned.
        self.timepoint_done = Event()

        # Invariant: Reference to the shared ReusableBarrierCond for synchronization.
        self.barrier = None
        # Invariant: Dictionary of locks, protecting access to each sensor data location.
        self.locks = {}

        # Invariant: The dedicated thread responsible for this device's operational logic.
        self.thread = DeviceThread(self)
        self.thread.start()


    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs initial setup for the device, including creating data locks and
        (for device 0) initializing the shared reusable barrier for all devices.
        @param devices A list of all Device objects in the simulation.
        """
        # Block Logic: Initializes a Lock for each sensor data location to ensure thread-safe access.
        for loc in self.sensor_data:
            self.locks[loc] = Lock()

        # Block Logic: Device with ID 0 is responsible for creating the global barrier
        # and distributing it to all other devices.
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            # Block Logic: Propagates the newly created barrier to all other devices.
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script and its target location to the device for future execution.
        If `script` is not None, it's added to the internal list and `script_received` is set.
        If `script` is None, it signals that all scripts for the current timepoint are assigned,
        and `timepoint_done` is set.
        @param script The script object to execute, or `None` to signal timepoint completion.
        @param location The data location relevant to the script execution.
        """
        # Block Logic: Conditional handling based on whether a script is provided or a timepoint is being signaled.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location in a thread-safe manner.
        Acquires a lock for the location before returning data.
        @param location The key for the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        # Block Logic: Ensures thread-safe access to sensor data by acquiring a lock.
        if location in self.sensor_data:
            self.locks[location].acquire()
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location and releases its lock.
        @param location The key for the sensor data.
        @param data The new data to set.
        """
        # Block Logic: Updates sensor data and releases the corresponding lock, typically after `get_data` was called.
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locks[location].release()

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its dedicated operational thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief The dedicated operational thread for a Device object, managing its worker pool.

    This thread orchestrates the continuous operation of a device, including:
    1. Fetching neighbor information from the supervisor.
    2. Managing a pool of worker threads (`self.work`) via a `Queue` for script execution.
    3. Coordinating script assignment and timepoint completion.
    4. Synchronizing with other DeviceThreads using a `ReusableBarrierCond`.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device object this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Invariant: List to hold references to the worker threads in the pool.
        self.threads = []
        # Invariant: Queue to hold tasks (scripts, locations, neighbors) for worker threads.
        self.queue = Queue()
        self.create_threads()

    def create_threads(self):
        """
        @brief Creates and starts a fixed pool of worker threads.

        Block Logic: Initializes 8 worker threads, each targeting the `self.work` method,
        and starts them to begin processing tasks from the queue.
        """
        for _ in xrange(8):
            thread = Thread(target=self.work)
            self.threads.append(thread)

        for thread in self.threads:
            thread.start()

    def join_threads(self):
        """
        @brief Gracefully shuts down and joins all worker threads in the pool.

        Block Logic: Puts `None` sentinels into the queue to signal worker threads to exit,
        waits for all queued tasks to be marked as done, and then joins all worker threads.
        """
        # Block Logic: Sends termination signals (None, None, None) to each worker thread.
        for _ in xrange(8):
            self.queue.put((None, None, None))

        # Block Logic: Waits until all tasks (including termination signals) in the queue are processed.
        self.queue.join()

        # Block Logic: Joins each worker thread, ensuring they have completed their execution.
        for thread in self.threads:
            thread.join()

    def work(self):
        """
        @brief The target function for worker threads.

        Block Logic: Continuously retrieves tasks (script, location, neighbors) from the queue,
        executes the script on collected data, updates device and neighbor data,
        and then marks the task as done. Exits if a `None` sentinel is received.
        """
        while True:
            script, location, neighbours = self.queue.get()

            # Pre-condition: Checks for the `None` sentinel to signal thread termination.
            if script is None:
                self.queue.task_done()
                break

            script_data = []

            # Block Logic: Collects data from neighboring devices for the specified location.
            for device in neighbours:
                # Pre-condition: Ensures data is not collected from itself if it's in the neighbors list.
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

            # Block Logic: Collects data from its own device for the specified location.
            data = self.device.get_data(location)

            if data is not None:
                script_data.append(data)

            # Pre-condition: If `script_data` is not empty, there is data to process.
            if script_data != []:
                # Block Logic: Executes the script with the collected data.
                result = script.run(script_data)

                # Block Logic: Updates the data on neighboring devices (excluding itself) with the script's result.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)

                # Block Logic: Updates the data on its own device with the script's result.
                self.device.set_data(location, result)

            # Inline: Marks the current task as complete in the queue.
            self.queue.task_done()

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Invariant: The loop continues as long as the supervisor provides neighbor information,
        processing timepoints, queuing scripts, and synchronizing.
        """
        while True:
            # Block Logic: Retrieves current neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Pre-condition: If `neighbours` is None, it signifies a shutdown or termination condition.
            if neighbours is None:
                break

            # Block Logic: Inner loop for processing a single timepoint, handling script assignment and completion.
            while True:
                # Pre-condition: Checks if new scripts have been received for this timepoint.
                if self.device.script_received.isSet():
                    self.device.script_received.clear() # Inline: Clears the event after processing assigned scripts.
                    # Block Logic: Queues each assigned script as a task for the worker thread pool.
                    for (script, location) in self.device.scripts:
                        self.queue.put((script, location, neighbours))

                # Pre-condition: Checks if the current timepoint has been explicitly marked as done.
                if self.device.timepoint_done.isSet():
                    self.device.timepoint_done.clear() # Inline: Clears the event for the next timepoint.
                    # Inline: Resets `script_received` for the next timepoint's script assignments.
                    self.device.script_received.set()
                    break # Block Logic: Exits the inner loop as the timepoint processing is complete.

            # Block Logic: Waits until all tasks (scripts) for the current timepoint have been processed by workers.
            self.queue.join()

            # Block Logic: Synchronizes with all other DeviceThreads using the conditional reusable barrier,
            # ensuring all devices complete their timepoint before proceeding to the next.
            self.device.barrier.wait()

        # Block Logic: After the main loop terminates, gracefully shuts down all worker threads.
        self.join_threads()
