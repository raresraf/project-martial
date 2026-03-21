"""
Models a network of communicating devices that execute computational scripts
in synchronized, discrete time steps.

This module defines a `Device` that acts as a node in a distributed system,
managed by a `supervisor`. Each device has its own sensor data and a pool of
worker threads to execute tasks. Synchronization across all devices is handled
by a shared barrier, ensuring that all devices complete a time step before
proceeding to the next.
"""

from threading import Event, Thread, Lock, Condition, RLock
from Queue import Queue
from barrier import ReusableBarrierCond


class Device(object):
    """
    Represents a single device (node) in a simulated distributed network.

    Each device manages its own sensor data, a collection of scripts to be
    executed, and a pool of worker threads. It coordinates with other devices
    using shared synchronization primitives.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding the initial sensor data,
                                keyed by location.
            supervisor: An external entity responsible for managing the
                        network topology and assigning scripts.
        """
        self.sync_barrier = None
        self.acquire_stage_lock = None
        self.device_init_event = Event()

        # A dictionary of re-entrant locks, one for each data location.
        self.location_data_lock = {location:RLock() for location in sensor_data}

        self.device_id = device_id
        self.supervisor = supervisor
        self.sensor_data = sensor_data

        self.timepoint_ended = False
        self.script_condition = Condition()
        self.scripts = []

        self.thread = DeviceThread(0, self)
        # Initializes a fixed-size pool of 8 worker threads.
        self.worker_pool = [DeviceWorker(i, self) for i in xrange(1, 9)]

        self.neighbours = []
        self.work_queue = Queue()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization primitives.

        Device with device_id 0 acts as the coordinator, creating the shared
        barrier and lock. Other devices wait for the coordinator to finish
        initialization and then receive a reference to these shared objects.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            # Device 0 is the coordinator and creates the shared synchronization objects.
            self.acquire_stage_lock = Lock()
            self.sync_barrier = ReusableBarrierCond(len(devices))
        else:
            # All other devices wait for Device 0 to initialize the sync objects.
            for device in devices:
                if device.device_id == 0:
                    device.device_init_event.wait()
                    self.sync_barrier = device.sync_barrier
                    self.acquire_stage_lock = device.acquire_stage_lock
        # Signals that this device's synchronization primitives are ready.
        self.device_init_event.set()
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed for a specific location.

        This method is typically called by the supervisor to queue tasks for the
        current time step. It blocks if the current timepoint has already ended.

        Args:
            script: The script object to be executed.
            location: The data location the script will operate on.
        """
        with self.script_condition:
            while self.timepoint_ended:
                self.script_condition.wait()
            if script is not None:
                self.scripts.append((script, location))
            else:
                # A None script is a signal that the current timepoint has ended.
                self.timepoint_ended = True
                self.script_condition.notify_all()

    def get_data(self, location):
        """
        Provides thread-safe read access to sensor data for a given location.

        Args:
            location: The location from which to retrieve data.

        Returns:
            The data at the specified location, or None if the location
            does not exist.
        """
        if location in self.sensor_data:
            with self.location_data_lock[location]:
                return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Provides thread-safe write access to sensor data for a given location.

        Args:
            location: The location at which to set data.
            data: The new data value to be set.
        """
        if location in self.sensor_data:
            with self.location_data_lock[location]:
                self.sensor_data[location] = data

    def acquire_location(self, location):
        """
        Acquires the lock for a specific data location.

        Args:
            location: The location to lock.

        Returns:
            True if the lock was acquired successfully, False otherwise.
        """
        if location in self.location_data_lock:
            self.location_data_lock[location].acquire()
            return True
        return False

    def release_location(self, location):
        """
        Releases the lock for a specific data location.

        Args:
            location: The location lock to release.
        """
        if location in self.location_data_lock:
            try:
                self.location_data_lock[location].release()
            except RuntimeError:
                # This error can occur if a thread tries to release a lock it
                # doesn't own. We ignore it to prevent crashing, though it may
                # indicate a logic error in lock management.
                pass

    def shutdown(self):
        """Shuts down the device by joining its main control thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a Device.

    This thread orchestrates the device's lifecycle, managing the execution of
    scripts within synchronized time steps and handling the shutdown of
    worker threads.
    """

    def __init__(self, thread_id, device):
        """
        Initializes the main device thread.

        Args:
            thread_id (int): The identifier for this thread.
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % thread_id)
        self.thread_id = thread_id
        self.device = device

    def stop_device(self):
        """
        Gracefully stops all worker threads in the device's pool.

        This is achieved by sending a "poison pill" (None) to the work queue
        for each worker.
        """
        for _ in xrange(len(self.device.worker_pool)):
            self.device.work_queue.put(None)
        self.device.work_queue.join()
        for thread in self.device.worker_pool:
            thread.join()

    def run(self):
        """
        The main simulation loop for the device.

        The loop proceeds in synchronized time steps. In each step, it:
        1. Retrieves its current neighbors from the supervisor.
        2. Waits until the supervisor signals that all scripts for the current
           time step have been assigned.
        3. Puts all assigned scripts into the work queue for the workers.
        4. Waits for the workers to complete all tasks for the time step.
        5. Resets the timepoint state.
        6. Waits at a global barrier, synchronizing with all other devices
           before starting the next time step.
        """
        for thread in self.device.worker_pool:
            thread.start()

        while True:
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                # If the supervisor provides no neighbors, it's a signal to shut down.
                self.stop_device()
                break

            with self.device.script_condition:
                while not self.device.timepoint_ended:
                    self.device.script_condition.wait()

                for script in self.device.scripts:
                    self.device.work_queue.put(script)

                self.device.work_queue.join()

                self.device.timepoint_ended = False
                self.device.script_condition.notify_all()

            # Synchronize with all other devices before proceeding to the next time step.
            self.device.sync_barrier.wait()


class DeviceWorker(DeviceThread):
    """
    A worker thread that executes computational scripts.

    Workers fetch tasks from a shared work queue and perform a distributed
    read-compute-write cycle, coordinating with neighboring devices to access
    and update shared data.
    """

    def __init__(self, thread_id, device):
        """
        Initializes a worker thread.

        Args:
            thread_id (int): The identifier for this worker thread.
            device (Device): The parent device this worker belongs to.
        """
        super(DeviceWorker, self).__init__(thread_id, device)

    def run(self):
        """
        The main loop for the worker thread.

        It continuously fetches scripts from the work queue and executes them.
        The execution of a script involves:
        1. Acquiring a global lock to ensure atomic acquisition of location locks.
        2. Acquiring the specific location lock on its own device and its neighbors.
        3. Gathering data from all devices where the lock was successfully acquired.
        4. Running the script with the collected data.
        5. Writing the result back to all devices that were part of the computation.
        6. Releasing all acquired locks.
        The loop terminates when it receives a "poison pill" (None).
        """
        while True:
            item = self.device.work_queue.get()
            if item is None:
                # "Poison pill" received, task is done.
                self.device.work_queue.task_done()
                break
            (script, location) = item

            acquired_devices = []
            script_data = []

            # Use a global lock to serialize the lock acquisition phase across all workers.
            with self.device.acquire_stage_lock:
                if self.device.acquire_location(location):
                    acquired_devices.append(self.device)
                for device in self.device.neighbours:
                    if device.device_id != self.device.device_id:
                        if device.acquire_location(location):
                            acquired_devices.append(device)

            # Read phase: Collect data from all successfully locked devices.
            for device in acquired_devices:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Compute and Write phase.
            if len(script_data) > 0:
                result = script.run(script_data)

                for device in acquired_devices:
                    device.set_data(location, result)

            # Release phase: Release all acquired locks.
            for device in acquired_devices:
                device.release_location(location)

            self.device.work_queue.task_done()
