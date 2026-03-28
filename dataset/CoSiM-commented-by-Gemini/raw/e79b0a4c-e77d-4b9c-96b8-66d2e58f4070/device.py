"""
Defines components for a simulated concurrent sensor network, including a
`ReusableBarrier` implementation.

This module contains the `Device` and `DeviceThread` classes that model nodes in a
sensor network. It also includes a `ReusableBarrier` class for thread
synchronization, which is used by the devices.
"""

from threading import Event, Thread, Lock, Semaphore

class Device(object):
    """
    Represents a single device (node) in the sensor network simulation.

    Each device operates with a single worker thread and is responsible for
    processing scripts, managing its own sensor data, and synchronizing with
    other devices using a shared barrier.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor data.
        supervisor: A reference to the supervisor managing the simulation.
        scripts (list): A list of (script, location) tuples to be processed.
        timepoint_done (Event): An event to signal that a timepoint's script
                                 assignments are complete.
        data_lock (Lock): A lock to protect this device's sensor_data during
                          updates from other devices.
        barrier (ReusableBarrier): A shared barrier to synchronize all devices
                                   at the end of a timepoint.
        location_locks (dict): A shared dictionary of locks, one for each
                               sensor data location, ensuring mutual exclusion.
        thread (DeviceThread): The single worker thread for this device.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance with a single worker thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.data_lock = Lock()
        self.barrier = None
        self.location_locks = {}
        self.thread = DeviceThread(self)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device and shared simulation resources, then starts the worker thread.

        If this is the master device (device_id == 0), it creates the shared
        `ReusableBarrier` and `location_locks` dictionary and distributes them
        to all other devices.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
            for device in devices:
                if device.device_id != self.device_id:
                    device.barrier = self.barrier
                    device.location_locks = self.location_locks
        self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed or signals the end of a timepoint.

        Args:
            script: The script object to execute. If None, it signals that
                    all scripts for the current timepoint have been assigned.
            location (int): The data location the script applies to.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates the sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its worker thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The single worker thread for a Device, responsible for all processing.
    """
    def __init__(self, device):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the worker thread.

        In each cycle, the thread waits for a timepoint to end, processes all
        assigned scripts by collaborating with neighbors, and then synchronizes
        with all other devices at a global barrier.
        """
        while True:
            # Fetches the current list of neighbors from the supervisor for this timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            # The shutdown signal is a None value for neighbors.
            if neighbours is None:
                break

            # Wait for the supervisor to signal that all scripts for this timepoint are assigned.
            self.device.timepoint_done.wait()
            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()

            for (script, location) in self.device.scripts:
                # Lazily initialize a lock for a location if not seen before.
                if location not in self.device.location_locks:
                    self.device.location_locks[location] = Lock()
                # Acquire the lock for the location to ensure exclusive processing.
                self.device.location_locks[location].acquire()
                
                script_data = []
                # Gather data from all neighbors.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                # Gather data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execute script and update data on all involved devices.
                if script_data:
                    result = script.run(script_data)
                    # Update neighbors with the new result, locking each one.
                    for device in neighbours:
                        device.data_lock.acquire()
                        device.set_data(location, result)
                        device.data_lock.release()
                    # Update the local device, ensuring its own lock is used.
                    self.device.data_lock.acquire()
                    self.device.set_data(location, result)
                    self.device.data_lock.release()

                # Release the lock for the location.
                self.device.location_locks[location].release()

            # Wait at the global barrier for all devices to finish their timepoint processing.
            self.device.barrier.wait()

class ReusableBarrier():
    """
    A reusable barrier implementation for a fixed number of threads.

    This barrier uses a two-phase protocol with semaphores to ensure that all
    threads wait at the barrier until the last one arrives, after which all are
    released and the barrier can be used again.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier. Consists of two phases."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier wait.

        The last thread to enter the phase releases all other waiting threads.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread arrived, release all threads waiting on the semaphore.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset count for the next use of this phase.
                count_threads[0] = self.num_threads
        # All other threads wait here until released by the last thread.
        threads_sem.acquire()

class MyThread(Thread):
    """
    A simple example thread class used for testing the ReusableBarrier.
    """
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier

    def run(self):
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "
",
