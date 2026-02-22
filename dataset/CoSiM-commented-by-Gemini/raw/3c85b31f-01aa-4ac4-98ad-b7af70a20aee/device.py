
"""
@brief A distributed device simulation with a master-worker setup and critical flaws.
@file device.py

This module implements a simulation of devices that process sensor data in
parallel time steps. This version uses a master-worker pattern for initialization
(Device 0 is the master) and a two-phase semaphore-based barrier for
synchronization. Unlike other versions, it does not use a worker thread pool;
each device's main thread performs all the work.

WARNING:
This implementation contains multiple severe concurrency flaws and bugs.
1.  **Critical Race Condition**: In the `DeviceThread.run` method, only the master
    thread (device 0) fetches the list of `neighbours`. All other threads use a
    stale or uninitialized local `neighbours` variable, leading to incorrect
    behavior and data corruption.
2.  **Flawed Barrier**: The `ReusableBarrier` implementation holds a lock while
    releasing waiting threads, which is an anti-pattern that can lead to
    serialization and deadlocks.
3.  **Potential Deadlock**: The locking pattern in `DeviceThread.run` for
    accessing `locations_locks` is prone to deadlock. It acquires a dictionary
    lock and then a location-specific lock before releasing the dictionary lock.
"""

from threading import Event, Thread, Lock
# This local import suggests a project structure that was not preserved.
# A flawed implementation is provided at the end of this file.
from utils import ReusableBarrier


class Device(object):
    """
    Represents a single device node in the simulation, managed by a master device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and its main control thread.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's local sensor data.
            supervisor (Supervisor): The central supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self, device_id)

        # The shared barrier for all devices.
        self.common_barrier = None
        # An event used by worker devices to wait for the master to finish setup.
        self.wait_initialization = Event()

        # A shared dictionary of locks, one for each data location.
        self.locations_locks = None
        # A lock to protect the creation of new locks in the locations_locks dict.
        self.lock_location_dict = Lock()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources using a master-worker pattern.

        Device 0 acts as the master, creating the shared barrier and lock
        dictionary. All other devices wait until the master is done.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Pre-condition: Check if this device is a worker or the master.
        if not self.device_id == 0:
            # Block for worker devices.
            # Wait for the master to signal that initialization is complete.
            self.wait_initialization.wait()
            # Start the main thread only after setup is complete.
            self.thread.start()
        else:
            # Block for the master device (device_id == 0).
            # The master creates the shared lock dictionary.
            self.locations_locks = {}

            # The master creates and distributes the shared barrier.
            barrier_size = len(devices)
            self.common_barrier = ReusableBarrier(len(devices))

            for dev in devices:
                dev.common_barrier = self.common_barrier
                dev.locations_locks = self.locations_locks
            
            # After setup, the master signals all worker devices to proceed.
            for dev in devices:
                if not dev.device_id == 0:
                    dev.wait_initialization.set()

            self.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to the device for the current time step.

        A `None` script signals the end of script assignment for the time step.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Invariant: Setting this event unblocks the DeviceThread to start work.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specific location. Not thread-safe.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates sensor data at a specific location. Not thread-safe.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its control thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, which performs all script execution.
    """

    def __init__(self, device, th_id):
        """Initializes the thread, storing its ID to check if it's the master."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.th_id = th_id

    def run(self):
        """
        The main, flawed, simulation loop for the device.
        """
        # Invariant: Loop continues as long as the supervisor is active.
        while True:
            # All threads synchronize at the start of each time step.
            self.device.common_barrier.wait()

            # CRITICAL FLAW: Only the master thread (th_id == 0) gets the
            # list of neighbours. Other threads will use a stale/uninitialized
            # `neighbours` local variable from a previous iteration, causing
            # data races and incorrect behavior.
            if self.th_id == 0:
                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    break # Master thread breaks, signaling simulation end.
            else:
                # Worker threads do nothing, relying on their incorrect local
                # `neighbours` variable.
                pass

            # Wait for the supervisor to finish assigning scripts for this step.
            self.device.timepoint_done.wait()

            current_scripts = self.device.scripts

            # Process all assigned scripts.
            for (script, location) in current_scripts:
                # DEADLOCK RISK: Acquires a global dictionary lock, then a
                # location-specific lock. This nested locking can cause deadlock.
                self.device.lock_location_dict.acquire()

                # Lazily initialize a lock for the location if not present.
                if not self.device.locations_locks.has_key(location):
                    self.device.locations_locks[location] = Lock()

                # Acquire the lock for the specific data location.
                self.device.locations_locks[location].acquire()
                self.device.lock_location_dict.release()

                script_data = []
                # Gathers data from neighbors (uses a flawed `neighbours` list).
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: Only run script if there is data.
                if script_data != []:
                    result = script.run(script_data)

                    # Propagate the result to itself and neighbors.
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the lock for the specific data location.
                self.device.locations_locks[location].release()

            self.device.timepoint_done.clear()


from threading import Semaphore, Lock


class ReusableBarrier(object):
    """
    An implementation of a reusable two-phase barrier using semaphores.

    WARNING: This implementation is flawed. It holds `count_lock` while releasing
    the semaphore, which is an anti-pattern that can serialize waiting threads
    and create deadlocks. A correct implementation would release the lock
    before signaling other threads.
    """

    def __init__(self, num_threads):
        """Initializes the barrier for a fixed number of threads."""
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads] # Use a list for pass-by-reference.
        self.count_threads2 = [self.num_threads]

        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0) # Gate for the first phase.
        self.threads_sem2 = Semaphore(0) # Gate for the second phase.

    def wait(self):
        """Blocks until all threads reach the barrier. Consists of two phases."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier synchronization.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # Pre-condition: Check if this is the last thread to arrive.
            if count_threads[0] == 0:
                # Block Logic: The last thread resets the counter and wakes up
                # all other threads by releasing the semaphore N times.
                # FLAW: The lock is held during these releases.
                for i in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        
        # All threads block here until the last thread has released them.
        threads_sem.acquire()
        
