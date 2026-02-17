# -*- coding: utf-8 -*-
"""Defines components for a simulated distributed sensor network.

This module contains the classes necessary to simulate a network of devices that
process sensor data in a synchronized, parallel manner. The simulation operates
in discrete time steps, synchronized by a reusable barrier.

- ReusableBarrier: A classic two-phase thread barrier for synchronization.
- Device: Represents a single device (node) in the network, managing its own
  data, state, and execution thread.
- DeviceThread: The worker thread for each Device, which executes the core data
  processing logic.
"""

from threading import Event, Thread, Lock
from utils import ReusableBarrier


class Device(object):
    """Represents a device node in the distributed sensor simulation.

    Each device holds its own sensor data, a list of scripts to execute, and
    manages a worker thread. One device (ID 0) acts as a coordinator to set up
    shared synchronization primitives.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor data,
            keyed by location.
        supervisor (object): A reference to a supervisor object that can provide
            information about the network topology (e.g., neighbors).
        scripts (list): A list of (script, location) tuples to be executed in
            the current time step.
        timepoint_done (Event): An event used to signal the worker thread to
            start processing the current time step's scripts.
        thread (DeviceThread): The worker thread that executes the device's logic.
        common_barrier (ReusableBarrier): A shared barrier to synchronize all
            devices at the start of each time step.
        wait_initialization (Event): An event used by the coordinator (device 0)
            to signal other devices that initialization is complete.
        locations_locks (dict): A shared dictionary mapping locations to Lock
            objects to ensure mutually exclusive access to data at each location.
        lock_location_dict (Lock): A lock to protect write access to the
            `locations_locks` dictionary itself.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self, 0)

        # To be initialized by the coordinator device (ID 0)
        self.common_barrier = None
        self.wait_initialization = Event()
        self.locations_locks = None
        
        # Lock to ensure atomic creation of location-specific locks
        self.lock_location_dict = Lock()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and starts the device threads.

        Device 0 acts as a coordinator. It creates and distributes the shared
        barrier and the dictionary for location-based locks to all other
        devices. Once shared resources are set up, it signals other devices
        to start their threads.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if not self.device_id == 0:
            # Worker devices wait for the coordinator to finish setup.
            self.wait_initialization.wait()
            self.thread.start()
        else:
            # Coordinator device (ID 0) initializes shared resources.
            self.locations_locks = {}
            self.common_barrier = ReusableBarrier(len(devices))

            # Distribute shared resources to all other devices.
            for dev in devices:
                dev.common_barrier = self.common_barrier
                dev.locations_locks = self.locations_locks
            # Signal worker devices that initialization is done.
            for dev in devices:
                if not dev.device_id == 0:
                    dev.wait_initialization.set()

            self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be run in the next time step.

        If the script is None, it signals that no more scripts are coming for
        this time step, and the worker thread can start processing.

        Args:
            script (object): The script object to be executed. Must have a `run` method.
            location (any): The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is the signal to start processing for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location.

        Args:
            location (any): The key for the desired sensor data.

        Returns:
            The sensor data if the location exists, otherwise None.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates the sensor data for a given location.

        Args:
            location (any): The key for the sensor data to update.
            data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's worker thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The worker thread responsible for executing a device's logic.

    This thread runs in a continuous loop, synchronized with other devices at
    each time step. It processes assigned scripts, interacts with neighboring
    devices, and updates sensor data.
    """

    def __init__(self, device, th_id):
        """Initializes the device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.th_id = th_id

    def run(self):
        """The main execution loop for the device thread."""
        while True:
            # Phase 1: Synchronize all threads at the start of a time step.
            self.device.common_barrier.wait()

            # The coordinator thread fetches neighbor info for all devices.
            if self.th_id == 0:
                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    # Supervisor signaling simulation end.
                    break
            else:
                # Other threads rely on the coordinator for this info.
                pass

            # Phase 2: Wait for the signal that scripts for this timepoint are assigned.
            self.device.timepoint_done.wait()

            current_scripts = self.device.scripts

            # Phase 3: Execute all assigned scripts for this time step.
            for (script, location) in current_scripts:
                # --- Critical Section for Location Data ---
                # Ensure the lock for this location exists.
                self.device.lock_location_dict.acquire()
                if not self.device.locations_locks.has_key(location):
                    self.device.locations_locks[location] = Lock()
                
                # Acquire the lock specific to the data's location.
                self.device.locations_locks[location].acquire()
                self.device.lock_location_dict.release()

                script_data = []
                # Gather data from all neighbors for the given location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather this device's own data for the location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Run the script with the aggregated data.
                    result = script.run(script_data)

                    # Disseminate the result to all neighbors and self.
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the location-specific lock.
                self.device.locations_locks[location].release()
                # --- End of Critical Section ---

            # Phase 4: Reset for the next time step.
            self.device.timepoint_done.clear()

from threading import Semaphore, Lock


class ReusableBarrier(object):
    """A reusable barrier implementation for synchronizing multiple threads.

    This barrier uses a two-phase mechanism to ensure that no thread can start
    a new `wait()` cycle until all threads have completed the previous one,
    preventing race conditions.
    """

    def __init__(self, num_threads):
        """Initializes the barrier for a fixed number of threads.

        Args:
            num_threads (int): The number of threads that will synchronize on
                this barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]

        # Lock to protect access to the internal thread counters.
        self.count_lock = Lock()

        # Semaphores to block threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to block until all threads have called this method.

        This is a two-phase wait. Threads are blocked once in `phase`, released,
        and then blocked a second time in the next `phase` call. This ensures
        all threads have left the first phase before any can re-enter it.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the counter for this phase.
            threads_sem (Semaphore): The semaphore used for blocking in this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all other waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        
        # All threads wait here until the last thread releases the semaphore.
        threads_sem.acquire()
