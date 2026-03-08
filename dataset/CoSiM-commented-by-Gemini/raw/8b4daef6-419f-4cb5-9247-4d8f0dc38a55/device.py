# -*- coding: utf-8 -*-
"""
A multi-threaded simulation of a distributed device network.

This script models a system of interconnected devices that operate in synchronized
time steps. Each device can run computational scripts on its own data and the data
of its neighbors. Synchronization is managed by a custom reusable barrier, ensuring
that all devices complete a time step before any can begin the next.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier():
    """
    A custom, reusable cyclic barrier for synchronizing a fixed number of threads.

    This implementation allows a group of threads to wait for each other to reach a
    common execution point. Once all threads have called wait(), they are all
    released and the barrier resets for the next use. It uses a two-phase
    protocol with two semaphores to prevent race conditions where fast threads
    could start the next cycle before slow threads have left the first one.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a fixed number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        # Counters for each phase. Using a list is a way to have a mutable
        # integer that can be passed by reference.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        # Semaphores to block and release threads for each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to block until all `num_threads` have called this method.
        """
        # First synchronization phase
        self.phase(self.count_threads1, self.threads_sem1)
        # Second synchronization phase ensures all threads from phase 1 have
        # been released before the barrier is reused.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Manages one of the two synchronization phases.

        Args:
            count_threads (list): The mutable counter for the current phase.
            threads_sem (Semaphore): The semaphore for blocking/releasing threads.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # The last thread to arrive resets the counter and releases all others.
            if count_threads[0] == 0:
                # Release all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # All threads will block here until the last thread releases the semaphore `num_threads` times.
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device (or node) in the distributed system.

    Each device has an ID, local sensor data, and runs a dedicated control
    thread (`DeviceThread`) to manage its operations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data,
                                mapping locations to values.
            supervisor (Supervisor): The supervisor object that manages the network topology.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Not used in this implementation.
        self.scripts = []
        self.devices = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barrier = None
        self.list_thread = []
        self.thread.start()
        # A list of locks, one for each potential data location.
        self.location_lock = [None] * 100

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Connects this device to its peers and establishes a shared barrier.

        This method should be called once to initialize the communication and
        synchronization fabric for the entire device group.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # The first device to be set up creates the shared barrier.
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            # Propagate the shared barrier to all other devices.
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        Assigns a computational script to be run at a specific location.

        This method is called by an external controller to give work to the device.
        A 'None' script is a special signal that a time step has concluded.

        Args:
            script: The script object to be executed.
            location (int): The data location the script will operate on.
        """
        ok = 0
        if script is not None:
            self.scripts.append((script, location))
            # Block-Logic: This complex block ensures all devices use the *same* lock
            # instance for the same location, enabling correct data serialization.
            if self.location_lock[location] is None:
                for device in self.devices:
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        ok = 1
                        break
                if ok == 0:
                    self.location_lock[location] = Lock()
            self.script_received.set()
        else:
            # A 'None' script signals the end of the script-assignment phase
            # for a time step, waking up the DeviceThread to begin computation.
            self.timepoint_done.set()

    def get_data(self, location):
        """Safely retrieves data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Safely updates data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to finish."""
        self.thread.join()


class NewThread(Thread):
    """
    A worker thread created to execute a single script for one time step.
    """
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script.

        The logic is to:
        1. Acquire a lock for the target data location to ensure exclusive access.
        2. Gather data from itself and its neighbors at that location.
        3. Run the script on the gathered data.
        4. Write the result back to itself and its neighbors.
        5. Release the lock.
        """
        script_data = []
        self.device.location_lock[self.location].acquire()
        
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = self.script.run(script_data)
            for device in self.neighbours:
                device.set_data(self.location, result)
            self.device.set_data(self.location, result)

        self.device.location_lock[self.location].release()


class DeviceThread(Thread):
    """
    The main control thread for a single Device.
    """

    def __init__(self, device):
        """
        Initializes the control thread.

        Args:
            device (Device): The device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop for the device.

        This loop represents the device's life cycle, progressing in synchronized
        time steps. It waits for a signal, performs its computations for the
        time step, and then synchronizes at a barrier before the next step.
        """
        while True:
            # Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If the supervisor returns None, the simulation is over.
            if neighbours is None:
                break

            # Wait for the external controller to signal that all scripts for
            # this time step have been assigned.
            self.device.timepoint_done.wait()

            # Spawn a worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                thread = NewThread(self.device, location, script, neighbours)
                self.device.list_thread.append(thread)

            # Start and wait for all worker threads for this time step to complete.
            for thread_elem in self.device.list_thread:
                thread_elem.start()
            for thread_elem in self.device.list_thread:
                thread_elem.join()
            self.device.list_thread = []

            # Reset for the next time step.
            self.device.timepoint_done.clear()
            # Synchronize with all other devices. This is a blocking call.
            self.device.barrier.wait()
