"""
This module provides a simulation of a networked device, including a custom
implementation of a reusable barrier using semaphores.

The design features a `Device` class that manages its state and a `DeviceThread`
that controls its execution logic. For each time step, the `DeviceThread` spawns
a new thread for every assigned script, a model that differs from fixed-pool or
sequential execution strategies. This implementation also contains several critical
concurrency issues, such as non-thread-safe data access and race conditions in
resource initialization.
"""

from threading import Lock, Thread, Event, Semaphore


class Device(object):
    """
    Represents a single device in the simulation.

    This class holds the device's state, including its sensor data and assigned
    scripts. It relies on a master `DeviceThread` to manage its operations.

    NOTE: The `get_data` and `set_data` methods are not thread-safe, which can lead
    to data corruption in a multi-threaded context.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device instance.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): The device's local sensor data.
            supervisor (object): The central simulation coordinator.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.location_locks = None
        
    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources for all devices.

        This method relies on device 0 to act as a coordinator to create and
        distribute the shared barrier and lock dictionary.

        NOTE: This setup has a potential race condition. If worker devices (ID != 0)
        execute this method before device 0 has finished initializing the shared
        resources, they will fail.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        Device.devices_no = len(devices)
        if self.device_id == 0:
            # Coordinator device creates the shared synchronization objects.
            self.barrier = ReusableBarrierSem(len(devices))
            self.location_locks = {}
        else:
            # Worker devices get a reference to the shared objects from the coordinator.
            self.barrier = devices[0].barrier
            self.location_locks = devices[0].location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed in the next time step.

        Args:
            script (object): The script to execute. If None, it signals the end of the timepoint.
            location: The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data for a given location.

        WARNING: This method is not thread-safe. Concurrent calls from multiple
        script threads could lead to unpredictable behavior.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets data for a given location.

        WARNING: This method is not thread-safe. Concurrent calls from multiple
        script threads could lead to data corruption.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a device.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        
    def run_scripts(self, script, location, neighbours):
        """
        The target function for executing a single script in its own thread.

        It manages locking for a specific location, gathers data, runs the script,
        and propagates the results.

        Args:
            script (object): The script to execute.
            location: The location context for the script.
            neighbours (list): A list of neighboring devices.
        """
        # NOTE: This lock creation logic has a race condition. If two threads
        # simultaneously check for a new location, both could try to create a lock,
        # with one overwriting the other.
        lock_location = self.device.location_locks.get(location)
        if lock_location is None and location is not None:
            self.device.location_locks[location] = Lock()
            lock_location = self.device.location_locks[location]
        
        lock_location.acquire()
        
        # Block Logic: Gathers data from the local device and its neighbors.
        script_data = []
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
            
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        # Invariant: The script is only run if there is data to process.
        if script_data != []:
            result = script.run(script_data)
            # Block Logic: Propagate the result back to the local device and its neighbors.
            for device in neighbours:
                device.set_data(location, result)
            self.device.set_data(location, result)
            
        lock_location.release()

    def run(self):
        """
        The main simulation loop for the device.
        """
        while True:
            # Get the current network topology from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value signals the end of the simulation.
                break
            
            # Wait for the supervisor to signal that scripts for the timepoint are ready.
            self.device.timepoint_done.wait()
            
            # Block Logic: Spawns a new thread for each assigned script.
            tlist = []
            for (script, location) in self.device.scripts:
                thread = Thread(target=self.run_scripts, args=(script, location, neighbours))
                tlist.append(thread)
                thread.start()
            
            # Wait for all script threads for the current timepoint to complete.
            for thread in tlist:
                thread.join()
            
            # Reset for the next timepoint and synchronize with other devices.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()


class ReusableBarrierSem():
    """
    A custom implementation of a reusable barrier using semaphores.

    This barrier synchronizes threads in two phases to ensure that threads from
    one synchronization cycle do not prematurely enter the next one.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a fixed number of threads.

        Args:
            num_threads (int): The number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        # Counters for each phase, protected by a single lock.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        
        # Semaphores to block threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread to arrive releases all threads waiting in phase 1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset counter for the next use of the barrier.
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # The last thread to arrive releases all threads waiting in phase 2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset counter for the next use of the barrier.
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()