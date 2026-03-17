"""
This module provides a simulation framework for a network of devices.

This implementation is distinct in that it does not use a persistent thread pool.
Instead, for each time-step, the main `DeviceThread` spawns a new `ScriptThread`
for every assigned task, waits for them all to complete (a fork-join model),
and then synchronizes with other devices.

The locking mechanism is centralized but relies on mapping a data location to an
index in a shared list of locks, which can be fragile.
"""

from threading import Event, Thread, Lock, Semaphore


class ReusableBarrier(object):
    """A reusable barrier for synchronizing a fixed number of threads.

    This implementation uses a two-phase protocol with two semaphores to ensure
    that threads from one "wave" do not overlap with threads from the next.
    """
    
    def __init__(self, num_threads):
        """
        Args:
            num_threads (int): The number of threads that will synchronize on this barrier.
        """
        self.num_threads = num_threads
        # Counters are stored in a list to be mutable across method calls.
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
        """Executes one phase of the barrier synchronization.

        Args:
            count_threads (list): A list containing the counter for the current phase.
            threads_sem (Semaphore): The semaphore for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all other waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        threads_sem.acquire()


class Device(object):
    """Represents a single device node in the simulation."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main lifecycle thread.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): The local data held by this device.
            supervisor (object): The central supervisor for network information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()
        self.location_locks = None

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared resources across all devices.

        One device (id 0) creates a shared barrier and a list of locks. The locks
        are indexed based on the discovery order of data locations, which can be
        a fragile mapping. These resources are then assigned to all devices.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        if 0 == self.device_id:
            self.barrier = ReusableBarrier(len(devices))
            
            locations = []
            for device in devices:
                for location in device.sensor_data:
                    if location not in locations:
                        locations.append(location)
            
            # Creates a list of locks, one for each unique location.
            self.location_locks = []
            for _ in range(len(locations)):
                self.location_locks.append(Lock())
            
            # Distributes the shared resources to all devices.
            for device in devices:
                device.barrier = self.barrier
                device.location_locks = self.location_locks

    def assign_script(self, script, location):
        """Assigns a script to the device for the current time-step.

        Args:
            script (object): The script object to be run.
            location (any): The data location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of a time-step's script assignments.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific location on this device."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates data at a specific location on this device."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class ScriptThread(Thread):
    """A short-lived thread created to execute a single script task."""

    def __init__(self, device, script, location, neighbours):
        """Initializes the thread with all necessary context for the task.

        Args:
            device (Device): The parent device.
            script (object): The script to execute.
            location (any): The data location to operate on.
            neighbours (list): A list of neighboring devices.
        """
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """Executes a single script.

        The thread acquires a shared lock for the data location, gathers data from
        itself and neighbors, runs the script, and propagates the result. The thread
        terminates after this single execution.
        """
        with self.device.location_locks[self.location]:
            script_data = []
            
            # Block Logic: Gathers data from neighboring devices.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Gathers data from the local device.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = self.script.run(script_data)
                
                # Block Logic: Propagates the result to all neighbors and the local device.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)


class DeviceThread(Thread):
    """The main control loop for a device, managing simulation time-steps."""

    def __init__(self, device):
        """
        Args:
            device (Device): The parent device this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop, executed once per time-step.

        This loop uses a fork-join model. It waits for a signal to begin the
        time-step, then creates and starts a new thread for each script. It
        waits for all these threads to complete before synchronizing at the
        network-wide barrier.
        """
        while True:
            vecini = self.device.supervisor.get_neighbours()
            if vecini is None:
                # End of simulation.
                break
            
            # Invariant: Waits for the supervisor to signal all scripts are assigned.
            self.device.timepoint_done.wait()
            threads = []
            
            # Functional Utility: Spawns and starts a new thread for each script task.
            if len(vecini) != 0:
                for (script, locatie) in self.device.scripts:
                    thread = ScriptThread(self.device, script, locatie, vecini)
                    threads.append(thread)
                    thread.start()
                
                # Invariant: Waits for all spawned script threads to complete.
                for thread in threads:
                    thread.join()
            
            self.device.timepoint_done.clear()
            
            # Invariant: All devices must synchronize at the barrier before the next step.
            self.device.barrier.wait()
