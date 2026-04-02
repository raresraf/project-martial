"""
This module implements a simulation of a distributed system of devices.

The simulation operates in synchronized time-steps, orchestrated by a custom
reusable barrier. Each device runs a main control thread that, at each
timepoint, spawns multiple worker threads (DataScript) to execute
computational tasks. Concurrency of these worker threads is managed by a
semaphore. Data access across devices is synchronized using location-based
locks. This architecture differs from a traditional fixed-thread-pool model by
dynamically creating threads for each task within a time-step.
"""

from threading import Event, Thread
from threading import Lock, Semaphore

class ReusableBarrier():
    """
    A custom, reusable barrier implementation for thread synchronization.

    This barrier ensures that all participating threads wait for each other at a
    synchronization point before any of them are allowed to continue. It uses a
    two-phase protocol with two semaphores to prevent threads from one iteration
    (or "wave") from proceeding before all threads from the previous iteration
    have passed the barrier, thus avoiding race conditions.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The number of threads that will synchronize on this barrier.
        """
        self.num_threads = num_threads
        # Counters for each phase, stored in a list to be mutable across instances.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        # Semaphores to block and release threads for each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        # The barrier consists of two distinct synchronization phases.
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier synchronization.

        Args:
            count_threads (list): The counter for the current phase.
            threads_sem (Semaphore): The semaphore for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # The last thread to arrive resets the counter and releases all waiting threads.
            if count_threads[0] == 0:
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads
        # All threads, including the last one, block on this semaphore until released.
        threads_sem.acquire()


class Device(object):
    """
    Represents a device node in the simulated distributed system.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The local sensor data for this device.
            supervisor (object): The central supervisor object managing the simulation.
        """
        self.done_setup = Event()
        self.device_id = device_id
        self.thread = DeviceThread(self)
        self.thread.start()
        self.script_received = Event()
        self.sensor_data = sensor_data
        # A semaphore to limit the number of concurrently running DataScript threads to 8.
        self.semaphore = Semaphore(value=8)

        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()

        self.nr_thread = 0
        self.lock_timepoint = Lock()
        self.script_list = []
        # A list of locks, one for each data location, shared among all devices.
        self.lock_index = []

        self.r_barrier = None

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared synchronization objects for all devices.

        This method is intended to be called on one device (device 0), which then
        creates and distributes the shared barrier and location locks to all
        other devices in the simulation.
        """
        used_devices = len(devices)
        if self.device_id is 0:
            r_barrier = ReusableBarrier(used_devices)
            # Assumes a fixed number of 24 locations for the locks.
            for _ in range(0, 24):
                self.lock_index.append(Lock())

            # Distribute the shared objects to all devices.
            for d in range(len(devices)):
                devices[d].lock_index = self.lock_index
                devices[d].r_barrier = r_barrier
                devices[d].done_setup.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific location.

        Args:
            script (object): The script to execute.
            location (any): The data location for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def close_scripts(self):
        """
        Waits for all spawned DataScript threads for the current timepoint to complete.
        """
        nrThreads = len(self.script_list)
        for i in range(0, nrThreads):
            self.script_list[i].join()

        # Clears the list of thread objects.
        for i in range(0, nrThreads):
            self.script_list.pop()

        self.nr_thread = 0

    def shutdown(self):
        """Waits for the main device thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating the simulation steps.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run_script(self, neighbours):
        """
        Spawns DataScript threads for each assigned script.
        
        For each script, it acquires a semaphore slot and starts a new thread
        to execute the script.
        """
        for (script, location) in self.device.scripts:
            self.device.semaphore.acquire()
            self.device.script_list.append(DataScript
            (neighbours, location, script, self.device))

            self.device.nr_thread = self.device.nr_thread + 1
            self.device.script_list[len(self.device.script_list)-1].start()

    def run(self):
        """The main execution loop of the device."""
        
        # Wait until the initial setup of shared objects is complete.
        self.device.done_setup.wait()

        while True:
            
            # Note: The `with ... as` syntax with a Lock is incorrect.
            # It seems the intent was to acquire a lock, but the assignment
            # to 'neighbours' is not a standard behavior of a Lock.
            # This is documented as-is, per the original code.
            with self.device.lock_timepoint as neighbours:

                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    break # End of simulation.

            # Wait for the signal to start the timepoint.
            self.device.timepoint_done.wait()
            self.run_script(neighbours)

            # First barrier: ensures all devices have launched their script threads.
            self.device.r_barrier.wait()
            self.device.timepoint_done.clear()
            # Wait for all local script threads to finish their work.
            self.device.close_scripts()
            # Second barrier: ensures all devices have completed all work for the
            # timepoint before starting the next one.
            self.device.r_barrier.wait()


class DataScript(Thread):
    """
    A thread to execute a single script on data from a specific location.
    """
    def __init__(self, neighbours, location, script, scr_device):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.script = script
        self.location = location
        self.scr_device = scr_device


    def getdata(self, script_data):
        """Appends the local device's data to the script_data list."""
        data = self.scr_device.get_data(self.location)
        if data is not None:
            script_data.append(data)

    def scriptdata(self, script_data):
        """
        Runs the script on the collected data and disseminates the results.
        Releases the semaphore once done.
        """
        if script_data != []:
            
            result = self.script.run(script_data)
            
            # Update the data on all neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)

            # Update the data on the local device.
            self.scr_device.set_data(self.location, result)
        # Release the semaphore slot, allowing another script to run.
        self.scr_device.semaphore.release()

    def run(self):
        """The main logic for the script execution thread."""
        # Acquire the lock for the specific location to ensure exclusive access.
        with self.scr_device.lock_index[self.location]:
            script_data = []

            # Gather data from all neighboring devices.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            self.getdata(script_data)
            self.scriptdata(script_data)
