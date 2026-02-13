"""
This module provides another simulation of a distributed system of devices, using a
location-based parallelization strategy for script execution.

Key components:
- ReusableBarrier: A custom two-phase barrier implementation for synchronizing threads.
- Device: Represents a node in the system, managing its data and scripts.
- ParallelScript: A thread class designed to execute all scripts associated with a
  single location for a given timepoint.
- DeviceThread: The main control loop for a device, which groups scripts by location
  and spawns ParallelScript threads to execute them.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    A custom implementation of a reusable barrier for thread synchronization.

    This barrier uses a two-phase protocol with semaphores to ensure that all
    threads wait at the barrier before any of them are released, and that the
    barrier can be reused immediately afterward.
    """
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The number of threads that will synchronize on this barrier.
        """
        self.num_threads = num_threads
        # Counters for each phase of the barrier.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # A lock to protect access to the counters.
        self.count_lock = Lock()
        # Semaphores for signaling in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        # The barrier consists of two phases to prevent race conditions on reuse.
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
        # All threads, including the last one, wait on the semaphore.
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device in the distributed simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): The device's local sensor data.
            supervisor (Supervisor): The central controller of the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # The shared synchronization barrier.
        self.barrier = None
        # A list of locks, one for each unique location in the simulation.
        self.big_lock = []

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (barrier, locks) to all devices.

        This method should be called once on any device to set up the entire system.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        barrier = ReusableBarrier(len(devices))
        lock1 = Lock() # This lock seems unused in the current implementation.

        # Determine the total number of unique locations across all devices.
        num_locations = {}
        for device in devices:
            for location in device.sensor_data.keys():
                num_locations[location] = 1

        # Create a lock for each unique location.
        big_lock = [Lock() for _ in range(len(num_locations))]

        # Distribute the shared barrier and locks to all devices.
        for device in devices:
            device.lock1 = lock1
            device.barrier = barrier
            device.big_lock = big_lock

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A 'None' script signals the end of script assignment for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] \
            if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the main device thread."""
        self.thread.join()


class ParallelScript(Thread):
    """
    A thread to execute all scripts associated with a single location.

    This approach parallelizes work by location, ensuring that all computations
    for a specific location happen within one thread, protected by a single lock.
    """
    def __init__(self, device, scripts, location, neighbours):
        """
        Initializes the ParallelScript thread.

        Args:
            device (Device): The parent device.
            scripts (list): The list of scripts to execute for the location.
            location (str): The location context for this thread.
            neighbours (list): The list of neighboring devices.
        """
        Thread.__init__(self)
        self.device = device
        self.scripts = scripts
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        The main execution logic for the thread.

        It acquires a lock for the location, processes all scripts, and then
        releases the lock.
        """
        # Invariant: The loop processes each script assigned to this location.
        for script in self.scripts:
            # Acquire the specific lock for this location to ensure data consistency.
            self.device.big_lock[self.location].acquire()

            # --- Critical Section for the location ---
            script_data = []
            
            # Gather data from neighbors.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from the local device.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                result = script.run(script_data)

                # Broadcast the result to all neighbors and the local device.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)
            
            # Release the lock for the location.
            self.device.big_lock[self.location].release()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    It manages the device's lifecycle, grouping scripts by location and spawning
    ParallelScript threads to execute them.
    """
    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop of the device thread."""
        # Invariant: The loop continues as long as the supervisor provides neighbors.
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to signal that all scripts for the timepoint are assigned.
            self.device.timepoint_done.wait()

            threads = []
            scripts = {}

            # Group all assigned scripts by their location.
            for (script, location) in self.device.scripts:
                if scripts.has_key(location):
                    scripts[location].append(script)
                else:
                    scripts[location] = [script]

            # Create and start a ParallelScript thread for each location that has scripts.
            for location in scripts.keys():
                new = ParallelScript(self.device, scripts[location],
                                     location, neighbours)
                threads.append(new)

            for thread in threads:
                thread.start()

            # Wait for all location-specific threads to complete.
            for thread in threads:
                thread.join()

            # Synchronize with all other devices at the end of the timepoint.
            self.device.barrier.wait()

            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()
