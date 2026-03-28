"""
Defines a two-level threaded model for a simulated concurrent sensor network.

This module contains a `Device` class which uses a `DeviceThread` to manage a
pool of `Worker` threads. This creates a hierarchical concurrency model where
a device's main thread orchestrates sub-threads that perform the actual data
processing. The file also includes a `ReusableBarrier` implementation for
synchronization.
"""

from threading import Event, Thread, Lock, Semaphore


class Device(object):
    """
    Represents a node in the sensor network, managing a `DeviceThread`.

    This class holds the device's state (ID, data) and its main thread, but
    delegates all concurrent operations and work management to the `DeviceThread`.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor data.
        supervisor: A reference to the supervisor managing the simulation.
        scripts (list): A list of (script, location) tuples to be processed.
        script_received (Event): Signals that script assignments have begun.
        timepoint_done (Event): Signals that all scripts for a timepoint are assigned.
        thread (DeviceThread): The main manager thread for this device.
        global_barrier (ReusableBarrier): A barrier to synchronize all devices.
        locks (list): A shared list of locks for all data locations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device and starts its main manager thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.global_barrier = None
        self.locks = None


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up shared resources for the simulation.

        If this is the master device (device_id == 0), it creates and distributes
        the global barrier and the shared location locks to all other devices.
        """
        if self.device_id == 0:
            # Create a barrier to synchronize all DeviceThreads.
            self.global_barrier = ReusableBarrier(len(devices))
            for device in devices:
                device.global_barrier = self.global_barrier

            # Aggregate all unique sensor locations to create a lock for each.
            self.locks = []
            locations = list(devices[0].sensor_data.keys())
            for index in range(1, len(devices)):
                aux = devices[index].sensor_data.keys()
                locations = list(set(locations).union(aux))

            for _ in range(len(locations)):
                self.locks.append(Lock())

            # Share the list of locks with all devices.
            for device in devices:
                device.locks = self.locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed or signals the end of a timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that scripts are ready to be processed.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates the sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    A manager thread that orchestrates a pool of `Worker` threads for a device.

    This thread does not perform script computations itself. Instead, it distributes
    the assigned scripts to its worker threads and manages the multi-level
    synchronization between workers, and with other devices.
    """

    def __init__(self, device):
        """Initializes the manager thread and its pool of 8 worker threads."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # This barrier synchronizes the 8 workers with this manager thread.
        self.barrier_parent = ReusableBarrier(9) # 8 workers + 1 manager
        self.threads = []
        for _ in range(8):
            self.threads.append(Worker(self.device, None, None, self.barrier_parent))
        for thread in self.threads:
            thread.start()

    def run(self):
        """
        Main loop for the manager thread.

        Coordinates script distribution and synchronization for each timepoint.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Shutdown signal
                break

            # Wait for supervisor to confirm scripts for the timepoint are assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Distribute scripts among the 8 worker threads.
            # Note: This logic appears to have a bug where only the last script in
            # a chunk is assigned if len(scripts) > 8. Documenting intended behavior.
            if len(self.device.scripts) <= 8:
                for index in range(len(self.device.scripts)):
                    self.threads[index].script = self.device.scripts[index]
                    self.threads[index].neighbours = neighbours
            else:
                aux = len(self.device.scripts) / 8
                inf = 0
                sup = aux
                for index in range(8):
                    if index == 7:
                        sup = len(self.device.scripts)
                    self.threads[index].neighbours = neighbours
                    for index2 in range(inf, sup):
                        self.threads[index].script = self.device.scripts[index2]
                    inf += aux
                    sup += aux

            # Sync 1: Wait for all workers to complete their script execution.
            self.barrier_parent.wait()
            # Wait for final supervisor signal for the timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Sync 2: A second synchronization point with workers before the global barrier.
            self.barrier_parent.wait()
            # Sync 3: Wait at the global barrier for all other devices to finish the timepoint.
            self.device.global_barrier.wait()

        # Shutdown sequence for worker threads.
        for thread in self.threads:
            thread.out = 1
        self.barrier_parent.wait() # Final sync to let workers see the 'out' flag.
        for thread in self.threads:
            thread.join()

class ReusableBarrier(object):
    """
    A reusable barrier implementation for a fixed number of threads.
    Uses a two-phase protocol with semaphores to prevent race conditions on reuse.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the calling thread until all threads have called wait()."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Executes one phase of the barrier synchronization."""
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0: # Last thread to arrive
                for _ in range(self.num_threads):
                    threads_sem.release() # Release all waiting threads
                count_threads[0] = self.num_threads # Reset for next use
        threads_sem.acquire() # All threads wait here until released.

class Worker(Thread):
    """
    A worker thread that executes data processing scripts.
    """
    def __init__(self, device, script, neighbours, barrier_parent):
        Thread.__init__(self, name="Worker Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.neighbours = neighbours
        self.workers_barrier = barrier_parent
        self.out = 0 # Shutdown flag

    def run(self):
        """
        Main loop for the worker. Waits for a job, executes it, and synchronizes.
        """
        while True:
            # Sync 1: Wait for the manager to assign work.
            self.workers_barrier.wait()

            if self.out == 1: # Check for shutdown signal from manager.
                break

            if self.neighbours is not None:
                script_data = []
                # Acquire the lock for the specific data location.
                self.device.locks[self.script[1]].acquire()

                # Gather data from neighbors.
                for device in self.neighbours:
                    data = device.get_data(self.script[1])
                    if data is not None:
                        script_data.append(data)
                # Gather data from the parent device.
                data = self.device.get_data(self.script[1])
                if data is not None:
                    script_data.append(data)

                # Execute script and update data if data was found.
                if script_data:
                    result = self.script[0].run(script_data)
                    for device in self.neighbours:
                        device.set_data(self.script[1], result)
                    self.device.set_data(self.script[1], result)
                
                # Release the location lock.
                self.device.locks[self.script[1]].release()
            
            # Sync 2: Signal to the manager that work is complete.
            self.workers_barrier.wait()
