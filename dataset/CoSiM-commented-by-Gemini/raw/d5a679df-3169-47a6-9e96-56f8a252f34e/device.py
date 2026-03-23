"""
This module implements a simulation of a distributed system of devices,
featuring a centralized setup and a two-phase reusable barrier for synchronization.
"""

from threading import Event, Thread, Semaphore, Lock


class ReusableBarrier(object):
    """
    A correct, reusable, two-phase barrier implemented using semaphores.
    It ensures that a group of threads can wait for each other to reach a
    synchronization point before any of them are allowed to continue.
    """

    def __init__(self, num_threads):
        """
        Initializes the barrier for a specific number of threads.

        Args:
            num_threads (int): The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Blocks the calling thread until all threads have called this method.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        Executes one phase of the barrier.

        Args:
            count_threads (list): A list containing the counter for the current phase.
            threads_sem (Semaphore): The semaphore for the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Last thread to arrive releases all other waiting threads.
                for _ in range(self.num_threads):
                    threads_sem.release()
                count_threads[0] = self.num_threads  # Reset for reuse.
        threads_sem.acquire()


class Device(object):
    """
    Represents a single device in the distributed simulation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.barrier = None
        self.InitializationEvent = Event()
        self.LockLocation = None  # Will hold the shared dictionary of locks
        self.LockDict = Lock()    # Lock to protect the dictionary of locks

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)

    def __str__(self):
        return f"Device {self.device_id}"

    def setup_devices(self, devices):
        """
        Performs a centralized setup, where device 0 initializes and distributes
        shared synchronization objects to all other devices.
        """
        if self.device_id == 0:
            # Device 0 is the master for setup.
            n = len(devices)
            self.barrier = ReusableBarrier(n)
            self.LockLocation = {}  # Shared dictionary for location-specific locks

            # Distribute shared objects to all devices.
            for d in devices:
                d.LockLocation = self.LockLocation
                d.barrier = self.barrier
                if d.device_id != 0:
                    d.InitializationEvent.set()  # Signal other devices to proceed.
        else:
            # Other devices wait for setup to be completed by device 0.
            self.InitializationEvent.wait()

        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be run at a certain location."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main worker thread for a device, handling the simulation loop.
    """

    def __init__(self, device):
        Thread.__init__(self, name=f"Device Thread {device.device_id}")
        self.device = device

    def run(self):
        """
        The main simulation loop for the device thread.
        """
        while True:
            # Synchronize all devices at the beginning of a time step.
            self.device.barrier.wait()

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break  # End of simulation.

            # Wait until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            for (script, location) in self.device.scripts:
                # Dynamically create and acquire a lock for the script's location.
                with self.device.LockDict:
                    if location not in self.device.LockLocation:
                        self.device.LockLocation[location] = Lock()
                
                with self.device.LockLocation[location]:
                    # Gather data from self and neighbors.
                    script_data = [dev.get_data(location) for dev in neighbours if dev.get_data(location) is not None]
                    local_data = self.device.get_data(location)
                    if local_data is not None:
                        script_data.append(local_data)

                    if script_data:
                        # Run the script and propagate the result.
                        result = script.run(script_data)
                        for dev in neighbours:
                            dev.set_data(location, result)
                        self.device.set_data(location, result)

            # Clear the event to prepare for the next timepoint.
            self.device.timepoint_done.clear()
