"""
This module simulates a distributed system of devices that concurrently process
sensor data. The simulation uses multiple threads and various synchronization
primitives to coordinate the devices' operations in discrete time steps.
"""

from threading import Event, Thread, Lock, Semaphore

class Device(object):
    """
    Represents a single device in the distributed system simulation.
    Each device has its own sensor data and executes scripts on this data in
    coordination with its neighbors.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.
        :param device_id: A unique identifier for the device.
        :param sensor_data: A dictionary representing the device's local sensor data,
                            mapping locations to values.
        :param supervisor: A supervisor object that provides neighborhood information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal that a script has been assigned.
        self.scripts = []  # A list of (script, location) tuples to be executed.
        self.timepoint_done = Event()  # Event to signal that a timepoint's scripts are all assigned.
        self.lock = {}  # A dictionary of locks, one for each data location.
        self.barrier = None  # A barrier for synchronizing with other devices at the end of a timepoint.
        self.devices = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device with information about all other devices in the system.
        This method is called once to initialize the simulation environment.
        :param devices: A list of all Device objects in the simulation.
        """
        self.devices = devices
        # A reusable barrier for all devices to synchronize at the end of a time step.
        self.barrier = ReusableBarrierSem(len(self.devices))

        # Create a shared lock for each unique data location across all devices.
        for location in self.sensor_data:
            self.lock[location] = Lock()
        for device in devices:
            for location in device.sensor_data:
                self.lock[location] = Lock()

        # Distribute the shared barrier and locks to all devices.
        for i in xrange(len(self.devices)):
            self.devices[i].barrier = self.barrier
            self.devices[i].lock = self.lock

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.
        :param script: The script object to execute.
        :param location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signifies the end of script assignments for the current timepoint.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data from a specific sensor location on this device.
        :param location: The location to get data from.
        :return: The data at the given location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates data at a specific sensor location on this device.
        :param location: The location to update.
        :param data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class MyThread(Thread):
    """
    A worker thread that executes a single script on a specific data location.
    It coordinates with neighboring devices to gather data and update results.
    """
    def __init__(self, my_id, device, neighbours, lock, script, location):
        Thread.__init__(self, name="Thread %d from device %d" % (my_id, device.device_id))
        self.device = device
        self.my_id = my_id
        self.neighbours = neighbours
        self.lock = lock
        self.script = script
        self.location = location

    def run(self):
        """
        Executes the script. It gathers data from its device and neighbors,
        runs the script, and distributes the result back.
        """
        # Acquire the lock for the specific location to ensure exclusive access.
        with self.lock[self.location]:
            script_data = []
            
            # Gather data from all neighboring devices for the given location.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from the parent device itself.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data:
                # Execute the script on the collected data.
                result = self.script.run(script_data)

                # Propagate the result back to the neighbors.
                for device in self.neighbours:
                    device.set_data(self.location, result)

                # Update the result on the parent device.
                self.device.set_data(self.location, result)

    def shutdown(self):
        """Shuts down the worker thread."""
        self.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device. It manages a pool of worker threads
    to execute scripts and handles the device's lifecycle and synchronization.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.numThreads = 0
        self.listThreads = []

    def run(self):
        """
        The main loop for the device. It waits for scripts, executes them using a
        thread pool, and synchronizes with other devices at the end of each timepoint.
        """
        while True:
            # Get the list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # If the supervisor returns None, it's a signal to terminate.
                break

            # Wait until all scripts for the current timepoint have been received.
            self.device.script_received.wait()

            # --- Simple Thread Pool Management ---
            # Manages a pool of up to 8 worker threads (MyThread).
            for (script, location) in self.device.scripts:
                if len(self.listThreads) < 8:
                    # If the pool is not full, create a new worker thread.
                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.append(thread)
                    thread.start()
                    self.numThreads += 1
                else:
                    # If the pool is full, find a finished thread to replace.
                    index = -1
                    for i in xrange(len(self.listThreads)):
                        if not self.listThreads[i].is_alive():
                            self.listThreads[i].join()
                            index = i

                    self.listThreads.remove(self.listThreads[index])

                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.insert(index, thread)
                    self.listThreads[index].start()
                    self.numThreads += 1

            # Wait for all worker threads for the current timepoint to complete.
            for i in xrange(len(self.listThreads)):
                self.listThreads[i].join()

            # Wait for the signal that all scripts have been assigned for this timepoint.
            self.device.timepoint_done.wait()
            
            # Reset events for the next timepoint.
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            
            # Synchronize with all other devices before starting the next timepoint.
            self.device.barrier.wait()


class ReusableBarrierSem():
    """
    A reusable barrier implementation using semaphores.
    This allows a set of threads to wait for each other to reach a certain point
    of execution before any of them are allowed to proceed. It is reusable,
    meaning it can be used multiple times (e.g., in a loop).
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        # Semaphores used to block and release threads for the two phases.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have called wait().
        It consists of two phases to ensure reusability.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread to arrive releases all waiting threads for phase 1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next use.
        self.threads_sem1.acquire()

    def phase2(self):
        """Second synchronization phase, to prevent race conditions for reuse."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # The last thread to arrive releases all waiting threads for phase 2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for next use.
        self.threads_sem2.acquire()
