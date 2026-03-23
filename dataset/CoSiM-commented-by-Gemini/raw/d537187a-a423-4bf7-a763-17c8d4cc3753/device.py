"""
This module simulates a distributed system of devices that process sensor data
concurrently using multithreading. It includes advanced synchronization mechanisms
like a reusable barrier.
"""

from threading import Lock, Event, Thread, Semaphore, Condition


class ReusableBarrierSem:
    """
    A reusable barrier implemented using semaphores.
    This barrier allows a set of threads to wait for each other to reach a
    certain point before proceeding. It is reusable, meaning it can be used
    multiple times.
    """

    def __init__(self, num_threads):
        """
        Initializes the reusable barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that will synchronize on this barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to wait until all threads have called this method.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the two-phase barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase of the two-phase barrier."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()


class Device(object):
    """
    Represents a device in the distributed system. Each device runs multiple
    threads to process scripts on sensor data.
    """
    location_locks = []
    barrier = None
    nr_t = 8  # Number of threads per device

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of the device's sensor data.
            supervisor (Supervisor): A supervisor object to get neighbors from.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.neighbours_event = Event()
        self.threads = [DeviceThread(self, i) for i in range(Device.nr_t)]
        for thread in self.threads:
            thread.start()

    def __str__(self):
        return f"Device {self.device_id}"

    @staticmethod
    def setup_devices(devices):
        """
        Sets up the shared barrier for all devices.
        
        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        Device.barrier = ReusableBarrierSem(Device.nr_t * len(devices))

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device.

        Args:
            script (Script): The script to be executed.
            location (str): The location associated with the script.
        """
        if location not in [elem[0] for elem in Device.location_locks]:
            Device.location_locks.append((location, Lock()))

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (str): The location of the sensor data.

        Returns:
            The sensor data, or None if not available.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates the sensor data for a given location.

        Args:
            location (str): The location of the sensor data.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by joining all its threads.
        """
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread within a Device. These threads perform the actual script
    execution and data synchronization.
    """

    def __init__(self, device, index):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent device.
            index (int): The index of this thread within the device.
        """
        Thread.__init__(self, name=f"Device Thread {device.device_id}-{index}")
        self.device = device
        self.index = index
        self.neighbours = None

    def run(self):
        """
        The main loop for the worker thread.
        """
        while True:
            # The first thread gets the neighbors and notifies others.
            if self.index == 0:
                self.neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours_event.set()
            else:
                self.device.neighbours_event.wait()
                self.neighbours = self.device.threads[0].neighbours
            
            if self.neighbours is None:
                break

            self.device.timepoint_done.wait()

            # Process a subset of scripts assigned to the device.
            for j in range(self.index, len(self.device.scripts), Device.nr_t):
                location, script = self.device.scripts[j]

                # Acquire lock for the location to ensure exclusive access.
                for loc, lock in Device.location_locks:
                    if location == loc:
                        lock.acquire()

                # Gather data from neighbors and self.
                script_data = [dev.get_data(location) for dev in self.neighbours if dev.get_data(location) is not None]
                local_data = self.device.get_data(location)
                if local_data is not None:
                    script_data.append(local_data)

                if script_data:
                    # Execute the script on the gathered data.
                    result = script.run(script_data)

                    # Propagate the result to neighbors and self.
                    for dev in self.neighbours:
                        dev.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the lock for the location.
                for loc, lock in Device.location_locks:
                    if location == loc:
                        lock.release()

            # Synchronize with all other threads in the system.
            Device.barrier.wait()
            if self.index == 0:
                self.device.timepoint_done.clear()

            if self.index == 0:
                self.device.neighbours_event.clear()
            Device.barrier.wait()
