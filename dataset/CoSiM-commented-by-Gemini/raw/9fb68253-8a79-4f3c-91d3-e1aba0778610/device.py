"""
This module defines a simulated device for a distributed sensor network,
including a reusable barrier implementation.

It provides classes for a `Device`, its main `DeviceThread`, and a
`ReusableBarrierSem` for synchronization. This setup is designed to simulate a
network of devices that process sensor data in a coordinated, time-stepped
manner.
"""

from threading import Lock, Thread, Event, Semaphore


class Device(object):
    """
    Represents a single device in the simulated network.

    Each device runs a main thread to manage its lifecycle, which includes
    setting up shared resources like barriers and location locks, receiving
    scripts from a supervisor, and executing them in coordination with other
    devices.

    Attributes:
        device_id (int): The unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor data.
        supervisor: The supervisor object that manages the network.
        script_received (Event): An event to signal the arrival of new scripts.
        scripts (list): A list of scripts to be executed by the device.
        timepoint_done (Event): An event to signal the completion of a timepoint.
        thread (DeviceThread): The main thread for the device.
        barrier (ReusableBarrierSem): A barrier for synchronizing with other devices.
        location_locks (dict): A dictionary of locks for each location.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        
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
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device with shared resources for the network.

        The device with ID 0 is responsible for creating the barrier and
        location locks, which are then shared with all other devices.
        """
        Device.devices_no = len(devices)
        # Pre-condition: This block is executed only by the device with ID 0,
        # which acts as the master for setting up shared resources.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            self.location_locks = {}
        else:
            self.barrier = devices[0].barrier
            self.location_locks = devices[0].location_locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific location.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a device.

    This thread manages the device's lifecycle, processing scripts for each
    timepoint and synchronizing with other devices using a barrier.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
    def run_scripts(self, script, location, neighbours):
        """
        Executes a script at a given location.

        This method acquires a lock for the location, gathers data from the
        device and its neighbors, runs the script, and updates the data with
        the result.
        """
        lock_location = self.device.location_locks.get(location)
        if lock_location is None and location is not None:
            self.device.location_locks[location] = Lock()
            lock_location = self.device.location_locks[location]
        lock_location.acquire()
        script_data = []
        
        # Invariant: Gathers data from all neighboring devices.
        for device in neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
            
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        if script_data != []:
            
            result = script.run(script_data)

            
            # Post-condition: The result is distributed to the device and its
            # neighbors.
            for device in neighbours:
                device.set_data(location, result)
            
            self.device.set_data(location, result)
        lock_location.release()

    def run(self):
        """
        The main loop of the device thread.

        It waits for the supervisor to signal the end of a timepoint, then
        creates and runs threads for each assigned script. After all scripts
        are executed, it waits at a barrier for all other devices to finish.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            self.device.timepoint_done.wait()
            tlist = []
            for (script, location) in self.device.scripts:
                thread = Thread(target=self.run_scripts, args=(script, location, neighbours))
                tlist.append(thread)
                thread.start()
            for thread in tlist:
                thread.join()
            self.device.timepoint_done.clear()
            self.device.barrier.wait()


class ReusableBarrierSem():
    """
    A reusable barrier implemented using semaphores.

    This barrier synchronizes a fixed number of threads in two phases,
    allowing it to be reused multiple times.
    """

    def __init__(self, num_threads):
        """Initializes a new ReusableBarrierSem instance."""

        self.num_threads = num_threads
        self.count_threads1 = self.num_threads


        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock()
        
        self.threads_sem1 = Semaphore(0)
        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            # Pre-condition: If this is the last thread to arrive, release all
            # waiting threads for the first phase.
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """The second phase of the barrier."""
        with self.counter_lock:
            self.count_threads2 -= 1
            # Pre-condition: If this is the last thread to arrive, release all
            # waiting threads for the second phase.
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()