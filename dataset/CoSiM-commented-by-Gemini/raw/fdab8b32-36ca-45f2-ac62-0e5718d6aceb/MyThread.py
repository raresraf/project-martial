"""
This module defines a simulation of a distributed system with multi-threaded
devices. It consists of three main classes: Device, DeviceThread (the control
thread for a device), and MyThread (the worker thread).
"""

from threading import Thread, Lock, Event

class MyThread(Thread):
    """
    A worker thread for a Device.
    
    This thread executes scripts assigned to it by the device's control thread.
    It waits for permission to run, processes its scripts, and signals when it's
    finished.
    """

    def __init__(self, device):
        """
        Initializes the MyThread worker.

        Args:
            device: The Device instance this worker belongs to.
        """
        Thread.__init__(self)
        self.device = device
        self.scripts_list = []
        self.neighbours_list = []
        self.permission = Event()
        self.finish = Event()
        self.thread_killed = 0
        self.lists = Lock()

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            # Wait for the control thread to give permission to run.
            self.permission.wait()

            # If the thread is marked as killed, exit the loop.
            if self.thread_killed == 1:
                break

            self.permission.clear()
            self.finish.clear()

            # Process all scripts in the list.
            while self.scripts_list and self.neighbours_list:
                with self.lists:
                    script, place = self.scripts_list.pop(0)
                    neighbours = self.neighbours_list.pop(0)

                # Acquire the lock for the specific location.
                with self.device.scripts_locks[place]:
                    data_list = []
                    # Gather data from the current device.
                    data = self.device.get_data(place)
                    if data is not None:
                        data_list.append(data)

                    # Gather data from neighbors.
                    for neighbour in neighbours:
                        data = neighbour.get_data(place)
                        if data is not None:
                            data_list.append(data)

                    if data_list:
                        # Run the script and update data.
                        result = script.run(data_list)
                        self.device.set_data(place, result)
                        for neighbour in neighbours:
                            neighbour.set_data(place, result)

            # Signal that this worker has finished its tasks for the timepoint.
            self.finish.set()


from threading import Event, Thread, Lock
from MyBarrier import MyBarrier # Assuming MyBarrier is defined in a separate file.

class Device(object):
    """
    Represents a device in the simulated distributed system.

    Each device has a main control thread (DeviceThread) and a pool of worker
    threads (MyThread) to execute scripts in parallel.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id: A unique identifier for the device.
            sensor_data: A dictionary representing the device's sensor data.
            supervisor: A supervisor object that manages the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.ready = Event()

        # The main control thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

        self.threads = []
        self.nr_threads = 8 

        # Create a pool of worker threads.
        for _ in range(self.nr_threads):
            thread = MyThread(self)
            self.threads.append(thread)
            thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier and locks for all devices. This is called
        by the master device (device_id 0).
        """
        if self.device_id == 0:
            nr_devices = len(devices)
            barrier = MyBarrier(nr_devices)

            # This locking mechanism seems to be creating a global pool of locks
            # and distributing them, which is an unusual pattern.
            places = []
            locks = []
            for device in devices:
                places.extend(device.sensor_data.keys())
                data = len(device.sensor_data.keys())
                for _ in range(data):
                    locks.append(Lock())
                device.barrier = barrier
                device.scripts_locks = locks
                device.ready.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed. If script is None, it signals the
        end of script assignments for the current timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Sets sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its threads."""
        for thread in self.threads:
            thread.join()
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a Device.

    This thread waits for scripts, distributes them to the worker threads,
    and synchronizes with other devices at the end of each timepoint.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device: The Device instance this thread controls.
        """
        Thread.__init__(self)
        self.device = device

    def run(self):
        """The main execution loop for the control thread."""
        self.device.ready.wait()

        while True:
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                # Signal all worker threads to terminate.
                for thread in self.device.threads:
                    thread.thread_killed = 1
                    thread.permission.set()
                break

            # Wait for all scripts for the current timepoint to be assigned.
            self.device.script_received.wait()

            # Distribute the scripts among the worker threads.
            for i, script in enumerate(self.device.scripts):
                crt = i % self.device.nr_threads
                self.device.threads[crt].scripts_list.append(script)
                self.device.threads[crt].neighbours_list.append(neighbours)

            # Give all worker threads permission to start processing.
            for thread in self.device.threads:
                thread.permission.set()

            self.device.script_received.clear()

            # Wait for all worker threads to finish.
            for thread in self.device.threads:
                thread.finish.wait()

            # Wait at the barrier for all other devices to finish the timepoint.
            self.device.barrier.wait()
