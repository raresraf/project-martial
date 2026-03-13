"""
A simulation framework for a network of communicating devices.

This module provides classes to simulate a network of devices that execute
scripts and share data. It features a `Device` class, a `DeviceThread` that
distributes scripts to a pool of `ScriptThread` workers, and a `Barrier` class
for synchronization.
"""


from threading import Event, Thread, Lock, Condition


class Device(object):
    """
    Represents a single device in the simulated network.

    Each device has a set of scripts to execute, sensor data, and can
    communicate with its neighbors. The device's logic is orchestrated by a
    `DeviceThread`.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id: A unique identifier for the device.
            sensor_data: A dictionary of sensor data for this device.
            supervisor: A supervisor object that manages the network.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.data_locks = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up static, shared resources for all devices.

        This method, intended to be called by a single device, initializes a
        shared barrier and a set of locks for all data locations across all
        devices in the simulation.

        Args:
            devices: A list of all devices in the simulation.
        """
        
        if self.device_id == 0:

            new_bar = Barrier(len(devices))

            locations = []
            for device in devices:
                for (location, value) in device.sensor_data.items():
                    locations.append(location)

            max_loc = max(locations)
            data_locks = []
            for i in range(max_loc + 1):
                data_locks.append(Lock())
            
            for device in devices:
                device.set_barrier_locks(new_bar, data_locks)

    def set_barrier_locks(self, barrier, data_locks):
        """
        Assigns the shared barrier and data locks to this device.

        Args:
            barrier: The shared Barrier instance.
            data_locks: A list of Lock objects for data locations.
        """
        
        self.barrier = barrier
        self.data_locks = data_locks

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device.

        Args:
            script: The script to be executed.
            location: The location associated with the script.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location: The location to get data from.

        Returns:
            The sensor data for the given location, or None if not available.
        """
        
        if location in self.sensor_data:
            data = self.sensor_data[location]
            return data
        else:
            return None

    def set_data(self, location, data):
        """
        Sets the sensor data for a given location.

        Args:
            location: The location to set data for.
            data: The new data value.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a device, responsible for orchestrating script
    execution.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device: The Device object this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device thread.

        This loop waits for a timepoint, then distributes the device's scripts
        among a pool of `ScriptThread` workers for parallel execution. After
        all workers complete, it synchronizes with other devices.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            threads = []
            scripts = []

            for i in range(8):
                scripts.append([])

            count = 0
            
            for (script, location) in self.device.scripts:
                scripts[count % 8].append((script, location))
                count += 1

            
            for i in range(8):
                if len(scripts[i]) > 0:


                    thread = ScriptThread(self.device, scripts[i], neighbours)
                    thread.start()
                    threads.append(thread)

            
            for i in range(len(threads)):
                threads[i].join()

            
            self.device.barrier.wait()
            self.device.timepoint_done.clear()


class ScriptThread(Thread):
    """
    A worker thread that executes a subset of a device's scripts.
    """

    def __init__(self, device, scripts, neighbours):
        """
        Initializes a ScriptThread.

        Args:
            device: The parent device.
            scripts: A list of (script, location) tuples to execute.
            neighbours: A list of neighboring devices.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours

    def run(self):
        """
        The main execution logic for the script thread.

        For each assigned script, it acquires a lock for the script's
        location, gathers data, runs the script, and disseminates the result.
        """
        
        
        for (script, location) in self.scripts:



            self.device.data_locks[location].acquire()

            script_data = []

            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)
            if script_data != []:

                result = script.run(script_data)



                for device in self.neighbours:
                    device.set_data(location, result)

                self.device.set_data(location, result)

            self.device.data_locks[location].release()


class Barrier():
    """
    A reusable barrier synchronization primitive.

    This class implements a barrier that can be used to wait for a fixed number
    of threads to reach a certain point of execution before all of them are
    allowed to proceed.
    """

    def __init__(self, num_threads):
        """Initializes the Barrier with the number of threads to wait for."""
        
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Causes the calling thread to wait until all threads have called wait().

        When the last thread calls wait(), all waiting threads are woken up and
        the barrier is reset.
        """
        
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()
