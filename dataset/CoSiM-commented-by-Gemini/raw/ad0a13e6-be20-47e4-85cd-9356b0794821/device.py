"""
A simulation framework for a network of communicating devices.

This module provides classes to simulate a network of devices that execute
scripts and share data. It includes a custom ReusableBarrier implementation for
thread synchronization, a Device class, and a multi-threaded execution model
with a DeviceThread and multiple Worker threads.
"""


from threading import Event, Thread, Lock, Condition

class ReusableBarrier(object):
    """
    A reusable barrier synchronization primitive.

    This class implements a barrier that can be used to wait for a fixed number
    of threads to reach a certain point of execution before all of them are
    allowed to proceed. The barrier is reusable, meaning it resets after all
    threads have passed.
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
        the barrier is reset to its initial state.
        """
        
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()





class Device(object):
    """
    Represents a single device in the simulated network.

    Each device has a set of scripts to execute, sensor data, and can
    communicate with its neighbors. The device's logic is driven by a
    DeviceThread.
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
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.locks_location = []
        self.barrier_timepoint = None


        self.thread = DeviceThread(self)
        self.thread.start()



    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up static resources for all devices.

        This method is intended to be called by one device to initialize
        shared resources for all devices in the simulation, such as locks for
        locations and a shared timepoint barrier.

        Args:
            devices: A list of all devices in the simulation.
        """
        
        
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            
            iteration = 0
            while iteration < 100:
                iteration += 1
                lock = Lock()
                self.locks_location.append(lock)

            for device in devices:
                device.locks_location = self.locks_location
                device.barrier_timepoint = barrier


    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device.

        Args:
            script: The script to be executed.
            location: The location associated with the script.
        """
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
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
        return self.sensor_data[location] if location in self.sensor_data else None

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


class Worker(Thread):
    """
    A worker thread that executes a subset of a device's scripts.
    """
    
    def __init__(self, thread_id, device, neighbors, nr_scripts, scripts):
        """
        Initializes a Worker thread.

        Args:
            thread_id: A unique ID for this worker within a device.
            device: The parent device.
            neighbors: A list of neighboring devices.
            nr_scripts: The total number of scripts for the device.
            scripts: The list of scripts to be executed.
        """
        Thread.__init__(self)
        self.neighbors = neighbors
        self.device = device
        self.scripts = scripts
        self.nr_scripts = nr_scripts
        self.thread_id = thread_id

    def run(self):
        """
        The main execution loop for the worker thread.

        This thread executes a subset of the device's scripts, determined by
        its thread_id. It gathers data from neighbors, runs the script, and
        then disseminates the result.
        """
        for index in range(self.thread_id, self.nr_scripts, 8):


            (script, location) = self.scripts[index]

            with self.device.locks_location[location]:
                script_data = []
                
                for device in self.neighbors:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    


                    result = script.run(script_data)

                    
                    for device in self.neighbors:
                        device.set_data(location, result)

                    self.device.set_data(location, result)


class DeviceThread(Thread):
    """
    The main thread for a device, responsible for orchestrating worker threads.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device: The Device object this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.list_workers = []

    def run(self):
        """
        The main execution loop for the device thread.

        This loop waits for a timepoint to be triggered, then spawns multiple
        Worker threads to execute the device's scripts in parallel. After the
        workers complete, it synchronizes with other devices using a barrier.
        """

        while True:

            neighbors = self.device.supervisor.get_neighbours()
            if neighbors is None:
                break

            self.device.timepoint_done.wait()

            
            nr_scripts = len(self.device.scripts)
            
            for thread_id in range(0, 8):

                worker = Worker(thread_id, self.device, neighbors, nr_scripts, self.device.scripts)


                self.list_workers.append(worker)
                worker.start()

            
            for worker in self.list_workers:
                worker.join()

            self.device.timepoint_done.clear()
            
            self.device.barrier_timepoint.wait()


