"""
This module defines a simulated device for a distributed sensor network.

It includes classes for the device itself, a thread to manage the device's
lifecycle, and a thread to execute scripts on the device. The simulation
involves coordinating multiple devices using barriers and locks to process
sensor data at different locations.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem

class Device(object):
    """
    Represents a single device in the simulated network.

    Each device has a unique ID, its own sensor data, and a connection to a
    supervisor. It runs a main thread to manage its operations, which include
    setting up connections to other devices, receiving and executing scripts,
    and synchronizing with other devices at each timepoint.

    Attributes:
        device_id (int): The unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor data.
        sensor_data_lock (Lock): A lock to protect access to sensor_data.
        supervisor: The supervisor object that manages the network.
        scripts (list): A list of scripts to be executed by the device.
        timepoint_done (Event): An event to signal the completion of a timepoint.
        setup_device (Event): An event to signal that the device has been set up.
        thread (DeviceThread): The main thread for the device.
        barrier (ReusableBarrierSem): A barrier for synchronizing with other devices.
        devices (list): A list of all devices in the network.
        locations (list): A list of locks for each location.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a new Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        
        self.sensor_data_lock = Lock()
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        
        self.setup_device = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        
        self.barrier = None
        self.devices = []
        self.locations = []


    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id


    def set_locks(self):
        """
        Initializes a set of locks for the locations.

        This is used to ensure that only one device is accessing the data for a
        particular location at a time.
        """
        i = 0
        while i < 100:
            self.locations.append(Lock())
            i = i + 1


    def setup_devices(self, devices):
        """
        Sets up the device with information about other devices in the network.

        This method is called by the supervisor to provide the device with a
        list of all other devices and to set up the synchronization barrier.
        """
        self.devices = devices
        nr_devices = len(devices)
        
        barrier_setup = ReusableBarrierSem(nr_devices)

        # Pre-condition: This block is executed only by the device with ID 0,
        # which acts as the master for setting up shared resources.
        if self.device_id == 0:
            
            
            self.set_locks()
            for device in devices:
                
                device.locations = self.locations
                
                if device.barrier is None:
                    device.barrier = barrier_setup
                
                device.setup_device.set()


    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific location.

        If the script is None, it signals that the current timepoint is done.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        This method is thread-safe.
        """
        with self.sensor_data_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location.

        This method is thread-safe.
        """
        with self.sensor_data_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()



class MyThread(Thread):
    """
    A thread to execute a single script on a device.

    This thread is responsible for acquiring the lock for a specific location,
    gathering data from neighboring devices, executing the script, and updating
    the data for the device and its neighbors.
    """

    def __init__(self, device, location, neighbours, script):
        """Initializes a new MyThread instance."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.location = location
        self.script = script

    def run(self):
        """
        The main execution logic of the thread.

        It acquires a lock for the location, gathers data, runs the script,
        and distributes the result.
        """
        self.device.locations[self.location].acquire()
        script_data = []
        
        # Invariant: Gathers data from all neighboring devices at the specified
        # location.
        for device in self.neighbours:
            
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = self.script.run(script_data)
            
            # Post-condition: The result of the script is written back to the
            # device and all its neighbors.
            for device in self.neighbours:
                device.set_data(self.location, result)
                
                self.device.set_data(self.location, result)
        self.device.locations[self.location].release()


class DeviceThread(Thread):
    """
    The main thread for a device.

    This thread manages the device's lifecycle, including waiting for setup,
    processing scripts for each timepoint, and synchronizing with other devices.
    """

    def __init__(self, device):
        """Initializes a new DeviceThread instance."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop of the device thread.

        It waits for the device to be set up, and then enters a loop to
        process scripts for each timepoint, synchronizing with a barrier at
        the end of each timepoint.
        """
        self.device.setup_device.wait()
        
        while 1:
            threads = []
            current_script_id = 0
            start = 0
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()
            scripts_received = len(self.device.scripts)
            
            # Block-level comment: Creates a thread for each assigned script.
            while current_script_id < scripts_received:
                (script, location) = self.device.scripts[current_script_id]
                thread = MyThread(self.device, location, neighbours, script)
                threads.append(thread)
                current_script_id = current_script_id + 1

            
            # Block-level comment: This section implements a strategy for
            # executing the script threads in batches to avoid creating too
            # many threads at once. For a small number of scripts, they are
            # all run in parallel. For a larger number, they are run in
            # batches of 8.
            if scripts_received < 8:
                for threadd in threads:
                    threadd.start()
                for threadd in threads:
                    threadd.join()
            else:
                while 1:
                    if scripts_received >= 8:


                        stop = start+8
                        for i in range(start, stop):
                            threads[i].start()
                        for i in range(start, stop):
                            threads[i].join()
                        start = start+8
                        scripts_received = scripts_received - 8
                    else:


                        stop = start+scripts_received
                        for i in range(start, stop):
                            threads[i].start()
                        for i in range(start, stop):
                            threads[i].join()
                        break
            self.device.timepoint_done.clear()
            self.device.barrier.wait()