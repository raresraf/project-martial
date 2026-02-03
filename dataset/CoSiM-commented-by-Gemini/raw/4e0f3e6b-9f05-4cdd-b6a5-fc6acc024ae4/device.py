"""
This module defines a simulated device and a reusable barrier for a
distributed sensor network.

The `Device` class represents a single node in the network, which processes
sensor data based on scripts assigned by a supervisor. The `ReusableBarrier`
class provides a synchronization mechanism for the devices.
"""

from threading import Event, Thread, Condition, Lock


class ReusableBarrier():
    """
    A reusable barrier implemented using a condition variable.

    This barrier synchronizes a fixed number of threads, allowing them to wait
    for each other to reach a certain point before proceeding. It is reusable,
    meaning it can be used multiple times.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        
        self.cond = Condition()
        
        
    def wait(self):
        """
        Causes a thread to wait at the barrier.

        When the required number of threads have called this method, all of them
        are released.
        """
        self.cond.acquire()
        
        self.count_threads -= 1
        # Pre-condition: If this is the last thread to arrive, notify all
        # waiting threads and reset the barrier.
        if self.count_threads == 0:
            self.cond.notify_all()
            
            self.count_threads = self.num_threads
            
        else:
            self.cond.wait()
            
        self.cond.release()

class Device(object):
    """
    Represents a single device in the simulated network.

    Each device runs a main thread to manage its lifecycle, including script
    execution and synchronization with other devices.

    Attributes:
        device_id (int): The unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor data.
        supervisor: The supervisor object that manages the network.
        script_received (Event): An event to signal the arrival of new scripts.
        scripts (list): A list of scripts to be executed by the device.
        timepoint_done (Event): An event to signal the completion of a timepoint.
        thread (DeviceThread): The main thread for the device.
        barrier (ReusableBarrier): A barrier for synchronizing with other devices.
        lock (Lock): A lock to protect the device's data.
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
        self.lock = Lock()
    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device with shared resources for the network.

        The device with ID 0 is responsible for creating the barrier, which is
        then shared with all other devices.
        """
        
        
        # Pre-condition: This block is executed only by the device with ID 0,
        # which acts as the master for setting up the shared barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrier(len(devices))
        else:
            self.barrier = devices[0].barrier
        

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
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        
        self.thread.join()

class MyThread(Thread):
    """
    A thread to execute a single script on a device.

    This thread gathers data from the device and its neighbors, executes a
    script, and updates the data with the result.
    """
    
    def __init__(self, neighbours, device, location, script):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.device = device
        self.location = location
        self.script = script
    def run(self):
        script_data = []
        # Invariant: Gathers data from all neighboring devices.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)



        if script_data != []:
            
            result = self.script.run(script_data)
            # Invariant: The result is written back to the device and its
            # neighbors, with locking to ensure thread safety.
            for device in self.neighbours:
                device.lock.acquire()
                device.set_data(self.location, result)
                device.lock.release()
            
            
            self.device.lock.acquire()
            self.device.set_data(self.location, result)
            self.device.lock.release()

class DeviceThread(Thread):
    """
    The main thread for a device.

    This thread manages the device's lifecycle, processing scripts for each
    timepoint and synchronizing with other devices.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main loop of the device thread.

        It waits for a signal to start a new timepoint, creates and runs
        threads for each assigned script, and then waits at a barrier for all
        other devices to complete the timepoint.
        """
        
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            self.device.timepoint_done.wait()


            self.device.timepoint_done.clear()

            threads = []
            
            # Block-level comment: Creates and starts a new thread for each
            # assigned script.
            for (script, location) in self.device.scripts:
                t = MyThread(neighbours, self.device, location, script)
                t.start()
                threads.append(t)

            for i in range(len(threads)):
                threads[i].join()
            
            self.device.barrier.wait()