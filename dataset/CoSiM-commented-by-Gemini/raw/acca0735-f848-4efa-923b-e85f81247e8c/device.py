
"""
A simulation framework for a network of communicating devices.

This module provides classes to simulate a network of devices that execute
scripts and share data. It includes a custom Barrier implementation for thread
synchronization and a Device class that runs its logic in multiple threads.
"""


from threading import Event, Thread, Condition, Lock


class Barrier(object):
    """
A reusable barrier synchronization primitive.

This class implements a barrier that can be used to wait for a fixed number
of threads to reach a certain point of execution before all of them are
allowed to proceed.
"""

    def __init__(self, num_threads=0):
        """Initializes the Barrier with the number of threads to wait for."""
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        
        self.cond = Condition()

    def wait(self):
        """
        Causes the calling thread to wait until all threads have called wait().

        When the last thread calls wait(), all waiting threads are woken up and
        the barrier is reset to its initial state, so it can be reused.
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
    communicate with its neighbors. The device's logic is run by multiple
    DeviceThread instances.
    """
    
    
    bariera_devices = Barrier()
    locks = []

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
        self.locations = []
        
        self.nr_scripturi = 0
        
        self.script_crt = 0

        
        
        self.timepoint_done = Event()

        
        self.neighbours = []
        self.event_neighbours = Event()
        self.lock_script = Lock()
        self.bar_thr = Barrier(8)

        
        self.thread = DeviceThread(self, 1)
        self.thread.start()
        self.threads = []
        for _ in range(7):
            tthread = DeviceThread(self, 0)


            self.threads.append(tthread)
            tthread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up static resources for all devices.

        Initializes a barrier for all devices in the simulation and a list of
        locks for each location.

        Args:
            devices: A list of all devices in the simulation.
        """
        
        
        Device.bariera_devices = Barrier(len(devices))
        
        if Device.locks == []:
            for _ in range(self.supervisor.supervisor.testcase.num_locations):
                Device.locks.append(Lock())

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device.

        Args:
            script: The script to be executed.
            location: The location associated with the script.
        """
        
        if script is not None:
            self.scripts.append(script)
            self.locations.append(location)
            
            self.nr_scripturi += 1
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
        return self.sensor_data[location] if location in \
        self.sensor_data else None

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
        """Shuts down the device by joining all its threads."""
        
        self.thread.join()
        for tthread in self.threads:
            tthread.join()


class DeviceThread(Thread):
    """
A thread that executes the logic of a Device.
"""

    def __init__(self, device, first):
        """
        Initializes a DeviceThread.

        Args:
            device: The Device object this thread belongs to.
            first: A flag indicating if this is the main thread for the device.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.first = first

    def run(self):
        """The main execution loop for the device thread."""
        while True:
            
            
            if self.first == 1:
                
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.script_crt = 0
                self.device.event_neighbours.set()

            
            self.device.event_neighbours.wait()

            if self.device.neighbours is None:
                break

            
            self.device.timepoint_done.wait()

            while True:
                
                self.device.lock_script.acquire()
                index = self.device.script_crt
                self.device.script_crt += 1
                self.device.lock_script.release()

                
                
                if index >= self.device.nr_scripturi:
                    break

                
                location = self.device.locations[index]
                script = self.device.scripts[index]

                
                
                Device.locks[location].acquire()

                script_data = []
                    
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                        
                    result = script.run(script_data)

                    
                    
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                Device.locks[location].release()

            


            self.device.bar_thr.wait()
            
            if self.first == 1:
                self.device.event_neighbours.clear()
                self.device.timepoint_done.clear()
            self.device.bar_thr.wait()
            
            if self.first == 1:
                Device.bariera_devices.wait()

