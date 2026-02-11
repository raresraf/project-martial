"""
This module provides another variant of a simulated device for a concurrent system.

A key architectural feature of this implementation is the use of a global
dictionary to manage location-based locks, which is generally discouraged as
it creates a global state. It also features a two-phase, semaphore-based
reusable barrier.
"""


import sys

from threading import *


class ReusableBarrierSem():
    """
    A reusable barrier implemented using two semaphores and a lock.

    This implementation uses a two-phase protocol (`phase1` and `phase2`) to
    ensure that threads can safely reuse the barrier without race conditions
    from previous wait cycles.
    """


    
    
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """The first phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        
        self.threads_sem1.acquire()
    
    def phase2(self):
        """The second phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a single device in the simulation.

    This device class relies on a leader-based setup to initialize a shared
    barrier and a global dictionary for location-based locks.
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

        

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization objects for the device network.

        The device with ID 0 acts as a leader to set up a shared barrier.
        It also initializes a global dictionary `dic` to store locks, which
        is a highly unconventional and generally unsafe practice.
        """
        
        if self.device_id == 0:
            
            num_threads = len(devices)
            
            bar = ReusableBarrierSem(len(devices))
            

            for d in devices:
                
                d.barrier = bar
            
            
            # This creates a global dictionary to store locks. Using global
            # variables for shared state is risky and can make the system
            # difficult to reason about and maintain.
            global dic
            dic = {}
                       
        pass

    def assign_script(self, script, location):
        """
        Assigns a script and populates the global lock dictionary.

        If a lock for the given location does not exist in the global `dic`,
        it is created.
        """

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()


        
        # Check and lazily initialize locks in the global dictionary.
        if location in dic.keys():
            return
        
        else:
            
            dic[location] = Lock()
        

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for the device's lifecycle."""

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device.

        It synchronizes at a barrier, then processes each script by acquiring
        a lock from a global dictionary, and finally waits on an event.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for all devices to be ready before starting script execution.
            self.device.barrier.wait()

            
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Acquire a location-specific lock from the global dictionary.
                dic[location].acquire()
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                
                dic[location].release()
            
            # Wait for a signal that the timepoint is complete before the next cycle.
            self.device.timepoint_done.wait()
       