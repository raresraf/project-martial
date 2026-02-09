"""
@file device.py
@brief Defines a device model for a simulation using a global dictionary for locking.

@warning This implementation is critically flawed. It relies on a global variable
         (`dic`) to manage locks, which breaks encapsulation and is a poor design
         pattern. Furthermore, the main `DeviceThread` loop contains a deadlock
         bug due to the incorrect use of `timepoint_done.wait()`, which will
         cause the simulation to hang after the first timepoint.
"""

import sys
from threading import *

class ReusableBarrierSem():
    """
    A reusable barrier implemented using two semaphores and a lock.
    This is a standard, correct implementation of a two-phase barrier.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
    
    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """The first phase of the two-phase barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()
    
    def phase2(self):
        """The second phase of the two-phase barrier."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a device that uses a global dictionary for location-based locking.
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
        Initializes a shared barrier and a global dictionary for locks.
        
        @warning The use of `global dic` is a significant anti-pattern.
        """
        if self.device_id == 0:
            num_threads = len(devices)
            bar = ReusableBarrierSem(len(devices))
            for d in devices:
                d.barrier = bar
            
            # Initialize a global dictionary to hold location locks.
            global dic
            dic = {}
        pass

    def assign_script(self, script, location):
        """
        Assigns a script and lazily populates the global lock dictionary.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

        # If the location is new, create a lock for it in the global dictionary.
        if location in dic.keys():
            return
        else:
            dic[location] = Lock()
        
    def get_data(self, location):
        """Retrieves sensor data. This access is not synchronized."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data. This access is not synchronized."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a device.
    
    @warning Contains a deadlock bug.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop.
        
        @warning The `timepoint_done.wait()` call at the end of the loop will
                 block indefinitely after the first iteration, as the event is
                 never reset and re-set for subsequent timepoints. This is a
                 deadlock.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait at the barrier at the start of the timepoint.
            self.device.barrier.wait()

            # Process scripts serially.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Acquire the location-specific lock from the global dictionary.
                dic[location].acquire()
                
                # Gather data from neighbors and self.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    result = script.run(script_data)
                    
                    # Propagate results.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                
                # Release the lock.
                dic[location].release()
            
            # This wait will block forever after the first timepoint.
            self.device.timepoint_done.wait()