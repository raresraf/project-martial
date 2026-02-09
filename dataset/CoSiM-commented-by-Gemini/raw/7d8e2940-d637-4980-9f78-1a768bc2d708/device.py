"""
@file device.py
@brief Defines a device model for a distributed simulation using a global lock.

This file implements a simulation device where all script executions across the
entire system are serialized by a single, global lock. A custom, condition-based
`ReusableBarrier` is used for timepoint synchronization.
"""

from threading import *


class ReusableBarrier():
    """
    A reusable barrier implemented using a Condition variable.

    @note This implementation may be subject to race conditions. If a notified
          thread re-enters `wait()` before the last thread (which triggered the
          notification) has released the condition lock and exited the `wait()`,
          it could lead to deadlocks or unpredictable behavior.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 
 
    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.cond.acquire()                      
        self.count_threads -= 1;
        if self.count_threads == 0:
            self.cond.notify_all()             
            self.count_threads = self.num_threads    
        else:
            self.cond.wait();                    
        self.cond.release();


class Device(object):
    """
    Represents a device in the simulation, using a global lock for all data operations.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event() # This event is unused.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier  = None 
        self.lock = None 

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared global barrier and a global lock.

        The first device to call this method creates a single lock and a single
        barrier that are then shared by all devices in the simulation.
        """
        if self.barrier == None: 
            barrier = ReusableBarrier(len(devices))
            L = Lock()
            # Invariant: Distribute the same barrier and lock instances to all devices.
            for dev in devices: 
                dev.barrier = barrier
                dev.lock = L
                
    def assign_script(self, script, location):
        """Assigns a script to be executed."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that script assignment is complete.
            self.script_received.set() 

    def get_data(self, location):
        """Retrieves sensor data. This read is not synchronized by this method."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data. This write is not synchronized by this method."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop.

        It processes scripts serially, using a global lock to ensure that only
        one script in the entire system can run at any given time.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break   

            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()
            
            # Block Logic: Process all assigned scripts serially.
            for (script, location) in self.device.scripts:
                # Pre-condition: Acquire the single global lock, serializing all
                # script executions across all devices.
                self.device.lock.acquire()
                script_data = []
                
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
                    
                    # Propagate results back to all relevant devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                
                # Release the global lock.
                self.device.lock.release()
            
            # Wait at the barrier for all devices to finish the timepoint.
            self.device.barrier.wait()
            self.device.script_received.clear()