"""
This module implements a device simulation for a concurrent system.

It features a simplistic, non-reentrant reusable barrier and a critically
flawed locking strategy where each device uses its own lock, leading to a high
risk of race conditions between devices.
"""

from threading import Thread,Event,Condition,Lock

class ReusableBarrier():
    """
    A simple, non-reentrant barrier implementation using a Condition variable.

    This barrier is not a proper two-phase reusable barrier. Its attempt at
    reusability by resetting the counter can lead to race conditions if threads
    enter the `wait` cycle at different speeds.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    
        self.cond = Condition()                  
                                                 
 
    def wait(self):
        """Blocks until all threads have reached the barrier."""
        self.cond.acquire()                      
        self.count_threads -= 1;
        if self.count_threads == 0:
            # Last thread notifies all others and resets the counter.
            self.cond.notify_all()               
            self.count_threads = self.num_threads    
        else:
            self.cond.wait();                    
        self.cond.release();    

class Device(object):
    """
    Represents a single device in the simulation.

    Each device gets its own lock and uses a shared, class-level barrier for
    synchronization with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.lock = None
        self.barrier = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the shared barrier and individual locks for each device.

        The device with ID 0 creates a static barrier shared by all devices.
        Crucially, each device instance creates its own separate Lock.
        """
        
        
        for i in devices:


	        if self.device_id == 0:
	            # The leader device creates a barrier and assigns it to a class variable.
	            Device.barrier = ReusableBarrier(len(devices))
        self.lock = Lock()
        self.thread = DeviceThread(self, Device.barrier , self.lock)
        self.thread.start()


    def assign_script(self, script, location):
        """Assigns a script to the device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

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

    def __init__(self, device , barrier , lock):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device
        self.barrier = barrier
        self.lock = lock
        
    def run(self):
        """
        The main execution loop for the device.

        This loop contains a critical flaw: it uses a device-local lock, which
        does not prevent race conditions between different devices.
        """
        while True:
            
            # Wait for all devices to synchronize at the start of the cycle.
            self.barrier.wait()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            
            # Wait for the signal to start processing scripts.
            self.device.timepoint_done.wait()
            
            self.device.timepoint_done.clear()

            
            # CRITICAL FLAW: This lock is local to this device instance. It does
            # NOT prevent another device's thread from concurrently modifying
            # the same data on a shared neighbor, leading to race conditions.
            self.lock.acquire()
            for (script, location) in self.device.scripts:
                script_data = []
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    
                    # Unsafe concurrent modification of shared neighbor data.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
            self.lock.release()