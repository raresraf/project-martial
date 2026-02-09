"""
@file device.py
@brief Defines a device model for a simulation using a class-level barrier.

This file implements a `Device` and `DeviceThread` for a distributed simulation.
It uses a custom `ReusableBarrier` based on a `Condition` variable, which is
shared among all devices as a class-level attribute. Script processing is done
serially within the `DeviceThread` under a single lock.
"""

from threading import Thread,Event,Condition,Lock

class ReusableBarrier():
    """
    A reusable barrier implemented using a Condition variable.

    This barrier blocks threads calling `wait()` until a specified number of
    threads have arrived.

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
        """
        Blocks the calling thread until all threads have reached the barrier.
        """
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
    Represents a single device in the simulation.
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
        Initializes a shared, class-level barrier and a per-device lock.
        Also starts the device's main execution thread.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: The root device (ID 0) creates a class-level barrier
        # that is shared by all instances of Device.
        for i in devices:
	        if self.device_id == 0:
	            Device.barrier = ReusableBarrier(len(devices))

        # Each device instance gets its own lock.
        self.lock = Lock()
        self.thread = DeviceThread(self, Device.barrier , self.lock)
        self.thread.start()


    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of script assignment for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device.

    It synchronizes with other devices at the start of each timepoint and then
    processes all its assigned scripts serially.
    """

    def __init__(self, device , barrier , lock):
        """
        Initializes the DeviceThread.
        
        Args:
            device (Device): The parent device object.
            barrier: The shared ReusableBarrier instance.
            lock (Lock): The lock for this specific device.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.lock = lock
        
    def run(self):
        """
        The main simulation loop.

        @note The locking strategy here only protects the script execution loop on
              the local device. It does not prevent race conditions when multiple
              devices access the same data concurrently, as `get_data` and
              `set_data` calls on neighbor devices are not protected by a shared lock.
        """
        while True:
            # --- Barrier Wait ---
            # Synchronize with all other devices at the beginning of the timepoint.
            self.barrier.wait()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait for supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Pre-condition: Acquire the local lock to serialize this device's script processing.
            self.lock.acquire()
            # Block Logic: Process all assigned scripts serially in this thread.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Gather data from neighbors. This access is not synchronized.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Invariant: Data is gathered, and the script is ready to run.
                if script_data != []:
                    
                    result = script.run(script_data)

                    # Propagate result to neighbors. This is not synchronized.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Update local data.
                    self.device.set_data(location, result)
            
            self.lock.release()