"""
This module provides a framework for a distributed device simulation using
a multi-threaded approach with a coordinator device.

It features a `Device` class with an internal thread pool and a `ReusableBarrier`
implemented using `threading.Condition`. A device with ID 0 acts as a
coordinator, initializing shared resources like location-based locks and a
global synchronization barrier for all threads in the simulation.
"""

from threading import Thread, Condition, Lock

class ReusableBarrier(object):
    """
    A reusable barrier implemented using a `threading.Condition`.

    This allows a set number of threads to wait for each other to reach a
    synchronization point. It is "reusable" because it resets itself after
    all threads have passed.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads 
        self.cond = Condition()               
                                              

    def wait(self):
        """
        Blocks the calling thread until all threads have called `wait()`
        on this barrier instance.
        """
        
        self.cond.acquire()                   
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread has arrived; wake up all waiting threads.
            self.cond.notify_all()            
            # Reset the barrier for the next use.
            self.count_threads = self.num_threads 
        else:
            # Not the last thread, so wait to be notified.
            self.cond.wait()                 
        self.cond.release()                  


class Device(object):
    """
    Represents a single device in the simulation with its own thread pool.

    Device 0 acts as a special coordinator node that sets up shared
    synchronization objects for the entire device network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        


        self.device_id = device_id 
        self.sensor_data = sensor_data 
        self.supervisor = supervisor
        self.scripts_received = ReusableBarrier(9)
        self.scripts = {} 
        self.devices = None
        self.timepoint_done = None
        self.semafor = {}
        self.thread_list = []
        self.neighbours_barrier = ReusableBarrier(8)
        self.contor = 0

        for i in range(8):
            self.thread_list.append(DeviceThread(self, i))
            self.scripts.update({i:[]})

    def __str__(self):
        """String representation of the device."""
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources if this is the coordinator device.
        This must be called on device 0.
        """
        
        

        if self.device_id == 0:
            self.devices = devices
            
            # Create a global barrier for all threads in the simulation.
            self.timepoint_done = ReusableBarrier(8 * len(self.devices))
            
            # Create shared locks for each unique sensor location.
            for device in self.devices:
                
                for location in device.sensor_data:
                    if location not in self.semafor:
                        self.semafor.update({location: Lock()})
                # Pass shared resources to other devices.
                if device.device_id != 0:
                    
                    device.initialize_device(self.timepoint_done, self.semafor, self.devices)
            
            # Start the coordinator's own thread pool.
            for thread in self.thread_list:
                thread.start()

    def initialize_device(self, timepoint_done, semafor, devices):
        """
        Initializes a non-coordinator device with shared resources from the coordinator.
        """
        
        self.timepoint_done = timepoint_done
        self.semafor = semafor
        self.devices = devices
        
        # Start this device's thread pool.
        for thread in self.thread_list:
            thread.start()

    def assign_script(self, script, location):
        """
        Assigns a script to a worker thread in a round-robin fashion.
        """
        if script is not None:
            
            # Add script to the queue of the next thread.
            self.scripts[self.contor%8].append((script, location))
            self.contor += 1
        else:
            
            # A None script signals the end of script assignment for the timepoint.
            self.scripts_received.wait()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        

        value = None
        if location in self.sensor_data:
            value = self.sensor_data[location]
        return value

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device's worker threads."""
        
        for thread in self.thread_list:
            thread.join()


class DeviceThread(Thread):
    """
    A worker thread within a Device's internal thread pool.
    """

    def __init__(self, device, thread_id):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id
        self.neighbours = None

    def initialize_neighbours(self, neighbours):
        """Receives neighbor information from the device's primary thread (ID 0)."""
        
        self.neighbours = neighbours

    def run(self):
        """The main execution loop for the worker thread."""
        
        while True:

            
            # Thread 0 fetches neighbor data and distributes it to other threads in the pool.
            if self.thread_id == 0:
                self.neighbours = self.device.supervisor.get_neighbours()
                for thread in self.device.thread_list:
                    if thread.thread_id != 0:
                        
                        thread.initialize_neighbours(self.neighbours)

            self.device.neighbours_barrier.wait()

            # Supervisor signals shutdown by returning None for neighbours.
            if self.neighbours is None:
                self.device.timepoint_done.wait()
                break

            
            # Wait until all scripts for the timepoint have been assigned.
            self.device.scripts_received.wait()

            
            # Process all scripts assigned to this specific thread.
            for (script, location) in self.device.scripts[self.thread_id]:
                
                self.device.semafor[location].acquire()
                script_data = []
                
                # Gather data from neighbors and self.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                    if len(script_data) == 1:
                        # Skip if only self data is available.
                        self.device.semafor[location].release()
                        continue

                if script_data != []:
                    
                    # Run script and propagate results.
                    result = script.run(script_data)

                    
                    for device in self.neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                
                self.device.semafor[location].release()

            
            
            # Wait at the global barrier, synchronizing all threads of all devices.
            self.device.timepoint_done.wait()
