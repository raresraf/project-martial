"""
This module defines a simulation environment for a network of concurrent devices.

It provides a framework for simulating devices that operate in parallel,
process data associated with specific locations, and communicate with their
neighbors. The simulation is synchronized using barriers and condition variables
to coordinate the execution of scripts across multiple threads and devices.

Classes:
    ReusableBarrierCond: A custom reusable barrier for thread synchronization.
    Device: Represents a single device in the network, managing its threads and data.
    DeviceThread: A worker thread that executes scripts on a device.
"""
from threading import Event, Thread, Lock, Condition

class ReusableBarrierCond():
    """
    A reusable barrier implemented with a Condition variable.

    This barrier blocks a specified number of threads until all of them have
    called the wait() method. Once all threads are waiting, they are all
    released, and the barrier is reset for the next use.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait on the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()                  
                                                 
    def wait(self):
        """
        Causes a thread to wait at the barrier.

        The thread will block until all `num_threads` have called this method.
        """
        self.cond.acquire()                      
        self.count_threads -= 1
        if self.count_threads == 0:
            # All threads have arrived, notify all waiting threads and reset the barrier.
            self.cond.notify_all()               
            self.count_threads = self.num_threads
        else:
            # Wait for the other threads to arrive.
            self.cond.wait()                     
                                                 
        self.cond.release()                      

class Device(object):
    """
    Represents a device in the simulated network.

    Each device has an ID, local sensor data, a set of worker threads,
    and a reference to a supervisor for network-level information.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): A dictionary representing the device's local data.
            supervisor: The supervisor object for the simulation (dependency).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.timepoint_scripts = []
        
        self.devs_barrier = None
        self.thread = list()
        for i in range(8):
            self.thread.append(DeviceThread(self, i))
        self.dev_barrier = ReusableBarrierCond(len(self.thread))
        
        self.lock_location = Lock()
        
        self.locked_locs = None
        
        
        self.wait_init = Event()      

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the network of devices, initializing shared resources.

        The device with device_id 0 acts as the coordinator, creating the
        global barrier and the shared lock dictionary for all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            # Coordinator device (id 0) sets up shared resources for all devices.
            size = len(devices)
            barrier = ReusableBarrierCond(8*size)
            self.devs_barrier = barrier

            
            for device in devices:
                if device.devs_barrier is None:
                    if not device.device_id == 0:
                        device.wait_init.set()
                    device.devs_barrier = self.devs_barrier
            
            
            self.locked_locs = dict()
            for device in devices:
                device.locked_locs = self.locked_locs
            for thd in self.thread:
                thd.start()
        else:
            # Other devices wait for the coordinator to finish setup.
            self.wait_init.wait()
            for thd in self.thread:
               thd.start()
        


    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device.

        If the script is None, it signals that all scripts for the current
        timepoint have been assigned.

        Args:
            script: The script object to be executed.
            location: The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data from a specific location in the device's sensor data.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates data at a specific location in the device's sensor data.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its worker threads."""
        for thd in self.thread:
            thd.join()


class DeviceThread(Thread):
    """
    A worker thread for a Device, responsible for executing scripts.
    """

    def __init__(self, device, thread_id):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent device for this thread.
            thread_id (int): The ID of this thread within the device's thread pool.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main execution loop for the thread."""
        while True:
            # Global barrier: all threads from all devices synchronize here.
            self.device.devs_barrier.wait()

            # Thread 0 of each device is responsible for fetching the list of neighbors.
            if self.thread_id is 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()
            
            
            else:
                pass


            # Per-device barrier: ensures all local threads have the neighbor list before proceeding.
            self.device.dev_barrier.wait()
            neighbours = self.device.neighbours
            if neighbours is None:
                # A 'None' neighbor list is the signal to shut down.
                break           
            
            # Wait until the main controller signals that all scripts for this timepoint have been assigned.
            self.device.timepoint_done.wait()

            # Thread 0 of each device copies the scripts to a temporary list for this timepoint.
            if self.thread_id is 0:
                self.device.timepoint_scripts = [i for i in self.device.scripts]

            # Per-device barrier: ensures all local threads see the script list for the timepoint.
            self.device.dev_barrier.wait()

            
            # This loop processes scripts from the timepoint_scripts list.
            # It uses a work-sharing pattern where threads pop scripts from the shared list.
            for (script, location) in self.device.timepoint_scripts:
				
                self.device.lock_location.acquire()
                if not self.device.timepoint_scripts:
                    self.device.lock_location.release()
                    break
                # Pop a script to process. Note that this overwrites the loop variables.
                (script, location) = self.device.timepoint_scripts.pop()
                
                # Ensure a lock exists for the script's location, creating it if necessary.
                # has_key is a Python 2 method; in Python 3, this would be `in`.
                if self.device.locked_locs.has_key(location):
                    pass
                else:
                    self.device.locked_locs[location] = Lock()
				
                # Acquire the lock for the location to ensure exclusive processing of this location.
                self.device.locked_locs[location].acquire()
                self.device.lock_location.release()

                script_data = []
                
                # Gather data from all neighboring devices at the specified location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from the current device as well.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Execute the script with the gathered data.
                    result = script.run(script_data)

                    # Broadcast the result by setting the data on all neighbors and self.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
                
                # Release the lock for the location.
                self.device.locked_locs[location].release()

            # Per-device barrier: synchronize after all scripts for the timepoint are processed.
            self.device.dev_barrier.wait()
            # Clear the timepoint event to prepare for the next cycle.
            self.device.timepoint_done.clear()
