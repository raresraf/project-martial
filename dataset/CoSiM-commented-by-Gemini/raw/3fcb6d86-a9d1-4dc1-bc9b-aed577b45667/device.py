"""
@file device.py
@brief Defines a distributed device model and a reusable barrier for simulation.

This file contains classes for a simulated `Device` in a network, its corresponding
`DeviceThread` for concurrent execution, and a `ReusableBarrier` for synchronization.
It appears to be a part of a distributed sensor network simulation framework.
"""

from threading import *

class ReusableBarrier():
    """
    A reusable barrier implementation using semaphores.

    This barrier synchronizes a fixed number of threads at a rendezvous point.
    It is reusable, meaning the threads can wait on it multiple times. It employs
    a two-phase signaling mechanism to prevent threads from proceeding until all
    threads have arrived.
    """
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         
 
    def wait(self):
        """
        Causes a thread to wait at the barrier until all participating threads
        have called this method.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        Implements one phase of the two-phase barrier protocol.

        Args:
            count_threads (list): A list containing the current thread count for the phase.
            threads_sem (Semaphore): The semaphore used for signaling in this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            # Pre-condition: The last thread to arrive at the barrier enters this block.
            if count_threads[0] == 0:            
                # Invariant: All threads are waiting on `threads_sem`.
                # This loop releases all waiting threads.
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  # Reset for reuse
        threads_sem.acquire()                    
                                                 
 
class MyThread(Thread):
    """
    A sample thread class to demonstrate the usage of the ReusableBarrier.
    
    This class is likely intended for testing or as an example and does not appear
    to be part of the main device simulation logic.
    """
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
 
    def run(self):
        """
        The main execution logic for the test thread, looping and waiting on a barrier.
        """
        for i in xrange(10):
            self.barrier.wait()
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",


class Device(object):
    """
    Represents a single device in a simulated distributed network.

    Each device has an ID, sensor data, and can execute scripts. It interacts
    with a central supervisor and other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device.

        Args:
            device_id (int): The unique ID of the device.
            sensor_data (dict): A dictionary of the device's sensor readings.
            supervisor: The simulation supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the barrier for synchronization among all devices.

        This method appears intended to be called by a master or root device
        to set up a shared barrier for all devices in the simulation.

        Args:
            devices (list): A list of all device objects in the simulation.
        """
        self.devices = devices
        nrDeviceuri = len(devices)
        
        # Block Logic: This condition `self is devices[0]` checks if the current
        # instance is the first device in the list, which acts as the master
        # for setting up the shared barrier.
        self is devices[0]
        self.bar = ReusableBarrier(nrDeviceuri)
        if self is not devices[0]:
            print "Nu e primu" # Debug print: "Not the first one"
            
    def assign_script(self, script, location):
        """
        Assigns a script to the device to be run at a specific location.

        Args:
            script: The script object to execute.
            location: The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location: The location identifier.

        Returns:
            The data at the location, or None if not available.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data at a given location.

        Args:
            location: The location identifier.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The execution thread for a Device.
    
    This thread contains the main loop that drives the device's behavior
    throughout the simulation.
    """

    def __init__(self, device):
        """
        Initializes the device thread.

        Args:
            device (Device): The parent device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop for the device.
        
        The loop fetches neighbors, waits for scripts, executes them, and then
        synchronizes with other devices using a barrier at the end of each timepoint.
        """
        while True:
            # Block Logic: Fetches the list of neighbors for the current timepoint.
            # A return value of None from the supervisor signals simulation shutdown.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            
            # Waits for the supervisor to signal that all scripts for the
            # current timepoint have been assigned.
            if self.device.timepoint_done.wait() :
            	print "Waiting..." # Debug print
          
            
            if self.device.timepoint_done.clear() :
            	print "Clearing..." # Debug print

       
            # Block Logic: Processes all assigned scripts.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Block Logic: Gathers data from all neighbors at the script's location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Also include data from the current device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Invariant: `script_data` contains all data for the location from the neighborhood.
                if script_data != []:
                    
                    result = script.run(script_data)

                    # Block Logic: Propagates the result of the script execution back
                    # to all neighbors and the current device.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)
         
            
            # Block Logic: Waits at the barrier for all other devices to finish the
            # current timepoint before proceeding to the next one.
            if self.device.devices[0].bar.wait():
            	print "OK" # Debug print