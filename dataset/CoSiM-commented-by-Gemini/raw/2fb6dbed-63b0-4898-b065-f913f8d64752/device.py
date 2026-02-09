"""
A simulation framework for a network of devices with non-thread-safe execution.

This module defines a `Device` class and a custom `MyReusableBarrier`. The system
simulates devices processing data in synchronized time steps. A key feature of
this implementation is that the script execution logic reads and writes data
from multiple devices without any locking, making it susceptible to race conditions.
Synchronization between time steps is handled by a shared, custom-built barrier.
"""

from threading import *


class Device(object):
    """
    Represents a single device in the simulated network.

    The device manages its own data and a control thread. It relies on a 'master'
    device (the first in the list) to create and hold the shared barrier used for
    synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's internal sensor data.
            supervisor: The central supervisor managing the device network.
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
        Initializes the shared barrier for all devices.

        The first device in the `devices` list is designated as the master,
        responsible for creating and storing the single `MyReusableBarrier`
        instance that all other devices will use for synchronization.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.devices=devices
        if self==devices[0]:
            self.bar = MyReusableBarrier(len(devices))
        
        pass

    def assign_script(self, script, location):
        """
        Receives a script from the supervisor for the current timepoint.

        Args:
            script: The script object to execute.
            location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating script execution.

    This thread manages the device's operation in synchronized timepoints. It
    processes all assigned scripts for a timepoint and then waits on a shared
    barrier. The data access within the script execution is not thread-safe.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop, organized into discrete timepoints."""
        while True:
            # Get neighbours for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Wait for the supervisor to finish assigning scripts.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Block Logic: Sequentially process each assigned script for this device.
            # CRITICAL: This block reads and writes to shared `sensor_data` on
            # multiple devices without any locks, creating a race condition if
            # other devices operate on the same location concurrently.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Aggregate data from neighbours and self.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Invariant: Script runs only if there is data to process.
                if script_data != []:
                    result = script.run(script_data)

                    # Broadcast the result to all participants.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

            # Invariant: All devices must wait at the barrier, ensuring they all
            # complete the current timepoint before any can proceed.
            self.device.devices[0].bar.wait()


class MyReusableBarrier():
    """
    A custom, reusable barrier for synchronizing a fixed number of threads.
    
    Intent: To halt a set of threads until all have arrived at the synchronization
    point, using a two-phase protocol to allow for safe reuse in loops.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads to synchronize.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock()       
        self.threads_sem1 = Semaphore(0) 
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        """Causes a thread to block until all threads reach the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire()
         
    def phase2(self):
        """The second phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            self.count_threads1 = self.num_threads
         
        self.threads_sem2.acquire()