"""
This module provides a framework for simulating a network of distributed devices.

It defines a `Device` class that operates concurrently, managed by a `DeviceThread`.
The simulation proceeds in synchronized time steps. A master device (`device_id == 0`)
is responsible for setting up shared synchronization objects (`Condition` variables
for location-based locking and a shared `Barrier`) which are distributed to all
devices. Worker threads are spawned to execute computational scripts on sensor data
gathered from a device and its neighbors.
"""

from threading import Event, Thread, Condition
from barrier import Barrier
from worker import Worker


class Device(object):
    """
    Represents a single device in the distributed network simulation.

    Each device manages its own sensor data, executes assigned scripts, and
    synchronizes with other devices at discrete time steps.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device instance.

        Args:
            device_id (int): The unique identifier for this device.
            sensor_data (dict): A mapping of locations to sensor values.
            supervisor (Supervisor): A reference to the central supervisor object.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event used to signal that all scripts for a time step have been assigned.
        self.script_received = Event()
        self.scripts = []
        # List of shared Condition objects used for location-based locking.
        self.elocks = []
        # The shared barrier for synchronizing all devices at the end of a time step.
        self.barrier = Barrier(0)
        self.thread = DeviceThread(self)
        self.devices = []

    
    def __str__(self):
        return "Device %d" % self.device_id

    
    def create_conditions(self, condition_number):
        """
        Initializes a set of Condition objects to be used as locks.
        
        Note: This method appears to be unused in the current control flow.
        The primary lock creation happens in `setup_devices`.
        """
        for _ in xrange(condition_number):
            condition_location = Condition()
            self.elocks.append(condition_location)


    def setup_devices(self, devices):
        """
        Performs centralized setup for all devices in the simulation.

        This method must be called by a single "master" device (e.g., device_id == 0).
        It determines the required number of locks, creates shared Condition objects
        and a shared Barrier, distributes them to all devices, and starts each
        device's main thread.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        length = len(devices)
        
        barrier = Barrier(length)

        # Pre-condition: This block should only be executed by a single master device.
        if self.device_id == 0:
            condition_number = 0

            # Block Logic: Determine the total number of unique data locations
            # to decide how many locks are needed.
            for device in devices:
                for location in device.sensor_data.keys():
                    if location > condition_number:
                        condition_number = location
            
            # The number of locks should be one more than the max location index.
            condition_number += 1

            # Create the shared Condition objects.
            for _ in xrange(condition_number):
                condition_location = Condition()
                self.elocks.append(condition_location)

            # Block Logic: Distribute shared synchronization objects and start threads.
            for device in devices:
                # Assign the shared barrier instance.
                if barrier is not None:
                    device.barrier = barrier
                
                # Assign the shared list of locks.
                for j in xrange(condition_number):
                    device.elocks.append(self.elocks[j])
                
                # Start the device's main thread only after setup is complete.
                device.thread.start()

    def assign_script(self, script, location):
        """
        Assigns a computational script to the device.

        Args:
            script (Script): The script to execute. If None, it signals that
                             all scripts for the time step have been assigned.
            location (int): The data location the script targets.
        """
        # A 'None' script is the signal to unblock the DeviceThread and start processing.
        return self.script_received.set() if script is None \
            else self.scripts.append((script, location))


    def get_data(self, location):
        """
        Retrieves data from a specific sensor location on this device.
        """
        return self.sensor_data[location] if location in self.sensor_data \
            else None

    def set_data(self, location, data):
        """
        Updates data at a specific sensor location on this device.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Gracefully shuts down the device's main thread.
        """
        self.thread.join()



class DeviceThread(Thread):
    """
    The main, long-running thread for a Device.

    This thread orchestrates the device's participation in the simulation by
    spawning and managing Worker threads for each assigned script and synchronizing
    with other devices at a global barrier.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []


    def run(self):
        """
        The main control loop for the device.
        """
        while True:
            # The loop terminates when the supervisor indicates no more neighbors.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block until the supervisor signals that all scripts for the
            # current timepoint have been assigned.
            self.device.script_received.wait()

            # Block Logic: Create a worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                worker_thread = Worker(self.device, script, location, neighbours)
                self.workers.append(worker_thread)

            length = len(self.workers)

            # Block Logic: Start all worker threads and wait for them to complete.
            # This ensures all script computations for the time step are finished
            # before proceeding to the global barrier.
            for i in range(length):
                self.workers[i].start()
            
            for i in range(length):
                self.workers[i].join()
            
            # Clean up workers for the next iteration.
            self.workers = []

            # Reset the event and wait at the barrier for all other devices.
            self.device.script_received.clear()
            self.device.barrier.wait()

from threading import Thread


class Worker(Thread):
    """
    A short-lived thread that executes a single computational script.
    """
    def __init__(self, device, script, location, neighbours):
        """
        Initializes the worker thread.

        Args:
            device (Device): The parent device.
            script (Script): The script object to execute.
            location (int): The data location to operate on.
            neighbours (list): A list of neighboring Device objects.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def solve(self):
        """
        Contains the core logic for executing the script.
        """
        # Invariant: Acquire the lock for this specific location to ensure
        # exclusive access to the data during the read-modify-write operation.
        self.device.elocks[self.location].acquire()
        script_data = []

        # Block Logic: Gather data from the parent device and all its neighbors
        # at the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: Execute the script if data was found and disseminate the result.
        if script_data != []:
            # The script performs a computation on the aggregated data.
            result = self.script.run(script_data)
            
            # The result is written back to the parent and all neighbors,
            # updating the state for this location across the local neighborhood.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)

        self.device.elocks[self.location].release()

    def run(self):
        """
        The entry point for the thread's execution.
        """
        self.solve()
