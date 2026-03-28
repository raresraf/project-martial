"""
This module provides another variation of the device simulation framework, featuring
a manual, batch-based approach to thread management for script execution.

The design consists of a `Device` class for state, a main `DeviceThread` for
orchestration, and an `ExecutorThread` for running individual scripts. The main
thread creates a thread for each script but then runs them in fixed-size batches
of 8. This implementation contains several notable bugs, including a flawed locking
strategy for data access. It also appears to be written for Python 2, given the
use of `xrange`.
"""

from threading import Event, Thread, Lock
from mybarrier import ReusableBarrier

class Device(object):
    """
    Represents a device node in the simulation.

    This class holds the device's state and is managed by a `DeviceThread`.
    It uses a fixed-size list of locks for locations and a flawed locking
    mechanism for accessing sensor data.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        
        self.reusable_barrier = None 
        self.devices = [] 
        # NOTE: A fixed-size list for locks limits location IDs to 0-99.
        self.location_lock = [] 
        
        # WARNING: Using separate locks for get and set operations is a critical
        # bug. It does not prevent race conditions between readers and writers.
        # A single lock should be used to protect `sensor_data`.
        self.set_lock = Lock() 
        self.get_lock = Lock() 
        
        self.ready = Event() # Signals that device setup is complete.
                             
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources from the coordinator device (ID 0).
        """
        self.devices = devices
        # Pre-condition: This global setup is only performed by the coordinator device.
        if self.device_id == 0:
            reusable_barrier = ReusableBarrier(len(devices))
            for _ in range(100):
                self.location_lock.append(Lock())

            # Distribute shared resources to all other devices.
            for dev in devices:
                dev.reusable_barrier = reusable_barrier
                dev.location_lock = self.location_lock
                # Signal to each device that it can start its main loop.
                dev.ready.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A None script signals the end of assignments.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves data for a given location.
        WARNING: This method is not thread-safe due to using a separate lock from `set_data`.
        """
        with self.get_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets data for a given location.
        WARNING: This method is not thread-safe due to using a separate lock from `get_data`.
        """
        with self.set_lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """
    The main orchestration thread for a device, managing script execution in batches.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main simulation loop.
        """
        # Wait until the `setup_devices` method has been called and resources are ready.
        self.device.ready.wait()
        
        while True:
            thread_list = []
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Terminate signal from supervisor.

            # Wait for the signal that all scripts for the timepoint have been assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Create an executor thread for each assigned script.
            pos = 0
            for _ in self.device.scripts:
                thread_list.append(ExecutorThread(self.device, self.device.scripts, neighbours, pos))
                pos += 1

            # Block Logic: Manual thread pool execution in batches of 8.
            # This is a complex way to limit concurrency.
            scripts_left = len(self.device.scripts)
            current_pos = 0
            
            if scripts_left < 8:
                for thread in thread_list:
                    thread.start()
                for thread in thread_list:
                    thread.join()
            else:
                # Note: `xrange` is from Python 2. In Python 3, `range` is equivalent.
                while scripts_left >= 8:
                    # Start a batch of 8 threads.
                    for i in range(current_pos, current_pos + 8):
                        thread_list[i].start()
                    # Wait for the batch to complete.
                    for i in range(current_pos, current_pos + 8):
                        thread_list[i].join()
                    current_pos += 8
                    scripts_left -= 8

                # Process any remaining scripts.
                for i in range(current_pos, current_pos + scripts_left):
                    thread_list[i].start()
                for i in range(current_pos, current_pos + scripts_left):
                    thread_list[i].join()
            
            # Synchronize with all other devices before starting the next timepoint.
            self.device.reusable_barrier.wait()

class ExecutorThread(Thread):
    """A worker thread responsible for executing a single script."""
    def __init__(self, device, scripts, neighbours, pos):
        Thread.__init__(self, name="Executor Thread %d" % device.device_id)
        self.device = device
        self.scripts = scripts
        self.neighbours = neighbours
        self.pos = pos # The index of the script this thread should execute.

    def run(self):
        (script, location) = self.scripts[self.pos]
        
        # Acquire a location-specific lock to prevent data races.
        self.device.location_lock[location].acquire()
        
        script_data = []
        # Gather data from neighbors.
        for dev in self.neighbours:
            data = dev.get_data(location)
            if data is not None:
                script_data.append(data)
        # Gather data from the local device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
            
        if script_data:
            result = script.run(script_data)
            # Propagate results to all involved devices.
            for dev in self.neighbours:
                dev.set_data(location, result)
                # BUG: The local device's data is set repeatedly inside the loop.
                # This should be moved outside the for loop.
                self.device.set_data(location, result)
                
        self.device.location_lock[location].release()