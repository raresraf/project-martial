# -*- coding: utf-8 -*-
"""
Models a distributed network of computational devices.

This module implements a simulation where `Device` objects process sensor data
concurrently. This version is characterized by:
- A centralized setup mechanism (`setup_devices`) run by a single device to
  create and distribute a shared `ReusableBarrierSem` and a fixed-size list of locks.
- A `DeviceThread` that employs a manual, inefficient thread-batching strategy.
  Instead of a thread pool, it creates a new thread for every script and then
  starts and joins them in fixed-size chunks (e.g., 8 at a time).
- A coarse-grained lock on the `get_data` method.
"""

from threading import Event, Thread, Lock
import supervisor
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a single computational device in the distributed network.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's local sensor data.
            supervisor: An object that provides neighbor information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a batch of scripts has been assigned.
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        # A shared list of locks for all data locations.
        self.locations = []
        # A lock to protect access to the device's own sensor_data.
        self.get_data_lock = Lock()
        # An event to signal that the initial setup is complete.
        self.ready = Event()
        self.devices = None
        # A reusable barrier for synchronizing all devices between timepoints.
        self.barrier = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Coordinates the setup of all devices in the simulation.

        This method is run by the device with device_id 0. It creates a shared
        barrier and a fixed-size list of locks that are distributed to all devices.
        """
        self.devices = devices
        barrier = ReusableBarrierSem(len(devices))
        # Logic is centralized to the master device (device_id 0).
        if self.device_id == 0:
            i = 0
            # Note: A fixed-size list of 150 locks is created, which is brittle.
            while i < 150:
                self.locations.append(Lock())
                i = i + 1

            # Distribute the shared barrier and locks to all devices.
            for dev in devices:
                dev.barrier = barrier
                dev.locations = self.locations
                # Signal to each device that setup is complete and they can proceed.
                dev.ready.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals the end of a script batch.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts for the current timepoint
            # have been assigned.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data, protected by a lock."""
        with self.get_data_lock:
            return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a Device.

    This thread implements a manual batching system for executing scripts, creating
    a new thread for each script and managing their execution in fixed-size chunks.
    """
    def __init__(self, device):
        Thread.__init__(self, name="%d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the device thread.
        """
        # Wait until the initial device setup is complete.
        self.device.ready.wait()

        while True:
            # Get the current list of neighbors from the supervisor.
            neigh = self.device.supervisor.get_neighbours()
            # A None response signals simulation shutdown.
            if neigh is None:
                break

            # Pre-condition: Wait for the signal that a batch of scripts is ready.
            self.device.script_received.wait()
            self.device.script_received.clear()

            rem_scripts = len(self.device.scripts)
            threads = []
            
            # Create a new thread for every assigned script.
            i = 0
            while i < rem_scripts:
                threads.append(MyThread(self.device, neigh, self.device.scripts, i))
                i = i + 1
            
            # Inefficient Manual Thread Batching:
            # This logic manually manages thread execution in chunks rather than
            # using a more efficient thread pool.
            if rem_scripts < 8:
                # If fewer than 8 scripts, start and join them all at once.
                for thr in threads:
                    thr.start()
                for thr in threads:
                    thr.join()
            else:
                # If 8 or more, process them in chunks of 8.
                pos = 0
                while rem_scripts != 0:
                    if rem_scripts > 8:
                        # Start a chunk of 8 threads.
                        for i in range(pos, pos + 8):
                            threads[i].start()
                        # Wait for this chunk to complete before starting the next one.
                        for i in range(pos, pos + 8):
                            threads[i].join()
                        pos = pos + 8
                        rem_scripts = rem_scripts - 8
                    else:
                        # Process the final, smaller chunk.
                        for i in range(pos, pos + rem_scripts):
                            threads[i].start()
                        for i in range(pos, pos + rem_scripts):
                            threads[i].join()
                        rem_scripts = 0
            
            # Synchronize with all other devices before starting the next timepoint.
            self.device.barrier.wait()


class MyThread(Thread):
    """
    A short-lived worker thread responsible for executing a single script.
    """
    def __init__(self, device, neigh, scripts, index):
        Thread.__init__(self, name="%d" % device.device_id)
        self.device = device
        self.neigh = neigh
        self.scripts = scripts
        self.index = index

    def run(self):
        """
        The execution logic for a single script.
        """
        (script, loc) = self.scripts[self.index]
        # Acquire the specific lock for the data location.
        self.device.locations[loc].acquire()
        info = []
        
        # Invariant: Gather data from neighbors.
        for neigh_iter in self.neigh:
            aux_data = neigh_iter.get_data(loc)
            if aux_data is not None:
                info.append(aux_data)
        
        # Gather data from the local device.
        aux_data = self.device.get_data(loc)
        if aux_data is not None:
            info.append(aux_data)
        
        # Invariant: Only execute if data was gathered.
        if info != []:
            result = script.run(info)
            # Broadcast the result to neighbors and the local device.
            for neigh_iter in self.neigh:
                neigh_iter.set_data(loc, result)
                self.device.set_data(loc, result)
        
        # Release the lock for the location.
        self.device.locations[loc].release()
