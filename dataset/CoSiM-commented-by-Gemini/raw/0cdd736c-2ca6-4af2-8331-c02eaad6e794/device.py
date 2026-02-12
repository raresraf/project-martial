# -*- coding: utf-8 -*-
"""
This module simulates a network of devices with a complex, decentralized
setup protocol and a thread-per-task execution model.

The architecture includes:
- A decentralized, racy master-election process for initializing shared
  resources (a barrier and a list of locks).
- A "thread-per-task" model where each device's main thread spawns a new
  "SlaveThread" for each script.
- A "complete local work, then sync globally" time-step model, using thread
  joins locally and a reusable barrier globally.

Classes:
    Device: A node in the network.
    DeviceThread: The main control loop for a device.
    SlaveThread: A worker thread that executes one script.
"""

from threading import Event, Thread, Lock
# Assumes ReusableBarrierSem is defined in a 'barrier' module.
from barrier import ReusableBarrierSem

class Device(object):
    """
    Represents a device that participates in a decentralized setup protocol
    to elect a master and share synchronization objects.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        # --- State and Events ---
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.script_received = Event()
        self.timepoint_done = Event()
        self.devices_list = []
        self.slavet_started = [] # List of worker threads for the current timepoint.

        # --- Master/Slave Election and Setup ---
        self.master_node = True  # Assumes it's the master until proven otherwise.
        self.master_id = None
        self.master_barrier = Event() # Signals that the master has finished setup.
        self.locks_ready = Event()    # Signals that the master has created locks.
        
        # --- Synchronization Objects ---
        self.lock = Lock() # A per-device lock for its own set_data method.
        self.barrier = None # The global, shared barrier.
        self.data_lock = [None] * 100 # Per-location locks, copied from the master.

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        A decentralized protocol to elect a master and share resources.
        
        Note: This implementation is not thread-safe and depends on a
        sequential execution order to avoid race conditions where multiple
        devices might elect themselves as master.
        """
        # --- Slave Discovery ---
        # Check if another device has already become the master.
        for device in devices:
            if device is not None and device.master_id is not None:
                self.master_id = device.master_id
                self.master_node = False
                break
        
        # --- Path for Slaves ---
        if not self.master_node:
            for device in devices:
                if device is not None and device.device_id == self.master_id:
                    # Wait for the master to finish its setup.
                    device.master_barrier.wait()
                    if self.barrier is None:
                        self.barrier = device.barrier # Copy the shared barrier.
                self.devices_list.append(device)
        # --- Path for Master ---
        else:
            self.barrier = ReusableBarrierSem(len(devices))
            self.master_id = self.device_id
            # The master creates the per-location locks.
            self.data_lock = [Lock() for _ in range(100)]
            self.locks_ready.set()
            self.master_barrier.set()
            # The master ensures all devices know about the shared barrier.
            for device in devices:
                if device is not None:
                    device.barrier = self.barrier
                    self.devices_list.append(device)

    def assign_script(self, script, location):
        """
        Assigns a script and lazily copies the shared lock dictionary.
        This lazy copy is unusual and potentially inefficient.
        """
        if script is not None:
            self.scripts.append((script, location))

            # Find the master and wait for its locks to be ready.
            for device in self.devices_list:
                if device.device_id == self.master_id:
                    device.locks_ready.wait()
                    # Copy the lock list from the master.
                    self.data_lock = device.data_lock
                    break
            self.script_received.set()
        else:
            # Signal that all scripts for the timepoint are assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Thread-safely sets data using a per-device lock."""
        with self.lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        self.thread.join()


class DeviceThread(Thread):
    """The main control loop for a device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop, using a "thread-per-task" model.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None: # Shutdown signal.
                break
            
            # 1. Wait for the supervisor to assign all scripts.
            self.device.timepoint_done.wait()

            # 2. Create and start a new "SlaveThread" for each script.
            self.device.slavet_started = [
                SlaveThread(self.device, neighbours, location, script)
                for script, location in self.device.scripts
            ]
            for thread in self.device.slavet_started:
                thread.start()

            # 3. Wait for all local worker threads to complete.
            for thread in self.device.slavet_started:
                thread.join()
            
            self.device.slavet_started = []
            self.device.timepoint_done.clear()

            # 4. Wait at the global barrier to sync with other devices.
            self.device.barrier.wait()


class SlaveThread(Thread):
    """A worker thread that executes one script."""

    def __init__(self, device, neighbours, location, script):
        Thread.__init__(self, name="Slave Thread %d" % device.device_id)
        self.device = device
        self.neighbours = neighbours
        self.location = location
        self.script = script

    def run(self):
        """Acquires a per-location lock and executes the script logic."""
        # This assumes `location` is an integer index.
        self.device.data_lock[self.location].acquire()
        try:
            if self.neighbours is not None:
                script_data = []
                # --- Data Gathering ---
                for device in self.neighbours:
                    data = device.get_data(self.location)
                    if data is not None:
                        script_data.append(data)
                data = self.device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

                # --- Execution and Dissemination ---
                if script_data:
                    result = self.script.run(script_data)
                    for device in self.neighbours:
                        device.set_data(self.location, result)
                    self.device.set_data(self.location, result)
        finally:
            self.device.data_lock[self.location].release()