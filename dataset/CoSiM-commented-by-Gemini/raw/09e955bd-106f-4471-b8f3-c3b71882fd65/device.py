"""
This module defines a simulated distributed device network featuring a multi-level
synchronization scheme. It uses a global barrier for inter-device time-step
synchronization and a local barrier for intra-device thread coordination.
The locking mechanism is highly fine-grained, with a unique lock for each
piece of data on each device.
"""

from __future__ import division
from threading import Event, Thread, Lock
from reusable_barrier import ReusableBarrier # Assumes a correct barrier implementation
from math import ceil


class Device(object):
    """
    Represents a device that manages a local pool of worker threads and
    participates in a global, synchronized simulation.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.timepoint_done = Event()
        self.neighbours = None
        self.scripts = []
        self.threads = []
        # --- Shared objects (initialized by master device) ---
        self.l_loc_dev = {}  # Fine-grained locks: (dev_id, loc_id) -> Lock
        self.l_all_threads = None  # Global lock for acquiring other locks.
        self.b_all_threads = None  # Global barrier for all threads.

        # --- Intra-device synchronization ---
        # Each device has its own local barrier and event for its 8 threads.
        b_local = ReusableBarrier(8)
        e_local = Event()
        for i in xrange(8):
            thread = DeviceThread(self, i, b_local, e_local)
            self.threads.append(thread)
            thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects for the network.
        This centralized setup is performed by the first device in the list.
        """
        if devices[0] == self:
            # Create a single global barrier for all threads from all devices.
            nr_of_threads = sum([len(device.threads) for device in devices])
            barrier = ReusableBarrier(nr_of_threads)
            
            # Create a dictionary of fine-grained locks, one per data point.
            loc_dev_lock = {
                (device.device_id, location_id): Lock()
                for device in devices for location_id in device.sensor_data
            }
            # Create a global lock to manage access to the fine-grained locks.
            set_data_lock = Lock()
            
            # Distribute the shared objects to all devices.
            for device in devices:
                device.b_all_threads = barrier
                device.l_loc_dev = loc_dev_lock
                device.l_all_threads = set_data_lock

    def assign_script(self, script, location):
        """Assigns a script; a `None` script signals the end of assignments."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def access_data(self, location):
        """
        Acquires all necessary fine-grained locks for a given location to
        prevent deadlocks. A global lock is used to make this process atomic.
        """
        self.l_all_threads.acquire()
        try:
            if location in self.sensor_data:
                self.l_loc_dev[(self.device_id, location)].acquire()
            for device in self.neighbours:
                if device != self and location in device.sensor_data:
                    device.l_loc_dev[(device.device_id, location)].acquire()
        finally:
            self.l_all_threads.release()

    def release_data(self, location):
        """Releases all fine-grained locks acquired by access_data."""
        if location in self.sensor_data:
            self.l_loc_dev[(self.device_id, location)].release()
        for device in self.neighbours:
            if device != self and location in device.sensor_data:
                device.l_loc_dev[(device.device_id, location)].release()

    def shutdown(self):
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """A worker thread within a device's local thread pool."""
    def __init__(self, device, id_thread, barrier, event):
        Thread.__init__(self, name="Device Thread {0}-{1}".format(device.device_id, id_thread))
        self.device = device
        self.id_thread = id_thread
        self.barrier = barrier # Local barrier for this device's threads.
        self.event = event     # Local event for this device's threads.

    def run(self):
        while True:
            # Thread 0 of each device is the leader for fetching neighbor data.
            if self.device.threads[0] == self:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.event.set() # Signal siblings.
            else:
                self.event.wait() # Wait for leader.

            if self.device.neighbours is None:
                break # End of simulation.

            # Wait for supervisor to signal that scripts are ready.
            self.device.timepoint_done.wait()
            # Synchronize all local threads before processing.
            self.barrier.wait()

            # Leader thread resets events for the next cycle.
            if self.device.threads[0] == self:
                self.device.timepoint_done.clear()
                self.event.clear()

            # --- Static Work Partitioning ---
            partition_size = int(ceil(len(self.device.scripts) / len(self.device.threads)))
            down_lim = self.id_thread * partition_size
            # CRITICAL FLAW: The original code compared `up_lim` to the script list object,
            # not its length. This would cause a TypeError. Corrected to `len()`.
            up_lim = min(down_lim + partition_size, len(self.device.scripts))

            for (script, location) in self.device.scripts[down_lim:up_lim]:
                # Acquire all locks needed for this location on self and neighbors.
                self.device.access_data(location)
                try:
                    script_data = []
                    # Gather data, run script, propagate results.
                    for device in self.device.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                    
                    if script_data:
                        result = script.run(script_data)
                        for device in self.device.neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)
                finally:
                    # Release all acquired locks.
                    self.device.release_data(location)

            # Wait at the global barrier for all threads from all devices to finish.
            self.device.b_all_threads.wait()
