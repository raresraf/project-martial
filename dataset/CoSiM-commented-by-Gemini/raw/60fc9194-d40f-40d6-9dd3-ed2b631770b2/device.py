"""
@file device.py
@brief Defines a device model with a complex, multi-level synchronization scheme.

This file implements a simulation device using two thread classes with potentially
confusing names:
- `DeviceThreadPool`: The main control thread for a device.
- `DeviceThread`: A worker thread spawned to execute a single script.

Synchronization is managed via a leader device that distributes a main barrier,
a global lock, and a map of location-specific locks. An additional "inner"
barrier is used to coordinate the start of a timepoint's execution.
"""

from threading import Thread, Lock
from barrier import ReusableBarrier


class Device(object):
    """
    Represents a device in the simulation, coordinating script execution through
    a hierarchy of threads, locks, and barriers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.thread = DeviceThreadPool(self) # The main control thread for this device.
        
        self.barrier = None # The main barrier, shared across all devices.
        
        # An internal barrier to sync the main thread with the script assignment process.
        self.inner_barrier = ReusableBarrier(2)
        
        self.lock = None # A global lock, shared across all devices.
        
        self.inner_lock = Lock() # A lock specific to this device instance.
        
        # A map of location-specific locks, shared across all devices.
        self.lock_map = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Identifies a leader device to initialize and distribute shared resources.

        The device with the minimum ID is chosen as the leader. It creates and
        sets a shared barrier, a shared global lock, and a shared lock map for
        all devices, and then starts their main control threads.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        device_ids = [device.device_id for device in devices]
        leader_id = min(device_ids)

        # Pre-condition: Only the leader device executes this setup logic.
        if self.device_id == leader_id:
            barrier = ReusableBarrier(len(devices))
            lock = Lock()
            lock_map = {}
            # Invariant: Distribute the same shared objects to all devices.
            for device in devices:
                device.set_barrier(barrier)
                device.set_lock(lock)
                device.set_lock_map(lock_map)
                device.thread.start()

    def set_barrier(self, barrier):
        """Sets the shared main barrier."""
        self.barrier = barrier

    def set_lock(self, lock):
        """Sets the shared global lock."""
        self.lock = lock

    def set_lock_map(self, lock_map):
        """Sets the shared map of location-specific locks."""
        self.lock_map = lock_map

    def assign_script(self, script, location):
        """
        Assigns a script and synchronizes with the main thread to start execution.

        When a script is assigned, a location-specific lock is created if needed.
        When `script` is None, it signals the end of assignment by waiting on an
        `inner_barrier`, which unblocks the device's main `DeviceThreadPool` thread.
        """
        if script is not None:
            self.scripts.append((script, location))

            # Use the global lock to safely update the shared lock_map.
            with self.lock:
                if location not in self.lock_map:
                    self.lock_map[location] = Lock()
        else:
            # Signal to the `DeviceThreadPool` that script assignment is done.
            self.inner_barrier.wait()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Updates sensor data for a given location, using an instance-specific lock.

        @note Using a non-shared `inner_lock` here is a potential race condition,
              as other devices may attempt to modify the same data concurrently
              without acquiring this specific lock.
        """
        if location in self.sensor_data:
            with self.inner_lock:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThreadPool(Thread):
    """
    The main control thread for a device.
    
    @note The name is misleading; this class manages a pool of `DeviceThread`
          (worker) threads but is not itself a thread pool.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop.
        """
        while True:
            
            # Acquire the global lock to safely get the list of neighbors.
            with self.device.lock:
                neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                break

            # Wait for the `assign_script` method to signal that assignment is done.
            self.device.inner_barrier.wait()

            threads = []

            # Block Logic: Spawn and start a worker thread (`DeviceThread`) for each script.
            for (script, location) in self.device.scripts:
                thread = DeviceThread(self.device, script, location, neighbours)
                thread.start()
                threads.append(thread)

            # Wait for all worker threads to complete.
            for thread in threads:
                thread.join()

            # --- Main Barrier ---
            # Wait for all devices in the simulation to finish their timepoint.
            self.device.barrier.wait()


class DeviceThread(Thread):
    """
    A worker thread that executes a single script.

    @note The name is misleading; this is a worker thread, not the main control thread.
    """

    def __init__(self, device, script, location, neighbours):

        Thread.__init__(self, name="Device Thread %d" % device.device_id)

        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        Executes the script for its assigned location, using a location-specific lock.
        """
        # Pre-condition: Acquire the lock for the specific location from the shared map.
        with self.device.lock_map[self.location]:

            script_data = []
            
            # Block Logic: Gather data from neighbors and the local device.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Invariant: Data is gathered and ready for script execution.
            if len(script_data) != 0:
                
                result = self.script.run(script_data)

                # Propagate result back to all devices.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                self.device.set_data(self.location, result)