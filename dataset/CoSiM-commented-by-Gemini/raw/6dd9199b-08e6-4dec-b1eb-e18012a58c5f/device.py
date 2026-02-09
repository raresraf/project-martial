"""
@file device.py
@brief Defines a device model with a complex, two-barrier synchronization scheme.

This file implements a device simulation where the main supervisor thread
actively participates in a barrier synchronization (`barrier2`) to signal the
start of a timepoint's execution. A second barrier (`barrier1`) is also used
by the device threads. This design is highly unusual.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrier

class Device(object):
    """
    Represents a device in the simulation, participating in a two-barrier
    synchronization protocol.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event() # This event seems to be unused.


        self.thread = DeviceThread(self)
        self.thread.start()

        # Two separate barriers for different synchronization points.
        self.barrier1 = None
        self.barrier2 = None

        self.location_lock = []

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes two shared barriers and location-specific locks.

        Executed by the root device (ID 0).

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            self.barrier1 = ReusableBarrier(len(devices))
            # barrier2 includes the supervisor thread, hence len(devices) + 1.
            # However, the code uses len(devices), which might be a bug if the
            # supervisor also waits on it. Let's assume supervisor doesn't wait.
            # Based on assign_script, the supervisor *does* wait, so this is likely a bug.
            self.barrier2 = ReusableBarrier(len(devices))

            for device in devices:
                device.barrier1 = self.barrier1
                device.barrier2 = self.barrier2

            # Block Logic: Create a lock for each possible location index.
            max_loc = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > max_loc:
                        max_loc = location
            while max_loc >= 0:
                self.location_lock.append(Lock())
                max_loc = max_loc - 1
            
            # Distribute the list of location locks to all devices.
            for device in devices:
                device.location_lock = self.location_lock

    def assign_script(self, script, location):
        """
        Assigns a script and participates in barrier synchronization.
        
        @note The call to `barrier2.wait()` here is highly unconventional, as it
              makes the main supervisor thread a participant in the barrier,
              blocking it until all device threads also reach `barrier2`.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal device threads that script assignment is complete.
            self.script_received.set()
            # The supervisor thread itself waits on the barrier.
            self.barrier2.wait()

    def get_data(self, location):
        """Retrieves sensor data. This access is not synchronized by this method."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data. This access is not synchronized by this method."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, implementing an unusual
    double-barrier wait at the start of its loop.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop."""
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            # --- Start of Synchronization ---
            # 1. Wait for the 'script_received' event set by the supervisor.
            self.device.script_received.wait()
            # 2. Wait on barrier2, syncing with the supervisor and other devices.
            self.device.barrier2.wait()
            # 3. Wait on barrier1, for reasons that are unclear in the logic.
            #    This might be a bug or an attempt to create a second sync point.
            self.device.barrier1.wait()

            if neighbours is None:
                break

            # Block Logic: Process all assigned scripts serially.
            for (script, location) in self.device.scripts:
                # Pre-condition: Acquire lock for the specific location.
                self.device.location_lock[location].acquire()

                script_data = []
                # Gather data from neighbors and self.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Invariant: Data is gathered and ready for script execution.
                if script_data != []:
                    result = script.run(script_data)
                    # Propagate results back.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                self.device.location_lock[location].release()
            # @note There is no final barrier here, meaning a fast device could loop
            # around and call get_neighbours() for the next timepoint before other
            # devices have finished processing the current one.