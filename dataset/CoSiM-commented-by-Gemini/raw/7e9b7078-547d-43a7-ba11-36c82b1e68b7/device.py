"""
@file device.py
@brief Defines a complex device model using an internal pool of worker threads.

This file implements a simulation device that uses a fixed pool of long-running
worker threads (`DeviceThread`) to parallelize script execution within a single
device. It uses a complex and confusing system of three different barriers for
synchronization, both within the device and between devices.

@warning The synchronization logic is deeply flawed. The main cross-device
         barrier is only waited on by one of the worker threads, and there is no
         final barrier at the end of the loop to ensure all devices have completed
         the timepoint. This will lead to severe race conditions.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond

class Device(object):
    """
    Represents a device that manages its own internal pool of worker threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.bariera = None # Main barrier, shared between all devices.
        self.timepoint_done = Event()
        self.threads = []
        self.nr_threads = 8
        self.locks = {} # Shared map of location-specific locks.
        # An "inner" barrier, shared only by the worker threads of this device.
        self.bariera_interioara = ReusableBarrierCond(self.nr_threads)

        # Create and start a fixed pool of 8 worker threads upon initialization.
        for index in xrange(0, self.nr_threads):
            thread = DeviceThread(self, index)
            self.threads.append(thread)

        for thread in self.threads:
            thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared barrier and location-specific locks.
        """
        if self.device_id == 0:
            # Root device creates the main cross-device barrier.
            self.bariera = ReusableBarrierCond(len(devices))
            for dev in devices:
                dev.bariera = self.bariera
            
            # Discover all unique locations and create a shared lock for each.
            max_location = 0
            for device in devices:
                for location in device.sensor_data:
                    if location > max_location:
                        max_location = location
            
            for location in xrange(0, max_location + 1):
                self.locks[location] = Lock()
            for device in devices:
                device.locks = self.locks

    def assign_script(self, script, location):
        """Assigns a script to be executed."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of script assignment for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data. This read is not synchronized."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data. This write is not synchronized."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its worker threads."""
        for thread in self.threads:
            thread.join()

class DeviceThread(Thread):
    """
    A worker thread within a Device's internal thread pool.

    @warning The synchronization logic in the `run` method is convoluted and buggy.
    """
    def __init__(self, device, index):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.index = index # The unique index (0-7) of this thread within the device.

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            # --- Phase 1: Get Neighbors ---
            self.device.bariera_interioara.wait() # Sync all 8 local threads.
            # One thread is designated to get the neighbors list.
            if self.index == 0:
                self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.bariera_interioara.wait() # Sync again to ensure all threads see the list.
            
            if self.device.neighbours is None:
                break

            # --- Phase 2: Synchronization with Supervisor and other Devices (BUGGY) ---
            # Only the designated leader thread participates in cross-device sync.
            if self.index == 0:
                self.device.timepoint_done.wait()
                # This thread now waits for all other DEVICES, but the other 7 threads
                # of this device are not waiting here.
                self.device.bariera.wait()
            
            self.device.bariera_interioara.wait() # Sync all 8 local threads again.

            # --- Phase 3: Script Execution ---
            # Each thread processes a subset of the scripts using a modulo operation.
            for index in xrange(0, len(self.device.scripts)):
                if self.index == index % self.device.nr_threads:
                    (script, location) = self.device.scripts[index]
                    
                    # Correctly use the location-specific lock.
                    self.device.locks[location].acquire()
                    script_data = []
                    
                    for device in self.device.neighbours:
                        data = device.get_data(location)
                        if data is not None:
                            script_data.append(data)
                    
                    data = self.device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                    if script_data != []:
                        result = script.run(script_data)
                        for device in self.device.neighbours:
                            device.set_data(location, result)
                        self.device.set_data(location, result)
                    
                    self.device.locks[location].release()
            
            if self.index == 0:
                self.device.timepoint_done.clear()

            # @warning A final cross-device barrier wait is missing here. This means
            # this device's threads could loop around and start the next timepoint
            # before other devices have finished the current one.