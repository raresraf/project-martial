"""
@file device.py
@brief Defines a device model using a dynamic registration barrier.

This file implements a simulation device where each device instance dynamically
registers its thread with a single, class-level `ReusableBarrier` upon
initialization.

@warning This implementation is critically flawed as it contains no locking
         mechanism for data access (`get_data`, `set_data`), which will lead
         to race conditions in a multi-threaded environment.
"""

from threading import Event, Thread
import ReusableBarrier

class Device(object):
    """
    Represents a device that registers with a global barrier upon creation.
    """
    # A single, class-level barrier instance is created when the class is defined.
    reusable_barrier = ReusableBarrier.ReusableBarrier()

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device and registers it with the class-level barrier.
        
        Args:
            device_id (int): The unique ID for the device.
            sensor_data (dict): A dictionary of the device's sensor readings.
            supervisor: The central simulation supervisor.
        """
        # Each new device increments the number of threads the barrier expects.
        Device.reusable_barrier.add_thread()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event() # This event is unused.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        This setup method is a no-op in this implementation, as barrier
        setup is handled dynamically in the Device constructor.
        """
        pass

    def assign_script(self, script, location):
        """Assigns a script to be executed."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that script assignment is complete.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data. This read is not synchronized."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data. This write is not synchronized."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a Device.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop.
        
        @warning Data access within this loop is not synchronized. When multiple
                 devices run scripts that access shared locations, race conditions
                 will occur.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait at the shared barrier for all devices to be ready.
            Device.reusable_barrier.wait();
            
            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()
            self.device.script_received.clear();

            # Block Logic: Process all assigned scripts serially in this thread.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Gather data from neighbors (UNSAFE - NO LOCKING).
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    result = script.run(script_data)

                    # Propagate result to neighbors (UNSAFE - NO LOCKING).
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)