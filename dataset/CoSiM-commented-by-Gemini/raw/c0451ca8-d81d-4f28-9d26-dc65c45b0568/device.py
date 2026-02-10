from threading import Event, Thread
# Assumes a 'ReusableBarrier.py' module with a ReusableBarrier class that
# supports dynamic thread registration via an `add_thread` method.
import ReusableBarrier

class Device(object):
    """
    Represents a device in a simulation that uses a single, class-level
    barrier for synchronization.

    Architectural Role: This model uses a highly simplified, decentralized setup.
    A single `ReusableBarrier` instance is shared across all `Device` objects as
    a static class attribute. Each device registers its thread with the barrier
    upon initialization.

    Warning: This implementation is critically flawed as it contains no locking
    mechanisms for shared data access (`get_data`, `set_data`), which will
    lead to severe race conditions in a multi-threaded environment.
    """
    # A single barrier instance shared by all Device objects.
    reusable_barrier = ReusableBarrier.ReusableBarrier()

    def __init__(self, device_id, sensor_data, supervisor):
        # Each new device registers its thread with the class-level barrier.
        Device.reusable_barrier.add_thread()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        This setup method is a no-op in this architecture, as synchronization
        objects are managed at the class level.
        """
        pass

    def assign_script(self, script, location):
        """Assigns a script or triggers the script processing phase."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A `None` script signals that script assignment is complete.
            self.script_received.set()

    def get_data(self, location):
        """
        Retrieves data. Warning: This operation is NOT thread-safe and is
        prone to race conditions from concurrent access by other devices.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates data. Warning: This operation is NOT thread-safe and is
        prone to race conditions.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()

class DeviceThread(Thread):
    """
    The main control thread for a device, featuring serial script execution
    and a confusing synchronization pattern.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Supervisor signals shutdown.

            # Block Logic: Unusual two-stage wait.
            # 1. All threads wait on the global barrier.
            Device.reusable_barrier.wait()
            # 2. Each thread then waits on its own event. This implies the supervisor
            # must both manage the barrier and set events for every device.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Block Logic: Serial, non-thread-safe script execution.
            # All data access in this block is subject to race conditions.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Unprotected reads from neighbors.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    result = script.run(script_data)

                    # Unprotected writes to neighbors and self.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)