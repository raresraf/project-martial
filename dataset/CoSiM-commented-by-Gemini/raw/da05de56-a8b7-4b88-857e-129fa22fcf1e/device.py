"""
This module implements a simulated device for a concurrent system.

It features a custom barrier implementation that uses static counters and is
incrementally built as devices are instantiated. A critical design feature is
the lack of any locking mechanism during script execution, which makes it prone
to race conditions.
"""

from threading import Event, Thread, Condition


class Barrier():
    """
    A custom barrier for synchronizing a dynamic number of threads.

    This implementation uses a Condition variable and static counters. It is
    not a fully reusable, two-phase barrier, and its simple counter reset
    mechanism may be prone to race conditions if threads re-enter the wait
    cycle at different times.
    """
    # Static variables shared across all instances of the Barrier.
    num_threads = 0
    count_threads = 0

    def __init__(self):
        
        self.cond = Condition()
        self.thread_event = Event()

    def wait(self):
        """
        Blocks the calling thread until all registered threads have reached
        the barrier.
        """
        self.cond.acquire()
        Barrier.count_threads -= 1

        if Barrier.count_threads == 0:
            # Last thread notifies all waiting threads.
            self.cond.notify_all()
            # The counter is reset to make the barrier "reusable".
            Barrier.count_threads = Barrier.num_threads
        else:
            self.cond.wait()

        self.cond.release()

    @staticmethod
    def add_thread():
        """Statically increments the number of threads the barrier waits for."""
        Barrier.num_threads += 1
        Barrier.count_threads = Barrier.num_threads


class Device(object):
    """
    Represents a single device that registers with a global barrier upon creation.
    """
    # A single, static barrier instance is shared by all Device objects.
    barrier = Barrier()

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device and registers it with the static barrier.
        """
        # Each new device increments the barrier's thread count.
        Device.barrier.add_thread()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """A no-op setup method, as setup is handled in the constructor."""
        
        
        pass

    def assign_script(self, script, location):
        """Assigns a script to the device for execution."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # If no script is provided, set the event to unblock the device thread.
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Sets sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for the device's lifecycle."""

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device



    def run(self):
        """
        The main execution loop for the device.

        This loop synchronizes at a barrier, waits for scripts, and then
        executes them without any locking, creating a potential for race conditions.
        """
        while True:
            

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # 1. Synchronize all threads at the start of the cycle.
            Device.barrier.wait()
            # 2. Wait for the assign_script method to signal that scripts are ready.
            self.device.script_received.wait()
            self.device.script_received.clear()

            
            # CRITICAL: There is no locking mechanism in this loop. If multiple
            # devices' threads run this code simultaneously, race conditions will
            # occur when accessing and modifying shared data (e.g., a neighbor's
            # sensor_data).
            for (script, location) in self.device.scripts:
                script_data = []
                
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    
                    result = script.run(script_data)

                    # Unsafe concurrent modification of neighbor and self data.
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)