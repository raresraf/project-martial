"""
This module implements a device simulation using a modern ThreadPoolExecutor
for managing concurrent tasks within each device.

It features a high-level approach to threading but contains a critically
flawed and dangerous asymmetrical locking mechanism, where a `get` operation
acquires a lock and a `set` operation releases it, creating a high risk of
deadlocks.
"""

from threading import Event, Thread, Lock
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import barrier

class Device(object):
    """
    Represents a single device in the network.

    Each device manages its own data, a set of local locks, and a thread that
    dispatches tasks to a high-level thread pool.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.device_barrier = None
        self.script_received = Event()
        self.timepoint_done = Event()

        self.scripts = []
        self.future_list = []
        # Each device has its own set of locks, which are not shared.
        self.access_locks = {}
        for location in sensor_data:
            self.access_locks[location] = Lock()

        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""

        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes a shared barrier for all devices in the simulation."""

        if self.device_id == 0:
            device_barrier = barrier.ReusableBarrierCond(len(devices))
            for device in devices:
                device.set_barrier(device_barrier)

    def set_barrier(self, device_barrier):
        """Assigns the shared barrier to this device."""

        self.device_barrier = device_barrier

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device."""

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()


    def get_data(self, location):
        """
        Retrieves sensor data and ACQUIRES a lock for that location.

        CRITICAL FLAW: This method acquires a lock but does not release it.
        The caller is responsible for ensuring that `set_data` is eventually
        called on this same device and location to release the lock. This is a
        highly dangerous, asymmetrical pattern that can easily lead to deadlocks.
        """

        if location in self.sensor_data:
            self.access_locks[location].acquire()
            result = self.sensor_data[location]
        else:
            result = None

        return result

    def set_data(self, location, data):
        """
        Sets sensor data and RELEASES the lock for that location.

        CRITICAL FLAW: This method releases a lock it did not acquire. It is
        part of a dangerous asymmetrical pattern with `get_data`.
        """

        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.access_locks[location].release()

    def shutdown(self):
        """Waits for the device's main thread to terminate."""

        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, which dispatches tasks to a thread pool.
    """

    def execute(self, neighbours, script, location):
        """
        The target function for tasks executed by the thread pool.
        
        This function is responsible for the entire lifecycle of a script run,
        including the unsafe acquire/release locking pattern.
        """

        script_data = []

        
        # Acquire locks on all neighbors by calling their get_data methods.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

        
        # Acquire a lock on the current device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            result = script.run(script_data)

            

            # Release locks on all neighbors by calling their set_data methods.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    device.set_data(location, result)

            
            # Release the lock on the current device.
            self.device.set_data(location, result)

    def __init__(self, device):
        

        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Each device's control thread manages its own high-level thread pool.
        self.thread_pool = ThreadPoolExecutor(max_workers=8)

    def run(self):
        """The main loop for the device control thread."""
        while True:

            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            future_list = []

            
            # Wait for the signal to begin processing the timepoint.
            self.device.timepoint_done.wait()


            if self.device.script_received.is_set():
                self.device.script_received.clear()

                # Submit all scripts as tasks to the thread pool for execution.
                for (script, location) in self.device.scripts:
                    future = self.thread_pool.submit(self.execute, neighbours, script, location)
                    future_list.append(future)

            
            self.device.timepoint_done.clear()
            self.device.script_received.set()

            
            # Wait for all tasks submitted to this device's pool to complete.
            concurrent.futures.wait(future_list)

            # Wait at the global barrier for all other devices to finish their work.
            self.device.device_barrier.wait()

        
        # Gracefully shut down the thread pool when the simulation ends.
        self.thread_pool.shutdown()
