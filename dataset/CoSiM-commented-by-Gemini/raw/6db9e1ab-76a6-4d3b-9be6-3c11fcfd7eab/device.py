"""
This module implements a distributed device simulation using a complex and
buggy threading model.

The architecture consists of a main `DeviceThread` for each device, which in
turn manually manages a pool of `DeviceCore` worker threads for executing
scripts. The simulation's synchronization logic is deeply flawed, containing
deadlock-prone patterns in its data access methods.

WARNING: This code is not thread-safe and contains severe deadlocking bugs.
It is presented as an example of incorrect concurrency implementation.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierSem


class Device(object):
    """
    Represents a device in the simulation.

    It manages its sensor data and a main control thread. It contains a critical
    bug in its `get_data` and `set_data` methods that will lead to deadlocks.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.start_event = Event()
        self.thread = DeviceThread(self)
        
        # Creates a dictionary of locks, one for each data location.
        self.data_lock = {data: Lock() for data in sensor_data}

        self.barrier = None
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a shared barrier to all devices.
        """
        if self.barrier is None:
            self.barrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                dev.barrier = self.barrier
        # Signals that setup is complete and main threads can start.
        self.start_event.set()

    def assign_script(self, script, location):
        """Assigns a script to be executed by the device."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that script assignment for this step is over.
            self.timepoint_done.set()
        self.script_received.set()

    def get_data(self, location):
        """
        Retrieves data for a given location, acquiring a lock.

        WARNING: This method acquires a lock but NEVER releases it. This will
        cause an immediate deadlock if any other thread attempts to get data
        from the same location.
        """
        if location in self.sensor_data:
            self.data_lock[location].acquire()
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """
        Sets data for a given location and releases a lock.

        WARNING: This method releases a lock it did not acquire. It is
        dangerously coupled with `get_data`, which is expected to have acquired
        the lock beforehand.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.data_lock[location].release()

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceCore(Thread):
    """
    A worker thread intended to execute a single script.
    """
    def __init__(self, device, location, script, neighbours):
        Thread.__init__(self, name="Device core %d" % device.device_id)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        Gathers data, runs the script, and updates data.

        WARNING: This method will cause a deadlock. It calls the buggy
        `device.get_data()` method in a loop. The first call will acquire a
        lock and never release it, causing the second call (for the next device
        or even the same device) to block forever.
        """
        script_data = []
        # This loop will deadlock on the second iteration if neighbors share a location.
        for device in self.neighbours:
            if self.device.device_id != device.device_id:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
        
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data:
            result = self.script.run(script_data)
            # This part will likely never be reached due to the deadlock above.
            for device in self.neighbours:
                if self.device.device_id != device.device_id:
                    device.set_data(self.location, result)
            self.device.set_data(self.location, result)

class DeviceThread(Thread):
    """
    The main control thread for a device. It manually manages a pool of
    `DeviceCore` worker threads.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main simulation loop."""
        self.device.start_event.wait()

        while True:
            # Synchronize with all other devices at the start of a step.
            self.device.barrier.wait()
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Shutdown signal.

            # Wait until all scripts for the current step have been assigned.
            while not self.device.timepoint_done.is_set():
                self.device.script_received.wait()

            # --- Manual and inefficient thread pool management ---
            used_cores = 0
            free_core = list(range(8))
            threads = {}
            
            for (script, location) in self.device.scripts:
                # If there is a free "core", start a new worker.
                if used_cores < 8:
                    dev_core = DeviceCore(self.device, location, script, neighbours)
                    dev_core.start()
                    threads[free_core.pop()] = dev_core
                    used_cores += 1
                else:
                    # If all cores are busy, poll until one is finished.
                    for thread_id, thread in threads.items():
                        if not thread.isAlive():
                            thread.join()
                            free_core.append(thread_id)
                            used_cores -= 1
            
            # Join any remaining threads after all scripts are dispatched.
            for thread in threads.values():
                thread.join()

            # Reset events for the next simulation step.
            self.device.timepoint_done.clear()
            if self.device.script_received.is_set():
                self.device.script_received.clear()
