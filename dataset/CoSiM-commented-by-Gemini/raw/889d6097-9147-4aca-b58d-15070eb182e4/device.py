"""
@file device.py
@brief Defines a device model using a barrier with dynamic thread registration.

This file implements a simulation device where each device instance dynamically
registers with a single, class-level `ReusableBarrier` upon initialization.

@warning This implementation is critically flawed as it contains no locking
         mechanism for data access (`get_data`, `set_data`), which will lead
         to race conditions in a multi-threaded environment.
"""

from threading import *


class ReusableBarrier():
    """
    A reusable barrier implemented with a Condition variable and static methods
    for dynamically adding participating threads.

    @note This implementation may be subject to race conditions, as a notified
          thread could re-enter `wait()` before the last thread has released
          the condition lock, potentially causing a deadlock.
    """
    num_threads = 0
    count_threads = 0

    def __init__(self):
        self.cond = Condition()
        self.thread_event = Event()

    def wait(self):
        """Blocks the calling thread until all registered threads have reached the barrier."""
        self.cond.acquire()
        ReusableBarrier.count_threads -= 1

        if ReusableBarrier.count_threads == 0:
            self.cond.notify_all()
            ReusableBarrier.count_threads = ReusableBarrier.num_threads
        else:
            self.cond.wait()

        self.cond.release()

    @staticmethod
    def add_thread():
        """Statically increments the number of threads the barrier should wait for."""
        ReusableBarrier.num_threads += 1
        ReusableBarrier.count_threads = ReusableBarrier.num_threads


class Device(object):
    """
    Represents a device that registers with a global barrier upon creation.
    """
    # A single, class-level barrier instance is created when the class is defined.
    barr = ReusableBarrier()    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device and registers it with the class-level barrier.
        """
        # Each new device increments the number of threads the barrier expects.
        Device.barr.add_thread()

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        This setup method is a no-op, as barrier setup is handled
        dynamically in the Device constructor.
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
                 will occur, leading to data corruption.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Wait at the shared barrier for all devices to be ready.
            Device.barr.wait()
            
            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Block Logic: Process all assigned scripts serially in this thread.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Gather data from neighbors (UNSAFE - NO LOCKING).
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    result = script.run(script_data)

                    # Propagate result to neighbors (UNSAFE - NO LOCKING).
                    for device in neighbours:
                        device.set_data(location, result)

                    self.device.set_data(location, result)