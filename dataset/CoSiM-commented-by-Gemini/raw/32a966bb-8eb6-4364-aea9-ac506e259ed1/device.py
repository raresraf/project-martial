"""
A simulation framework for a network of devices using a flawed global barrier.

This module defines a `Device` class and a custom `ReusableBarrier` that manages
its state with class-level variables, creating a single, shared barrier for the
entire system. The device threads synchronize at the beginning of each time step
but lack a second barrier at the end, which can lead to deadlocks. Furthermore,
all data access during script execution is non-atomic and not protected by locks,
creating significant race conditions.
"""

from threading import Event, Thread, Condition


class ReusableBarrier():
    """
    A non-reusable barrier implementation intended to synchronize a dynamic
    number of threads using a single, shared class instance.
    
    This barrier's state (`num_threads`, `count_threads`) is stored at the class
    level. It uses a `Condition` variable to block threads.
    Note: This single-phase implementation is not safely reusable and is prone
    to the "lost wake-up" problem in concurrent loops.
    """
    num_threads = 0
    count_threads = 0

    def __init__(self):
        """Initializes the Condition variable for the barrier."""
        self.cond = Condition()
        self.thread_event = Event()

    def wait(self):
        """
        Blocks the calling thread until all registered threads reach this point.
        """
        self.cond.acquire()
        ReusableBarrier.count_threads -= 1

        # Invariant: The last thread to arrive notifies all waiting threads
        # and resets the counter.
        if ReusableBarrier.count_threads == 0:
            self.cond.notify_all()
            ReusableBarrier.count_threads = ReusableBarrier.num_threads
        else:
            self.cond.wait()

        self.cond.release()

    @staticmethod
    def add_thread():
        """
        Static method to increment the number of threads the barrier should wait for.
        """
        ReusableBarrier.num_threads += 1
        ReusableBarrier.count_threads = ReusableBarrier.num_threads


class Device(object):
    """
    Represents a single device that uses a shared, class-level barrier.
    """
    # A single, shared barrier instance for all Device objects.
    barr = ReusableBarrier()    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device and registers its thread with the global barrier.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The device's internal sensor data.
            supervisor: The central supervisor managing the device network.
        """
        Device.barr.add_thread()

        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        A no-op method, as barrier setup is handled in the constructor.
        """
        pass

    def assign_script(self, script, location):
        """
        Receives a script from the supervisor and signals when assignments are done.

        Args:
            script: The script object to execute.
            location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device.

    This thread's logic synchronizes with other devices only at the beginning
    of its loop. It reads and writes shared data without locks.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device



    def run(self):
        """The main execution loop, organized into discrete timepoints."""
        while True:
            # Get neighbours for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break
            
            # Invariant: All threads must synchronize at the barrier at the
            # start of the timepoint.
            Device.barr.wait()

            # Wait for the supervisor to finish assigning scripts for this device.
            self.device.script_received.wait()
            self.device.script_received.clear()

            # Block Logic: Sequentially process each assigned script.
            # CRITICAL: This block reads and writes to shared `sensor_data` on
            # multiple devices without any locks, creating a race condition.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Aggregate data from neighbours and self.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Invariant: Script runs only if there is data to process.
                if script_data != []:
                    result = script.run(script_data)

                    # Broadcast the result to all participants.
                    for device in neighbours:
                        device.set_data(location, result)

                    self.device.set_data(location, result)
            # The lack of a second barrier here before the loop repeats can
            # lead to deadlocks.