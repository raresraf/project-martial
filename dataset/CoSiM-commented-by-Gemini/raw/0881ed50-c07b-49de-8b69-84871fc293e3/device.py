"""
This module implements a simulated network of devices that process sensor data.
It features a custom `ReusableBarrier` for synchronization and employs a single,
global lock to serialize the core data processing logic across all devices,
ensuring that only one device executes a script at any given time.
"""

from threading import Event, Thread, Condition, Lock

class ReusableBarrier(object):
    """
    A custom, reusable barrier implementation that blocks a set number of threads
    until all of them have reached the barrier. Once all threads are waiting,
    they are all released and the barrier resets for the next use.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()
        
    def wait(self):
        """
        Causes a thread to wait at the barrier. The last thread to arrive
        (i.e., the one that brings the count to zero) notifies all waiting
        threads and resets the barrier.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # Invariant: If the count reaches zero, all threads have arrived at the barrier.
        if self.count_threads == 0:
            # Wake up all waiting threads.
            self.cond.notify_all()
            # Reset the barrier for the next synchronization point.
            self.count_threads = self.num_threads
        else:
            # Wait to be notified by the last thread.
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a single device in the simulation. It holds sensor data and
    participates in a synchronized, step-by-step computation orchestrated by
    a supervisor.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor (Supervisor): The central controller for the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.barr = None
        self.lock = None
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes a single shared barrier and a single global lock
        to all devices in the simulation. This method should be called by all devices,
        but only device 0 will perform the setup.

        Args:
            devices (list): A list of all devices in the simulation.
        """
        # This check ensures that the setup is performed only once by device 0.
        if devices[0].barr is None and devices[0].device_id == self.device_id:
            bariera = ReusableBarrier(len(devices))
            lock = Lock()
            # Distribute the same barrier and lock instance to every device.
            for i in devices:
                i.barr = bariera
            for j in devices:
                j.lock = lock

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script signals that the
        script assignment phase for the current timepoint is complete.

        Args:
            script (Script): The script to be executed.
            location (str): The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Signal that the device can proceed with the computation for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data from a specified location. Note: This method is not
        thread-safe by itself and relies on external locking.

        Args:
            location (str): The location from which to retrieve data.

        Returns:
            The sensor data if the location exists, otherwise None.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates sensor data at a specified location. Note: This method is not
        thread-safe by itself and relies on external locking.

        Args:
            location (str): The location to update.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main execution thread for a device. It processes scripts and handles
    synchronization with other devices.
    """

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main control loop. It waits for a signal to start processing,
        executes scripts under a global lock, and then synchronizes at a barrier.
        """
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # End of simulation.
                break

            # Wait until the supervisor signals that script assignment is complete.
            self.device.timepoint_done.wait()

            # Process all scripts for the current timepoint.
            for (script, location) in self.device.scripts:
                # Acquire the global lock. This serializes script execution across all devices.
                self.device.lock.acquire()
                script_data = []
                
                # Gather data from neighbors.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: Only run script if data has been collected.
                if script_data:
                    result = script.run(script_data)

                    # Distribute the result to neighbors.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Update local data.
                    self.device.set_data(location, result)
                # Release the global lock, allowing another device to proceed.
                self.device.lock.release()

            # Clear the event for the next timepoint.
            self.device.timepoint_done.clear()
            # Invariant: Wait for all devices to finish their processing for this
            # timepoint before starting the next one.
            self.device.barr.wait()