"""
This module defines a distributed device simulation framework.

The architecture features one main thread per device (`DeviceThread`) which
handles all script execution sequentially. It does not use a worker pool.
Device with device_id 0 acts as a coordinator, creating and distributing
shared synchronization objects (a ReusableBarrier and location-based locks)
to all other devices.

- ReusableBarrier: A custom barrier for synchronizing all devices between cycles.
- Device: A node in the network.
- DeviceThread: The single thread of execution for a device, which processes
  all its assigned scripts sequentially.
"""

from threading import Event, Thread, Condition, Semaphore, Lock, RLock

class ReusableBarrier():
    """A reusable barrier implementation using a Condition variable."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # Last thread to arrive notifies all waiting threads and resets the counter.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Wait to be notified by the last thread.
            self.cond.wait()
        self.cond.release()


class Device(object):
    """Represents a device node in the simulation network.

    Each device has a single thread that executes its assigned scripts sequentially.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that all scripts for a cycle have been received.
        self.script_received = Event()
        self.scripts = []
        self.devices = None
        # The barrier is a shared object, initialized by the coordinator (device 0).
        self.timepoint_done = None
        # The locks are also shared, mapping locations to Lock objects.
        self.semafor = {}
        # The single execution thread for this device.
        self.thread = DeviceThread(self)

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes and distributes shared resources for the device network.

        Device 0 acts as the coordinator, creating a shared barrier and a
        dictionary of locks for all unique data locations. Other devices
        receive a reference to these shared objects.
        """
        self.devices = devices
        if self.device_id == 0:
            # Coordinator (Device 0) creates the shared barrier.
            self.timepoint_done = ReusableBarrier(len(self.devices))
            
            # Coordinator discovers all unique locations and creates a lock for each.
            for device in self.devices:
                for location, data in device.sensor_data.iteritems():
                    if location not in self.semafor:
                        self.semafor.update({location:Lock()})
            for location, data in self.sensor_data.iteritems():
                if location not in self.semafor:
                    self.semafor.update({location:Lock()})
        else:
            # Other devices find the coordinator to get a reference to shared objects.
            for device in self.devices:
                if device.device_id == 0:
                    self.timepoint_done = device.timepoint_done
                    self.semafor = device.semafor
        
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to the device's list for the current cycle.

        A 'None' script signals the end of assignment, unblocking the DeviceThread.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # All scripts are assigned, set event to start processing.
            self.script_received.set()

    def get_data(self, location):
        """Gets sensor data for a specific location."""
        value = None
        if location in self.sensor_data:
            value = self.sensor_data[location]
        return value

    def set_data(self, location, data):
        """Sets sensor data for a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The single thread of execution for a device.

    It waits for scripts, executes them sequentially, and then synchronizes
    with other devices at a barrier.
    """

    def __init__(self, device):
        """Initializes the device's main thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop for the device."""
        while True:
            # Get neighbors for the current cycle.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # End of simulation: perform one final barrier wait to sync shutdown.
                self.device.timepoint_done.wait()
                break

            # Wait until all scripts for the cycle have been assigned.
            self.device.script_received.wait()
            
            # Execute all assigned scripts sequentially in this single thread.
            for (script, location) in self.device.scripts:
                # Acquire the shared lock for the data location.
                self.device.semafor[location].acquire()
                script_data = []
                
                # Gather data from neighboring devices.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather data from this device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Run the script and store the result.
                    result = script.run(script_data)

                    # Distribute the result to all relevant devices.
                    for device in neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
                
                # Release the lock for the data location.
                self.device.semafor[location].release()
            
            
            # Wait at the barrier for all other devices to finish their cycles.
            self.device.timepoint_done.wait()
            
            # Clear state for the next cycle.
            self.device.scripts = []
            self.device.script_received.clear()
