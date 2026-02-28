"""
Models a distributed, parallel computation over a network of devices.

This script simulates a network of devices operating in discrete time steps,
following a Bulk Synchronous Parallel (BSP) model. In each time step (or "timepoint"),
devices execute scripts in parallel, synchronize at a global barrier, and then
proceed to the next time step.
"""

from threading import Event, Lock, Thread, Condition


class ReusableBarrierCond():
    """
    A reusable barrier implementation using a `threading.Condition`.

    This barrier causes threads to block on the `wait()` method until a
    predefined number of threads have all called `wait()`. Once the count is
    reached, all waiting threads are released and the barrier automatically
    resets for its next use.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()


    def wait(self):
        """
        Causes a thread to wait at the barrier.

        When the required number of threads arrive, all are notified and released.
        """
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                # All threads have arrived, notify all and reset for next use.
                self.cond.notify_all()
                self.count_threads = self.num_threads
            else:
                # Not all threads have arrived yet, wait to be notified.
                self.cond.wait()


class Device(object):
    """
    Represents a single device (or node) in the simulated network.

    Each device manages its own sensor data, has an associated execution thread,
    and communicates with neighboring devices under the coordination of a
    supervisor.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device and starts its main execution thread.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been received for processing.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that the current timepoint is complete.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

        # The master device holds the global barrier and locks.
        self.master = None
        self.bariera = None  # Polish for "barrier".
        self.lock = Lock()
        # A dictionary of locks, one for each data location, likely Polish for "padlocks".
        self.lacate = {}

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes the network by designating a master and a shared barrier.

        This method bootstraps the simulation environment, ensuring all devices
        share the same synchronization primitives managed by the master device.
        """
        self.master = devices[0]
        self.master.bariera = ReusableBarrierCond(len(devices))


    def assign_script(self, script, location):
        """
        Assigns a script to this device for execution in the current timepoint.

        A `None` script is a signal that this device has no more work for this
        timepoint, allowing its worker thread to proceed to the barrier.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Signal that script assignment is done for this timepoint.
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """Retrieves sensor data from a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its execution thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main worker thread for a Device.

    This thread orchestrates the device's participation in the BSP model,
    executing a loop for each timepoint that involves waiting for work,
    processing data, and synchronizing with other devices.
    """

    def __init__(self, device):
        """Initializes the worker thread for a given device."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop for the device."""
        while True:
            # Get the set of neighbors for the current timepoint.
            neighbours = self.device.supervisor.get_neighbours()

            # A `None` value for neighbors is the signal to shut down.
            if neighbours is None:
                break

            # Phase 1: Wait for the supervisor to assign scripts.
            self.device.script_received.wait()

            tlist = []
            # Phase 2: Process assigned scripts.
            for (script, location) in self.device.scripts:
                # Gather data from neighbors and self for the script.
                script_data = []
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data:
                    # Execute each script in its own thread.
                    my_thread = MyThread(script, script_data, location,
                    neighbours, self.device, self.device.master.lock,
                    self.device.master.lacate)
                    my_thread.start()
                    tlist.append(my_thread)

            # Wait for all script-execution threads to complete.
            for my_thread in tlist:
                my_thread.join()

            # Phase 3: Global synchronization.
            # Wait at the master barrier for all devices to finish this timepoint.
            self.device.master.bariera.wait()

            # Wait for the signal that the timepoint is fully complete.
            self.device.timepoint_done.wait()
            # Clear events to prepare for the next timepoint.
            self.device.script_received.clear()

class MyThread(Thread):
    """
    A thread dedicated to executing one script on a set of data.

    This thread ensures that updates to a specific data location are atomic
    across the network by using a location-specific lock.
    """

    def __init__(self, script, script_data, location, neighbours,
                own_device, mlock, lacate):
        """Initializes the script-execution thread."""
        Thread.__init__(self)
        self.script = script
        self.script_data = script_data
        self.location = location
        self.neighbours = neighbours
        self.own_device = own_device
        self.mlock = mlock
        self.lacate = lacate

    def run(self):
        """Executes the script and disseminates the results."""
        # Run the computation.
        result = self.script.run(self.script_data)

        # Acquire the lock for this specific data location to ensure safe updates.
        if self.location in self.lacate:
            self.lacate[self.location].acquire()
        else:
            # Lazy initialization of the lock for this location.
            self.lacate[self.location] = Lock()
            self.lacate[self.location].acquire()

        # Disseminate the result to all neighbors.
        for device in self.neighbours:
            device.set_data(self.location, result)
        # Update the data on the local device as well.
        self.own_device.set_data(self.location, result)

        # Release the lock for the location.
        self.lacate[self.location].release()
