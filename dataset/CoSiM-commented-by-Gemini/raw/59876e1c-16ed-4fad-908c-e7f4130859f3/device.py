"""
Models a distributed, parallel computation over a network of devices.

This script implements a Bulk Synchronous Parallel (BSP) simulation using a
thread pool model. Each `Device` manages its own work queue and a pool of
`Worker` threads that consume tasks from it. Synchronization is handled at two
levels: a local barrier for the workers within a device and a global barrier
for all devices between time steps.
"""

from threading import Thread, Lock
# Assumes a 'barrier' module with a ReusableBarrierCond class exists.
import barrier

class Device(object):
    """
    Represents a single device (node) in the network.

    It encapsulates its own data, a work queue (`Workpool`), and a single master
    thread (`DeviceThread`) that manages a pool of `Worker` threads for each
    timepoint.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device, its work pool, and its master thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Global barrier for synchronizing all devices.
        self.barrier = None
        # Shared dictionary of locks for data locations.
        self.locationslocks = {}
        self.neighbours = []
        self.workpool = Workpool()
        self.thread = DeviceThread(self)

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources for the entire network.

        The master device (ID 0) creates shared locks for all data locations and
        a global barrier. It then distributes these to all devices and starts their
        main threads.
        """
        if self.device_id == 0:
            locationslocks = {}
            for dev in devices:
                for location in dev.sensor_data:
                    locationslocks[location] = Lock()

            barr = barrier.ReusableBarrierCond(len(devices))

            for dev in devices:
                dev.locationslocks = locationslocks
                dev.barrier = barr
                dev.thread.start()



    def assign_script(self, script, location):
        """
        Assigns a script to the device by adding it to the work pool.

        A `None` script signals that no more work will be added for this timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.workpool.putwork(script, location)
        else:
            # Signal to the work pool that production is done.
            self.workpool.endwork()

    def get_data(self, location):
        """Retrieves sensor data from a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its master thread."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a device, managing the lifecycle of worker threads
    for each timepoint.
    """

    def __init__(self, device):
        """Initializes the main thread for a device."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.workers = []
        # A local barrier to synchronize this thread with its 8 workers.
        self.workerbar = barrier.ReusableBarrierCond(9)

    def run(self):
        """The main execution loop, managing timepoints."""
        while True:
            # Populate the work pool with all scripts for the current timepoint.
            self.device.workpool.putlistwork(self.device.scripts)

            self.device.neighbours = self.device.supervisor.get_neighbours()
            neighbours = self.device.neighbours

            # A `None` value for neighbors is the signal to shut down.
            if neighbours is None:
                break

            # Create and start a new pool of worker threads for this timepoint.
            for i in xrange(8):
                worker = Worker(self.workerbar, self.device)
                self.workers.append(worker)
                self.workers[i].start()

            # Wait on the local barrier. This unblocks when all 8 workers have
            # exhausted the work pool and also waited on this barrier.
            self.workerbar.wait()

            # Clean up the completed worker threads.
            for i in range(8):
                self.workers[i].join()
            del self.workers[:]

            # Wait on the global barrier for all other devices to finish their timepoint.
            self.device.barrier.wait()


class Workpool(object):
    """A simple, thread-safe work queue implementation."""

    def __init__(self):
        """Initializes the queue, a lock, and a completion flag."""
        self.scripts = []
        self.lock = Lock()
        self.done = False


    def getwork(self):
        """
        Atomically retrieves one work item from the queue.

        Returns:
            - A (script, location) tuple if work is available.
            - An empty tuple `()` if the pool is temporarily empty but not done.
            - `None` if the pool is empty and `done` is True, signaling termination.
        """
        with self.lock:
            if not self.done or self.scripts:
                if self.scripts:
                    return self.scripts.pop()
                else:
                    return ()
            else:
                return None


    def putwork(self, script, location):
        """Atomically adds a single work item to the queue."""
        with self.lock:
            self.scripts.append((script, location))

    def endwork(self):
        """Marks the work pool as complete; no more items will be added."""
        with self.lock:
            self.done = True

    def putlistwork(self, scripts):
        """Resets the pool and populates it with a new list of work items."""
        with self.lock:
            self.done = False
            self.scripts = list(scripts)


class Worker(Thread):
    """
    A worker thread that consumes and executes tasks from a `Workpool`.
    """

    def __init__(self, barr, device):
        """Initializes the worker with a reference to its local barrier and device."""
        Thread.__init__(self, name="Worker Thread")
        self.lock = Lock()
        self.barrier = barr
        self.device = device

    def run(self):
        """Main loop for the worker; continuously fetches and executes work."""
        while True:
            work = self.device.workpool.getwork()

            if work is None:
                # Work pool is exhausted; wait on the local barrier and terminate.
                self.barrier.wait()
                return
            elif work: # Check if work is not an empty tuple
                self.update(work[0], work[1])

    def update(self, script, location):
        """
        Executes a single script and updates data for the given location.
        This method handles locking, data gathering, execution, and dissemination.
        """
        self.device.locationslocks[location].acquire()
        try:
            # Gather data from neighbors and self.
            script_data = []
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Execute the script and disseminate the result.
            if script_data:
                result = script.run(script_data)
                for device in self.device.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
        finally:
            self.device.locationslocks[location].release()
