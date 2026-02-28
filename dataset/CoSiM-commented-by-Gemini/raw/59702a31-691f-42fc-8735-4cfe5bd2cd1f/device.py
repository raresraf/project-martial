"""
Models a distributed, parallel computation over a network of devices.

This script simulates a network where each device has a persistent internal
thread pool to process work in parallel for each time step. The system uses a
global barrier and a shared set of location-based locks to synchronize state
across all threads of all devices, following a Bulk Synchronous Parallel (BSP)
model.
"""

from threading import Lock, Event, Thread, Condition

class ReusableBarrier():
    """A reusable barrier implemented with a Condition variable."""
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()


    def wait(self):
        """
        Blocks the calling thread until the required number of threads arrive.
        Once all threads are waiting, they are all released and the barrier resets.
        """
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()


class Device(object):
    """
    Represents a single device (node) which manages a pool of worker threads.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and its internal pool of worker threads."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        # Event to signal that all scripts for a timepoint have been assigned.
        self.timepoint_done = Event()

        # Synchronization primitives for coordinating the internal thread pool.
        self.gotneighbours = Event()
        self.zavor = Lock() # Polish for "latch" or "bolt".
        self.threads = []
        self.neighbours = []
        self.nthreads = 8
        # Global synchronization primitives, to be overwritten in setup.
        self.barrier = ReusableBarrier(1)
        self.lockforlocation = {}
        self.num_locations = supervisor.supervisor.testcase.num_locations
        
        # Create and start the internal pool of worker threads.
        for i in xrange(self.nthreads):
            self.threads.append(DeviceThread(self, i))
        for i in xrange(self.nthreads):
            self.threads[i].start()


    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources for the entire network.

        The master device (device 0) creates a single global barrier sized for
        all worker threads in the system, and a dictionary of shared locks for
        each data location. These are then assigned to all other devices.
        """
        barrier = ReusableBarrier(devices[0].nthreads*len(devices))
        lockforlocation = {}
        for i in xrange(0, devices[0].num_locations):
            lock = Lock()
            lockforlocation[i] = lock
        for i in xrange(0, len(devices)):
            devices[i].barrier = barrier
            devices[i].lockforlocation = lockforlocation


    def assign_script(self, script, location):
        """Assigns a computational script to this device."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A `None` script signals the end of script assignment for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining all its worker threads."""
        for i in xrange(self.nthreads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A persistent worker thread within a Device's internal thread pool.
    """
    def __init__(self, device, id_thread):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id_thread = id_thread

    def run(self):
        """The main execution loop for the worker thread."""
        while True:
            # --- Neighbor Discovery Phase ---
            # Use a device-level lock to ensure neighbors are fetched only once
            # by the first thread that gets here.
            self.device.zavor.acquire()
            if not self.device.gotneighbours.is_set():
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.gotneighbours.set()
            self.device.zavor.release()
            
            # A `None` value for neighbors is the signal to shut down.
            if self.device.neighbours is None:
                break

            # Wait for the supervisor to finish assigning all scripts for this timepoint.
            self.device.timepoint_done.wait()
            
            # --- Work Distribution and Execution Phase ---
            # Statically select a subset of scripts to work on based on thread ID.
            myscripts = []
            for i in xrange(self.id_thread, len(self.device.scripts), self.device.nthreads):
                myscripts.append(self.device.scripts[i])

            # Process the assigned subset of scripts.
            for (script, location) in myscripts:
                # Acquire the global lock for the specific data location.
                self.device.lockforlocation[location].acquire()
                
                # Gather data from neighbors and self.
                script_data = []
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Execute script and disseminate results.
                if script_data:
                    result = script.run(script_data)
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the lock for the location.
                self.device.lockforlocation[location].release()

            # --- Synchronization and Reset Phase ---
            # First barrier: all threads wait here after finishing their work.
            self.device.barrier.wait()
            
            # One designated thread resets the device's state for the next timepoint.
            if self.id_thread == 0:
                self.device.timepoint_done.clear()
                self.device.gotneighbours.clear()
            
            # Second barrier: ensures all threads wait for the reset to complete
            # before starting the next timepoint loop.
            self.device.barrier.wait()
			