"""
This script models a distributed network of devices for parallel computation,
similar to other versions in this dataset. This implementation uses a model where
the main thread of a device spawns multiple worker threads for each timepoint
to process a list of computational scripts.

NOTE: This implementation contains a critical race condition in the work
distribution mechanism, making it non-functional and unreliable under concurrency.
"""
from threading import Event, Thread, Lock, Semaphore

class Device(object):
    """
    Represents a device node in the simulated network. It manages its data,
    scripts, and the main control thread.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and its main control thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """String representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared simulation resources like the global
        barrier and location-based locks. This is run only by the device with
        the lowest ID.
        """
        # This complex logic ensures that only one device (the one with the
        # lowest ID) performs the global setup.
        is_lowest_id = True
        for dev in devices:
            if self.device_id > dev.device_id:
                is_lowest_id = False
                break

        if is_lowest_id:
            barrier = ReusableBarrierSem(len(devices))
            map_locations = {}
            for dev in devices:
                # Assign the shared barrier to each device.
                dev.barrier = barrier
                # Create locks for any new data locations found.
                for location in dev.sensor_data:
                    if location not in map_locations:
                        map_locations[location] = Lock()
                # Assign the shared map of locks to each device.
                dev.map_locations = map_locations

    def assign_script(self, script, location):
        """Appends a script to the device's task list for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            # This event is set on every script assignment, which is unusual.
            self.script_received.set()
        else:
            # A 'None' script signals the end of script assignment.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main control thread for a clean shutdown."""
        self.thread.join()

class DeviceThread(Thread):
    """The main control thread for a device, orchestrating work for each timepoint."""

    def __init__(self, device):
        Thread.__init__(self)
        self.device = device

    def run(self):
        """The main loop that manages the simulation lifecycle for this device."""
        while True:
            self.device.timepoint_done.clear()
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # End of simulation.
                break

            # Wait until all scripts for the timepoint have been assigned.
            self.device.timepoint_done.wait()
            
            script_list = list(self.device.scripts) # Make a copy for the workers.
            thread_list = []
            
            # Spawn a fixed number of worker threads.
            # CRITICAL FLAW: All threads are given a reference to the same list object,
            # and they will modify it concurrently without a lock, causing a race condition.
            for i in xrange(8):
                # The 'index' is always 0, which is also part of the flawed design.
                thread = SingleDeviceThread(self.device, script_list, neighbours, 0)
                thread.start()
                thread_list.append(thread)
            
            # Wait for all worker threads to complete.
            for thread in thread_list:
                thread.join()

            # Wait at the global barrier for all other devices to finish.
            self.device.barrier.wait()

class SingleDeviceThread(Thread):
    """
    A worker thread intended to process a single script.
    This class has a severe concurrency flaw.
    """
    def __init__(self, device, script_list, neighbours, index):
        Thread.__init__(self)
        self.device = device
        self.script_list = script_list # A reference to a list shared between threads.
        self.neighbours = neighbours
        self.index = index

    def run(self):
        """
        Pops one task from the shared list and executes it. This is not thread-safe.
        """
        # RACE CONDITION: Multiple threads call list.pop(0) on the same list
        # concurrently. This will lead to data loss and unpredictable behavior.
        if self.script_list:
            try:
                (script, location) = self.script_list.pop(self.index)
                self.compute(script, location)
            except IndexError:
                # This exception is likely to occur due to the race condition.
                pass

    def update(self, result, location):
        """Broadcasts the computation result to the local device and its neighbors."""
        for device in self.neighbours:
            device.set_data(location, result)
        self.device.set_data(location, result)

    def collect(self, location, neighbours, script_data):
        """Gathers data for a given location from the device and its neighbors."""
        # Acquire lock to ensure safe access to sensor data during collection.
        self.device.map_locations[location].acquire()
        for device in self.neighbours:
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)
        
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)
        # The lock is released in compute(), which is poor practice. It should be
        # released here or in a 'finally' block to ensure release.

    def compute(self, script, location):
        """Performs the data collection, script execution, and result update."""
        script_data = []
        self.collect(location, self.neighbours, script_data)

        try:
            if script_data:
                result = script.run(script_data)
                self.update(result, location)
        finally:
            # Release the lock that was acquired in the collect() method.
            self.device.map_locations[location].release()

class ReusableBarrierSem():
    """A standard, reusable two-phase barrier for thread synchronization."""

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Blocks the thread until all participating threads reach the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads
        self.threads_sem1.acquire()

    def phase2(self):
        """Second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()
