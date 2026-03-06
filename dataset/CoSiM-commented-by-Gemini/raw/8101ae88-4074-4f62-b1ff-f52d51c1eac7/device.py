"""
This module simulates a network of devices using a thread-pool-per-device
architecture. Each primary `DeviceThread` acts as a manager, spawning a pool of
worker `MyThread`s for each simulation timepoint to process scripts in parallel.

Synchronization is achieved through two mechanisms:
1. A shared, location-based dictionary of locks (`sync_location_lock`) ensures
   that only one worker (from any device) can operate on a given location at a time.
2. A custom `ReusableBarrierCond` based on `threading.Condition` is used to
   synchronize all `DeviceThread`s at the end of each timepoint.
"""

from threading import Event, Thread, Lock, Condition

class Device(object):
    """
    Represents a single device, managing its sensor data, scripts, and a pool
    of worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()

        self.thread = DeviceThread(self)
        self.thread.start()
        
        self.locations = []
        # This lock seems intended for data access but is used inconsistently.
        # The main protection comes from sync_location_lock.
        self.sync_data_lock = Lock()
        # A shared dictionary mapping each location to a specific lock.
        self.sync_location_lock = {}
        self.cores = 8
        self.barrier = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources like the barrier and
        location locks across all devices. Should be called by one "master" device.
        """
        if self.device_id == 0:
            locations_number = self.get_locations_number(devices)
            for location in range(locations_number):
                self.sync_location_lock[location] = Lock()
            self.barrier = ReusableBarrierCond(len(devices))
            # Distribute the shared objects to all other devices.
            for device in devices:
                device.barrier = self.barrier
                device.sync_location_lock = self.sync_location_lock

    def assign_script(self, script, location):
        """Assigns a script to the device for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A 'None' script signals the end of assignments for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data. Note: Not thread-safe by itself. Relies on
        external locking managed by the calling worker thread.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Sets sensor data. Note: Not thread-safe by itself. Relies on
        external locking managed by the calling worker thread.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's main controller thread to complete."""
        self.thread.join()

    def get_locations_number(self, devices):
        """Calculates the total number of unique locations across all devices."""
        for device in devices:
            for location in device.sensor_data:
                if location not in self.locations:
                    self.locations.append(location)
        return len(self.locations)


class DeviceThread(Thread):
    """
    The main controller thread for a device. It manages a pool of worker
    threads (`MyThread`) for each simulation step.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main simulation loop for the device."""
        while True:
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Termination signal.
            
            # Wait for the supervisor to signal that all scripts are assigned.
            self.device.timepoint_done.wait()
            
            # --- Thread Pool Management ---
            # Create a pool of worker threads for this timepoint.
            my_threads = [MyThread(self) for _ in range(self.device.cores)]
            
            # Distribute scripts to workers in a round-robin fashion.
            index = 0
            for (script, location) in self.device.scripts:
                my_threads[index % self.device.cores].assign_script(script, location)
                index += 1
            
            # Start and wait for all worker threads to complete.
            for thread in my_threads:
                thread.set_neighbours(neighbours)
                thread.start()
            for thread in my_threads:
                thread.join()
            
            self.device.scripts = [] # Reset for next timepoint.
            self.device.timepoint_done.clear()
            
            # After all local work is done, synchronize with other devices.
            self.device.barrier.wait()

class MyThread(Thread):
    """A worker thread that executes a subset of scripts for a device."""
    def __init__(self, parent_device_thread):
        Thread.__init__(self)
        self.parent = parent_device_thread
        self.scripts = []
        self.neighbours = []

    def set_neighbours(self, neighbours):
        self.neighbours = neighbours

    def assign_script(self, script, location):
        self.scripts.append((script, location))

    def run(self):
        """Executes the assigned scripts."""
        for (script, location) in self.scripts:
            # Acquire the lock for this specific location to prevent other workers
            # (from any device) from processing the same location concurrently.
            self.parent.device.sync_location_lock[location].acquire()
            
            script_data = []
            
            # --- Data Aggregation ---
            # The per-device sync_data_lock is acquired and released for each data access.
            # While this appears to add safety, the primary protection against race
            # conditions on location data comes from the outer sync_location_lock.
            for device in self.neighbours:
                device.sync_data_lock.acquire()
                data = device.get_data(location)
                device.sync_data_lock.release()
                if data is not None:
                    script_data.append(data)
            
            self.parent.device.sync_data_lock.acquire()
            data = self.parent.device.get_data(location)
            self.parent.device.sync_data_lock.release()
            if data is not None:
                script_data.append(data)

            # --- Computation and Write-back ---
            if script_data:
                result = script.run(script_data)
                
                for device in self.neighbours:
                    device.sync_data_lock.acquire()
                    device.set_data(location, result)
                    device.sync_data_lock.release()
                
                self.parent.device.sync_data_lock.acquire()
                self.parent.device.set_data(location, result)
                self.parent.device.sync_data_lock.release()
            
            self.parent.device.sync_location_lock[location].release()


class ReusableBarrierCond(object):
    """
    A barrier implementation using a `threading.Condition` variable.

    Note: This is not a truly "reusable" barrier in the general sense, as fast
    threads could loop around and re-enter the wait before slow threads have
    left, causing a deadlock. However, it is safe in this specific program
    because all threads are joined before the barrier is used again in the next
    timepoint.
    """
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Blocks until all `num_threads` have called this method."""
        with self.cond:
            self.count_threads -= 1
            if self.count_threads == 0:
                # Last thread arrived; notify all waiting threads and reset the counter.
                self.cond.notify_all()
                self.count_threads = self.num_threads
            else:
                # Not the last thread, so wait to be notified.
                self.cond.wait()
