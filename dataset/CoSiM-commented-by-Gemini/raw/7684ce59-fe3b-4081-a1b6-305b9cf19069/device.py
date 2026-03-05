"""
This module presents a variant of a multi-threaded device simulation framework.

It uses a model where a main `DeviceThread` spawns `DeviceWorkerThread`s,
each responsible for handling all scripts for a specific "location".
Synchronization is attempted with a custom barrier based on `threading.Condition`.

NOTE: This implementation contains several critical concurrency bugs.
1. The `ReusableBarrierCond` is not a correct implementation of a reusable barrier
   and is prone to race conditions.
2. The `setup_devices` method for distributing shared locks is racy.
3. The `get_data` and `set_data` methods are not thread-safe.
"""

from threading import Event, Thread, Condition, Lock

class Device(object):
    """
    Represents a device in the simulation.

    This class organizes scripts by location and uses a per-location worker
    thread model. It contains flawed setup and data access logic.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()
        # Scripts are organized in a dictionary keyed by location.
        self.scripts_dict = {}
        # A dictionary to hold Lock objects for each location.
        self.locations_locks = {}
        self.timepoint_done = None # The shared barrier object.
        self.neighbours = None
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources (barrier and locks).

        BUG: This setup logic is not thread-safe. Multiple devices could execute
        this method concurrently, leading to race conditions where different,
        unshared lock/barrier objects are created, breaking synchronization.
        The `if location not in self.locations_locks:` check is not atomic.
        """
        nr_devices = len(devices)
        if self.timepoint_done is None:
            self.timepoint_done = ReusableBarrierCond(nr_devices)
            for device in devices:
                if device.timepoint_done is None and device != self:
                    device.timepoint_done = self.timepoint_done

        # Attempt to create and distribute a shared lock for each location.
        for location in self.sensor_data.keys():
            if location not in self.locations_locks:
                self.locations_locks[location] = Lock()
                for device in devices:
                    if location not in device.locations_locks and device != self:
                        # This assignment should make the lock object shared.
                        device.locations_locks[location] = self.locations_locks[location]

    def assign_script(self, script, location):
        """Assigns a script to a specific location for this device."""
        if script is not None:
            # Group scripts by their target location.
            if location in self.scripts_dict:
                self.scripts_dict[location].append(script)
            else:
                self.scripts_dict[location] = [script]
        else:
            # A `None` script signals that all scripts for the time step have been assigned.
            self.scripts_received.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        BUG: This method is not thread-safe as it does not use a lock. It can
        race with `set_data`, leading to inconsistent or partial reads.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Updates sensor data for a given location.
        
        BUG: This method is not thread-safe as it does not use a lock. It can
        race with `get_data` or other `set_data` calls.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a device."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main simulation loop for the device."""
        while True:
            # Get the current list of neighbors from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            if self.device.neighbours is None:
                break # Shutdown signal.

            # Wait for the signal that all scripts have been assigned for this step.
            self.device.scripts_received.wait()

            # --- Inefficient Threading Model ---
            # Create a new worker thread for each location that has scripts.
            threads = []
            for location in self.device.scripts_dict.keys():
                thread = DeviceWorkerThread(self.device, location)
                thread.start()
                threads.append(thread)

            # Wait for all worker threads for all locations to complete.
            for thread in threads:
                thread.join()
            
            # Clear the received scripts and event for the next time step.
            self.device.scripts_dict.clear()
            self.device.scripts_received.clear()

            # Wait at the global barrier for all other devices to finish.
            self.device.timepoint_done.wait()

class DeviceWorkerThread(Thread):
    """A worker thread responsible for executing all scripts for one location."""

    def __init__(self, device, location):
        Thread.__init__(self, name="Device Worker %d Loc %d" % (device.device_id, location))
        self.device = device
        self.location = location

    def run(self):
        """
        Executes all scripts for the assigned location.
        It holds a single lock for the duration of all script executions.
        """
        # --- Coarse-Grained Locking ---
        # The lock is held while all scripts for this location are executed sequentially.
        self.device.locations_locks[self.location].acquire()
        
        for script in self.device.scripts_dict[self.location]:
            # --- Data Gathering ---
            script_data = []
            for device in self.device.neighbours:
                data = device.get_data(self.location) # Unsafe read
                if data is not None:
                    script_data.append(data)
            
            data = self.device.get_data(self.location) # Unsafe read
            if data is not None:
                script_data.append(data)

            # --- Script Execution and Propagation ---
            if script_data:
                result = script.run(script_data)
                # Propagate the result to neighbors and self.
                for device in self.device.neighbours:
                    device.set_data(self.location, result) # Unsafe write
                self.device.set_data(self.location, result) # Unsafe write

        self.device.locations_locks[self.location].release()


class ReusableBarrierCond(object):
    """
    An implementation of a barrier using a Condition variable.
    
    BUG: This implementation is not correctly reusable. When the last thread calls
    `notify_all` and resets the counter, waiting threads wake up and can re-enter
    the `wait` method before all threads from the original waiting group have exited.
    This breaks the synchronization guarantee of a barrier. A correct reusable
    barrier requires a two-phase approach to separate thread generations.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        self.cond.acquire()
        self.count_threads -= 1

        if self.count_threads == 0:
            # Last thread arrives: notify everyone and reset for the "next" round.
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # Not the last thread: wait to be notified.
            self.cond.wait()

        self.cond.release()
