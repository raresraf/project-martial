"""
This module defines a sophisticated, multi-threaded framework for a distributed
device simulation. It uses a combination of a main controller thread, a pool of
worker threads, and a complex set of synchronization primitives (barriers, locks,
and condition variables) to coordinate work within and between devices.
"""

from threading import Event, Thread, Lock, Condition
from reusable_barrier import ReusableBarrier

NUM_THREADS = 8

class Device(object):
    """
    Represents a device node, which encapsulates its own data, a controller
    thread, and a pool of worker threads for executing computational scripts.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the device and all associated threads and sync primitives.
        
        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): The local data store for this device.
            supervisor (Supervisor): The central supervisor for the simulation.
        """
        self.device_id = device_id
        self.supervisor = supervisor

        # Event to signal that the global device setup is complete.
        self.ready_to_start = Event()

        # Thread-safe access to local sensor data.
        self.data_lock = Lock()
        self.sensor_data = sensor_data

        # Manages the busy state of data locations within this device.
        self.location_busy = {location: False for location in self.sensor_data}
        self.location_busy_lock = Lock()

        # --- Script Management State ---
        self.scripts = []
        self.scripts_assigned = False # Flag: Have all scripts for the step been assigned?
        self.scripts_enabled = False  # Flag: Are workers allowed to process scripts?
        self.scripts_started_idx = 0  # Index for workers to pull from the scripts list.
        self.scripts_done_idx = 0     # Counter for completed scripts.

        # Condition variable to manage the producer-consumer flow of scripts.
        self.scripts_lock = Lock()
        self.scripts_condition = Condition(self.scripts_lock) # Workers wait on this.
        self.scripts_done_condition = Condition(self.scripts_lock) # Controller waits on this.

        # --- Thread Initialization ---
        self.thread_running = True
        self.thread = DeviceThread(self) # The main controller thread for this device.
        self.worker_threads = [ScriptWorker(self, i) for i in range(NUM_THREADS)]

        # Start all worker threads. The controller thread is started in setup_devices.
        for thread in self.worker_threads:
            thread.start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes global synchronization objects.
        
        Intended to be called by a single master device (device_id 0). It creates
        a global barrier for time-step synchronization and a shared dictionary of
        condition variables for location-based coordination.
        """
        if self.device_id == 0:
            timestep_barrier = ReusableBarrier(len(devices))
            location_conditions = {}

            # Create a condition variable for each unique data location across all devices.
            for device in devices:
                for location in device.sensor_data:
                    if not location in location_conditions:
                        location_conditions[location] = Condition()
            
            # Distribute the shared synchronization objects to all devices.
            for device in devices:
                device.location_conditions = location_conditions
                device.timestep_barrier = timestep_barrier

            # Signal all devices that the setup is complete and they can start.
            for device in devices:
                device.ready_to_start.set()

        self.thread.start()

    def assign_script(self, script, location):
        """

        Adds a script to the device's queue and notifies a worker.
        A 'None' script indicates that all scripts for the current step are assigned.
        """
        self.scripts_lock.acquire()
        if script is not None:
            self.scripts.append((script, location))
            self.scripts_condition.notify() # Wake up one worker thread.
        else:
            self.scripts_assigned = True
            # Wake up the main DeviceThread in case it's waiting for this signal.
            self.scripts_done_condition.notify_all()
        self.scripts_lock.release()

    def is_busy(self, location):
        """Thread-safely checks if a location is marked as busy."""
        self.location_busy_lock.acquire()
        ret = location in self.location_busy and self.location_busy[location]
        self.location_busy_lock.release()
        return ret

    def set_busy(self, location, value):
        """Thread-safely marks a location as busy or not busy."""
        self.location_busy_lock.acquire()
        self.location_busy[location] = value
        self.location_busy_lock.release()

    def has_data(self, location):
        """Thread-safely checks if the device contains data for a location."""
        self.data_lock.acquire()
        ret = location in self.sensor_data
        self.data_lock.release()
        return ret

    def get_data(self, location):
        """Thread-safely retrieves data from a given location."""
        self.data_lock.acquire()
        ret = self.sensor_data[location] if location in self.sensor_data else None
        self.data_lock.release()
        return ret

    def set_data(self, location, data):
        """Thread-safely sets data for a given location."""
        self.data_lock.acquire()
        if location in self.sensor_data:
            self.sensor_data[location] = data
        self.data_lock.release()

    def shutdown(self):
        """Waits for the controller and all worker threads to terminate."""
        self.thread.join()
        for thread in self.worker_threads:
            thread.join()

class ScriptWorker(Thread):
    """
    A worker thread that executes scripts. It acts as a consumer, pulling
    scripts from the device's shared script list.
    """
    def __init__(self, device, index):
        Thread.__init__(self, name="Worker thread %d for device %d" % (index, device.device_id))
        self.device = device
        self.lock = device.scripts_lock
        self.done_condition = device.scripts_done_condition
        self.condition = device.scripts_condition

    def run(self):
        """Main loop for the worker thread."""
        self.lock.acquire()

        while self.device.thread_running:
            # Wait until the controller enables scripts and there's a script to run.
            while self.device.thread_running and (not self.device.scripts_enabled or \
                   self.device.scripts_started_idx >= len(self.device.scripts)):
                self.condition.wait()

            if not self.device.thread_running:
                break
            
            # Atomically get the next script to run from the shared list.
            script = self.device.scripts[self.device.scripts_started_idx]
            self.device.scripts_started_idx += 1
            
            # Release the lock while running the script to allow other workers to run.
            self.lock.release()
            self.run_script(script[0], script[1])
            self.lock.acquire()

            # Atomically signal that one script is done.
            self.device.scripts_done_idx += 1
            self.done_condition.notify_all() # Notify the controller.

        self.lock.release()


    def run_script(self, script, location):
        """
        Executes a single script, handling complex cross-device synchronization.
        """
        # Acquire the condition variable for this location, which acts as a distributed lock.
        self.device.location_conditions[location].acquire()

        # Find all devices involved in this script (neighbors + self).
        script_devices = []
        for device in self.device.neighbours:
            if device.has_data(location):
                script_devices.append(device)
        if self.device.has_data(location):
            script_devices.append(self.device)

        if not script_devices:
            self.device.location_conditions[location].release()
            return

        # Wait until the location is free on ALL involved devices.
        while True:
            free = True
            for device in script_devices:
                if device.is_busy(location):
                    free = False
                    break
            if free:
                break
            # If not free, wait for another thread to notify on this location's condition.
            self.device.location_conditions[location].wait()

        # Mark the location as busy on all involved devices and gather data.
        script_data = []
        for device in script_devices:
            device.set_busy(location, True)
            script_data.append(device.get_data(location))
        self.device.location_conditions[location].notify_all()
        
        # Release lock, run the script (long operation), then re-acquire.
        self.device.location_conditions[location].release()
        result = script.run(script_data)
        self.device.location_conditions[location].acquire()

        # Write results back and mark locations as free.
        for device in script_devices:
            device.set_data(location, result)
            device.set_busy(location, False)
        # Notify any other threads waiting for this location to become free.
        self.device.location_conditions[location].notify_all()

        self.device.location_conditions[location].release()


class DeviceThread(Thread):
    """
    The main controller thread for a single device. It orchestrates the
    device's participation in the global, time-stepped simulation.
    """

    def __init__(self, device):
        """Initializes the controller thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main time-step loop for the device."""
        # Wait until global setup is complete.
        self.device.ready_to_start.wait()

        while True:
            # Synchronize with all other devices at the beginning of a time step.
            self.device.timestep_barrier.wait()

            # Fetch the list of neighbors for this time step.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            if self.device.neighbours is None:
                # A None value indicates the end of the simulation.
                break

            # --- Producer phase ---
            # Reset state for the new time step and enable worker threads.
            self.device.scripts_lock.acquire()
            self.device.scripts_started_idx = 0
            self.device.scripts_done_idx = 0
            self.device.scripts_enabled = True
            self.device.scripts_condition.notify_all() # Wake up waiting workers.
            self.device.scripts_lock.release()

            # --- Wait-for-completion phase ---
            # Wait until all scripts for this step have been assigned AND completed.
            self.device.scripts_lock.acquire()
            while not self.device.scripts_assigned or \
                  self.device.scripts_done_idx < len(self.device.scripts):
                self.device.scripts_done_condition.wait()
            
            # Disable workers and reset assignment flag for the next step.
            self.device.scripts_enabled = False
            self.device.scripts_assigned = False
            self.device.scripts_lock.release()

        # --- Shutdown phase ---
        # Signal all worker threads to terminate.
        self.device.scripts_lock.acquire()
        self.device.thread_running = False
        self.device.scripts_condition.notify_all()
        self.device.scripts_lock.release()
