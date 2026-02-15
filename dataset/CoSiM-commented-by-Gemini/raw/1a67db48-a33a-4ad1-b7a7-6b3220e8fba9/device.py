"""
This module defines a distributed device simulation framework where each device
manages a pool of persistent worker threads that handle both computation and
control flow for the simulation's time steps.
"""

from threading import Event, Thread, Lock, Condition

class ReusableBarrier(object):
    """
    A custom implementation of a reusable barrier using a Condition variable.
    
    This barrier blocks a set of threads until all of them have called the
    `wait()` method, at which point it releases them all and resets for reuse.
    """
    
    def __init__(self, num_threads):
        """Initializes the barrier for a specific number of threads."""
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """Blocks the calling thread until all threads have reached the barrier."""
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
    Represents a device node, which contains and manages a pool of worker threads.
    
    This class holds the device's state, data, and all the synchronization
    primitives required for its worker threads to coordinate among themselves
    and with other devices.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device and starts its pool of worker threads.
        
        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The local data store for this device.
            supervisor (Supervisor): The central simulation supervisor.
        """
        self.device_id = device_id
        self.thread_number = 8
        self.thread_lock = Lock() # Lock for coordinating setup tasks among workers.
        self.thread_barrier = ReusableBarrier(self.thread_number) # Local barrier for this device's workers.
        self.neighbours_list = []
        self.neighbours_list_collected = False # Flag to ensure neighbors are fetched once per step.
        self.scripts = []
        self.locations_list = {} # Shared dictionary of global location locks.
        self.global_barrier = None # Shared barrier for all threads across all devices.
        self.setup_event = Event() # Signals that global setup is complete.
        self.script_received = Event()
        self.timepoint_done = Event() # Signals that scripts are ready for processing.
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.threads = []
        for i in range(self.thread_number):
            self.threads.append(DeviceThread(self, i))
        for i in range(self.thread_number):
            self.threads[i].start()

    def __str__(self):
        """String representation of the Device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization objects.
        
        The master device (ID 0) creates the global barrier and location locks
        and signals other devices to copy the references.
        """
        if self.device_id == 0:
            # Create a shared lock for each unique data location.
            for device in devices:
                for loc in device.sensor_data:
                    self.locations_list.setdefault(loc, Lock())
            # Create a global barrier for all worker threads in the simulation.
            max_threads = self.thread_number * len(devices)
            self.global_barrier = ReusableBarrier(max_threads)
            self.setup_event.set() # Signal that setup is done.
        else:
            # Non-master devices wait for the master to finish setup.
            main_device = next((dev for dev in devices if dev.device_id == 0), None)
            main_device.setup_event.wait()
            # Copy references to the shared synchronization objects.
            self.global_barrier = main_device.global_barrier
            self.locations_list = main_device.locations_list
            self.setup_event.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device. A `None` script signals the end of
        assignments for the current step.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data from local storage.
        Note: This method is not thread-safe; locking is handled by the caller.
        """
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """
        Sets data in local storage.
        Note: This method is not thread-safe; locking is handled by the caller.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for all worker threads to complete."""
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    A persistent worker thread that executes scripts and handles synchronization.
    
    Each device runs a pool of these threads, which work together to manage
    the simulation's lifecycle for the device.
    """

    def __init__(self, device, my_id):
        """Initializes the worker thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.my_id = my_id

    def run(self):
        """The main simulation loop for the worker thread."""
        # Wait for the global device setup to complete.
        self.device.setup_event.wait()

        while True:
            # --- Per-Step Setup ---
            # Use a lock to ensure only one thread per device fetches the neighbors list.
            with self.device.thread_lock:
                if self.device.neighbours_list_collected is False:
                    self.device.neighbours_list_collected = True
                    self.device.neighbours_list = self.device.supervisor.get_neighbours()

            if self.device.neighbours_list is None:
                break # End of simulation.

            # All threads wait for the supervisor to finish assigning scripts for this step.
            self.device.timepoint_done.wait()

            # --- Computation Phase ---
            # Process a statically assigned subset of scripts using strided access.
            for i in xrange(self.my_id, len(self.device.scripts), self.device.thread_number):
                (script, location) = self.device.scripts[i]
                script_data = []
                
                # Acquire the global lock for this location to serialize access.
                self.device.locations_list[location].acquire()

                # Gather data from neighbors and self.
                for device in self.device.neighbours_list:
                    script_data.append(device.get_data(location))
                script_data.append(self.device.get_data(location))
                script_data = [d for d in script_data if d is not None]

                if script_data:
                    # Run script and distribute results.
                    result = script.run(script_data)
                    for device in self.device.neighbours_list:
                        device.set_data(location, result)
                    self.device.set_data(location, result)
                
                # Release the global location lock.
                self.device.locations_list[location].release()

            # --- Synchronization Phase ---
            # All threads from all devices synchronize here after computation.
            self.device.global_barrier.wait()

            # Use a lock to ensure only one thread per device resets the state for the next step.
            with self.device.thread_lock:
                if self.device.neighbours_list_collected is True:
                    self.device.neighbours_list_collected = False
                    self.device.timepoint_done.clear()
            
            # All threads within this device synchronize here before starting the next step.
            self.device.thread_barrier.wait()
