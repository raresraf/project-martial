"""
This module provides a framework for simulating a network of communicating devices.

It defines a `Device` class representing a node in the network, a `DeviceThread`
that runs the core logic for each device, and a custom `ReusableBarrierSem` for
synchronizing the devices at each time step. The simulation appears to model a
system where devices perform computations based on their own data and data from
their neighbors in discrete, synchronized rounds.
"""
from threading import *


class ReusableBarrierSem():
    """A reusable barrier implemented using semaphores.

    This barrier synchronizes a fixed number of threads at a rendezvous point.
    It is reusable, meaning threads can wait on it multiple times. It uses a
    two-phase signaling mechanism to prevent race conditions where faster threads
    could loop around and re-enter the barrier before slower threads have exited.
    """

    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()      # Lock to protect the internal thread counters.
        self.threads_sem1 = Semaphore(0)  # Semaphore for the first phase of the barrier.
        self.threads_sem2 = Semaphore(0)  # Semaphore for the second phase of the barrier.

    def wait(self):
        """Causes a thread to wait until all `num_threads` have called this method."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """The first phase of the barrier wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread to arrive releases all waiting threads.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for next use.
        
        self.threads_sem1.acquire() # Threads wait here until released.

    def phase2(self):
        """The second phase of the barrier wait, preventing race conditions."""
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # The last thread to arrive releases all waiting threads.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for next use.

        self.threads_sem2.acquire() # Threads wait here until released.


from threading import Event, Thread


class Device(object):
    """Represents a single device in a simulated distributed network."""

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self, 0)
    
        # --- Synchronization and shared state attributes ---
        self.com_barrier = None  # The shared barrier for all devices.
        self.initialize = Event() # Used to sequence the setup process.
        self.locked_locations = None # Shared dictionary of locks for data locations.
        self.lock_dict = Lock() # Lock to protect access to the locked_locations dictionary.

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the synchronization objects for a list of devices.
        Device 0 acts as the coordinator, creating the shared barrier and
        the dictionary for location-based locks.
        """
        if self.device_id != 0:
            # Non-coordinator devices wait for the setup to complete.
            self.initialize.wait()
        else:
            # Device 0 initializes the shared objects.
            self.com_barrier = ReusableBarrierSem(len(devices))
            self.locked_locations = {}
            for d in devices:
                # Distribute the shared objects to all other devices.
                d.com_barrier = self.com_barrier
                d.locked_locations = self.locked_locations
                if (d.device_id == 0):
                    pass
                else:
                    # Signal other devices that initialization is complete.
                    d.initialize.set()
            
        self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to be executed at a specific location for the current timepoint."""
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals the end of script assignment for the timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins the device's thread, ensuring clean termination."""
        self.thread.join()


class DeviceThread(Thread):
    """The worker thread that executes the main operational logic for a Device."""

    def __init__(self, device, thread_id):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """The main loop of the device, executed in discrete, synchronized timepoints."""
        while True:
            # --- Synchronization Point ---
            # All devices wait here, ensuring they start the timepoint together.
            self.device.com_barrier.wait()

            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbours is the signal to terminate.
                break

            # Wait until all scripts for this timepoint have been assigned.
            self.device.timepoint_done.wait()

            # --- Script Execution Phase ---
            for (script, location) in self.device.scripts:
                # Acquire a lock to ensure exclusive access to the location's lock.
                self.device.lock_dict.acquire()

                # Lazily initialize a lock for the specific data location if it doesn't exist.
                if (location not in self.device.locked_locations):
                    self.device.locked_locations[location] = Lock()
                
                # Acquire the lock for the specific location to prevent data races.
                self.device.locked_locations[location].acquire();
                self.device.lock_dict.release()

                # --- Data Gathering ---
                script_data = []
                # Gather data from all neighboring devices for the target location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Gather this device's own data for the location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # --- Computation and Data Propagation ---
                if script_data:
                    # Run the assigned script on the collected data.
                    result = script.run(script_data)

                    # Broadcast the result to all neighbors.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Update this device's own data with the result.
                    self.device.set_data(location, result)

                # Release the lock for the location, allowing others to process it.
                self.device.locked_locations[location].release();

            # Clear the event to prepare for the next timepoint.
            self.device.timepoint_done.clear()
