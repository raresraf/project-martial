"""
Models a distributed system of devices executing scripts in synchronized timepoints.

This file defines a simulation where multiple devices, each with its own thread,
operate in parallel. Device 0 acts as a coordinator for setup. All devices
synchronize at the beginning of each time step using a reusable barrier. Data
at specific locations is protected by locks that are dynamically created
as locations are accessed.
"""

from threading import Event, Thread, Lock
from utils import ReusableBarrier


class Device(object):
    """Represents a device in a simulated distributed environment.

    Each device has sensor data and can execute scripts. It coordinates with
    other devices using a shared barrier and locks for data consistency.

    Attributes:
        device_id (int): Unique identifier for the device.
        sensor_data (dict): Data store for the device, mapping locations to values.
        supervisor (Supervisor): An external entity managing the simulation's progress.
        scripts (list): A list of (script, location) tuples to be executed.
        timepoint_done (Event): An event to signal that all scripts for a timepoint
                                have been assigned and execution can begin.
        thread (DeviceThread): The worker thread that runs the device's logic.
        common_barrier (ReusableBarrier): A barrier shared among all devices to
                                          synchronize the start of each timepoint.
        wait_initialization (Event): An event used by device 0 to signal that
                                     global setup is complete.
        locations_locks (dict): A shared dictionary mapping locations to Lock objects.
        lock_location_dict (Lock): A lock to protect access to the locations_locks dict.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self, 0)

        # Barrier is initialized during setup by device 0.
        self.common_barrier = None
        
        # Event for coordinating the startup sequence.
        self.wait_initialization = Event()

        # Locks are initialized during setup by device 0.
        self.locations_locks = None
        
        # Lock to ensure thread-safe creation of location locks.
        self.lock_location_dict = Lock()

    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up the simulation environment and starts all device threads.

        If this device is ID 0, it acts as the coordinator: it creates the shared
        barrier and lock dictionary, distributes them to all other devices, and then
        signals them to start. Other devices wait for this signal.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if not self.device_id == 0:
            # Worker devices wait for the coordinator (device 0) to finish setup.
            self.wait_initialization.wait()
            self.thread.start()
        else:
            # Device 0 is the coordinator.
            
            # This dictionary will be shared by all devices to store location-specific locks.
            self.locations_locks = {}

            # Create and distribute the shared barrier.
            barrier_size = len(devices)
            self.common_barrier = ReusableBarrier(len(devices))

            for dev in devices:
                dev.common_barrier = self.common_barrier
                dev.locations_locks = self.locations_locks
            
            # Signal worker devices that setup is complete and they can start.
            for dev in devices:
                if not dev.device_id == 0:
                    dev.wait_initialization.set()

            self.thread.start()

    def assign_script(self, script, location):
        """Assigns a script to the device for the current timepoint.

        Args:
            script (Script): The script to be executed.
            location (int): The data location the script pertains to.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signals that all scripts for the timepoint have been received.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves data from a specific sensor location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates data at a specific sensor location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the device's thread to terminate."""
        self.thread.join()


class DeviceThread(Thread):
    """The main worker thread for a device.
    
    This thread manages the device's lifecycle through each timepoint of the
    simulation, including synchronization, script execution, and data access.
    """

    def __init__(self, device, th_id):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)


        self.device = device
        self.th_id = th_id

    def run(self):
        """The main execution loop for the device."""
        while True:
            # All threads wait at the barrier, synchronizing the start of a timepoint.
            self.device.common_barrier.wait()

            # The th_id check suggests a design that may have intended multiple
            # threads per device, but here th_id is always 0.
            if self.th_id == 0:
                neighbours = self.device.supervisor.get_neighbours()
                if neighbours is None:
                    # Supervisor signaling the end of the simulation.
                    break
            else:
                # This branch is currently unreachable as th_id is always 0.
                pass

            # Wait for the supervisor to assign all scripts for this timepoint.
            self.device.timepoint_done.wait()

            
            current_scripts = self.device.scripts

            
            for (script, location) in current_scripts:
                # Acquire a lock on the lock dictionary to safely check for/create a location lock.
                self.device.lock_location_dict.acquire()

                # Dynamically create a lock for a location if it's the first time it's accessed.
                if not self.device.locations_locks.has_key(location):
                    self.device.locations_locks[location] = Lock()

                # Acquire the specific lock for the location to ensure data consistency.
                self.device.locations_locks[location].acquire()
                
                # Release the dictionary lock once the location lock is held.
                self.device.lock_location_dict.release()

                script_data = []
                # Collect data from all neighbors for the specified location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Collect data from the current device as well.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Execute the script with the collected data.
                    result = script.run(script_data)

                    # Update the data on all neighbors and the current device with the result.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                # Release the lock for the location.
                self.device.locations_locks[location].release()
            
            # Clear the event and script list for the next timepoint.
            self.device.timepoint_done.clear()

# The following code appears to be a bundled dependency.
from threading import Semaphore, Lock


class ReusableBarrier(object):
    """A reusable barrier implementation using semaphores.
    
    This barrier synchronizes a fixed number of threads at a rendezvous point.
    It is reusable, meaning it can be used multiple times. It employs a two-phase
    protocol to prevent threads from one barrier instance from proceeding into
    the next instance before all threads have left the first one.
    """

    def __init__(self, num_threads):
        """Initializes the barrier for a given number of threads."""
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]

        
        self.count_lock = Lock()

        
        self.threads_sem1 = Semaphore(0)

        
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """Implements one phase of the barrier.
        
        Args:
            count_threads (list): A list containing the current thread count for this phase.
            threads_sem (Semaphore): The semaphore to block/release threads for this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # The last thread to arrive releases all other waiting threads.
                for i in range(self.num_threads):
                    
                    
                    threads_sem.release()
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        
        # Threads wait here until released by the last thread.
        threads_sem.acquire()
        
