"""
This module implements a simulation framework for a network of interconnected devices.

It defines the `Device` class, which represents a node in the network, and a
`DeviceThread` that manages the device's lifecycle and script execution. The
simulation proceeds in synchronized time steps, where each device can run
computational scripts on sensor data. Scripts can access data from the local
device and its neighbors. The framework uses various threading primitives to
manage concurrency and ensure data consistency.
"""

from threading import Event, Thread, Lock, Semaphore
from cond_barrier import ReusableBarrier


class Device(object):
    """Represents a single device in the distributed network simulation.

    Each device runs in its own thread, executes assigned scripts, and
    communicates with its neighbors. It holds sensor data and manages
    synchronization with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local
                                sensor data, keyed by location.
            supervisor (Supervisor): An object that manages the network topology,
                                   providing neighbor information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.scripts_to_run = []
        # Event to signal that all scripts for a timepoint have been assigned.
        self.timepoint_done = Event()
        # Event to signal that the initial setup of shared resources is complete.
        self.setup_done = Event()
        self.thread = DeviceThread(self)

        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Sets up shared synchronization objects for the entire device network.

        This method must be called before the simulation begins. It establishes
        a shared barrier for timepoint synchronization and a set of shared locks
        to protect access to shared resources and data locations.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        nr_devices = len(devices)
        self.barrier = ReusableBarrier(nr_devices)
        self.lock_get_neigh = Lock()
        self.lock_location = {}
        self.lock_check_loc = Lock()
        self.lock_scripts = Lock()

        # All devices share the same set of synchronization primitives,
        # initialized by the first device.
        self.barrier = devices[0].barrier
        self.lock_get_neigh = devices[0].lock_get_neigh
        self.lock_location = devices[0].lock_location
        self.lock_check_loc = devices[0].lock_check_loc

        for location in self.sensor_data:
            if not self.lock_location.has_key(location):
                # Create a unique lock for each sensor data location.
                self.lock_location[location] = Lock()

        self.setup_done.set()

    def assign_script(self, script, location):
        """Assigns a script to be run by the device in the current timepoint.

        Args:
            script (Script): The script object to be executed.
            location (any): The data location the script will operate on.
        """
        if script is not None:
            self.lock_scripts.acquire()
            self.scripts.append((script, location))
            self.scripts_to_run.append((script, location))
            self.lock_scripts.release()
        else:
            # A None script signals that no more scripts will be assigned for this timepoint.
            self.lock_scripts.acquire()
            self.timepoint_done.set()
            self.lock_scripts.release()

    def get_data(self, location):
        """Retrieves sensor data for a given location.

        Returns:
            The data if the location exists, otherwise None.
        """
        return (self.sensor_data[location]
                if location in self.sensor_data else None)

    def set_data(self, location, data):
        """Updates the sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main execution thread for a Device.

    This thread manages the device's lifecycle through a series of synchronized
    time steps. In each step, it executes assigned scripts and waits at a
    barrier for all other devices to complete.
    """

    def __init__(self, device):
        """Initializes the device thread.

        Args:
            device (Device): The parent device object this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # Limits the number of concurrently running scripts for this device to 8.
        self.sem_threads = Semaphore(8) 

    def run(self):
        """The main simulation loop for the device."""
        # Wait until the network-wide setup is complete.
        self.device.setup_done.wait()

        # Each iteration of this loop represents one timepoint in the simulation.
        while True:
            threads = []

            # Safely get the list of neighbors for this timepoint.
            self.device.lock_get_neigh.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.lock_get_neigh.release()

            # A None value for neighbours is the signal to terminate the simulation.
            if neighbours is None:
                break

            # Prepare the list of scripts to run for this timepoint.
            self.device.lock_scripts.acquire()
            self.device.scripts_to_run = self.device.scripts[:]
            finished = (self.device.timepoint_done.is_set() and
                        len(self.device.scripts_to_run) == 0)
            self.device.lock_scripts.release()

            # Process all scripts assigned for the current timepoint.
            while not finished:
                self.device.lock_scripts.acquire()
                local_scripts_to_run = self.device.scripts_to_run[:]
                self.device.lock_scripts.release()

                for (script, location) in local_scripts_to_run:
                    # Limit the number of concurrent script executions.
                    self.sem_threads.acquire()

                    # Check if another thread is already processing this location
                    # without blocking.
                    self.device.lock_check_loc.acquire()

                    if self.device.lock_location[location].locked():
                        # If locked, another script is active on this location. Skip for now.
                        self.device.lock_check_loc.release()
                        self.sem_threads.release()
                        continue

                    # Acquire the lock for this location to prevent race conditions.
                    self.device.lock_location[location].acquire()

                    # Remove script from the list of scripts to run.
                    self.device.lock_scripts.acquire()
                    self.device.scripts_to_run.remove((script, location))
                    self.device.lock_scripts.release()

                    self.device.lock_check_loc.release()

                    # Start the script execution in a new thread.
                    thread = Thread(target=run_script, args=(self, neighbours,
                                                             script, location))
                    threads.append(thread)
                    thread.start()
                
                # Check if all scripts for this timepoint are done.
                self.device.lock_scripts.acquire()
                finished = (self.device.timepoint_done.is_set() and
                            len(self.device.scripts_to_run) == 0)
                self.device.lock_scripts.release()

            # Wait for all script threads of this timepoint to complete.
            for thread in threads:
                thread.join()

            # Reset for the next timepoint and wait for all devices to be ready.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()

def run_script(parent_device_thread, neighbours, script, location):
    """Executes a single script.

    This function gathers data from the parent device and its neighbors at a
    specific location, runs the script with the collected data, and then
    propagates the result back to the same set of devices.

    Args:
        parent_device_thread (DeviceThread): The thread that spawned this script.
        neighbours (list): A list of neighbor Device objects.
        script (Script): The script to execute.
        location (any): The data location to operate on.
    """
    script_data = []
    
    # Gather data from all neighbors for the specified location.
    for device in neighbours:
        data = device.get_data(location)
        if data is not None:
            script_data.append(data)
    
    # Gather data from the parent device itself.
    data = parent_device_thread.device.get_data(location)
    if data is not None:
        script_data.append(data)

    if script_data != []:
        # Run the script's core logic with the aggregated data.
        result = script.run(script_data)

        # Distribute the result back to the parent and its neighbors.
        for device in neighbours:
            device.set_data(location, result)
        
        parent_device_thread.device.set_data(location, result)

    # Release the lock for the location and the semaphore slot.
    parent_device_thread.device.lock_location[location].release()
    parent_device_thread.sem_threads.release()
