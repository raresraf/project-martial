
"""
Models a distributed system of devices that process sensor data in parallel.

This module defines a simulation framework for a network of devices that
collaborate on processing sensor data. The simulation operates in discrete
time-steps (timepoints), synchronized across all devices using a reusable
barrier. Within each timepoint, devices execute assigned scripts in parallel
using a pool of worker threads. Locks are used to ensure exclusive access to
data at specific locations.
"""

from threading import Event, Thread, Lock
import cond_barrier


class Device(object):
    """Represents a single device in the distributed simulation.

    Each device manages its own sensor data and executes scripts assigned by a
    supervisor. It uses a dedicated thread (`DeviceThread`) to manage its
    lifecycle and a pool of worker threads for script execution.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's sensor data,
                            keyed by location.
        supervisor: An object that manages the network of devices and provides
                    neighbour information.
        script_received (Event): An event that is set when a new script is
                                 assigned to the device.
        scripts (list): A list of tuples, where each tuple contains a script
                        and its corresponding location.
        timepoint_done (Event): An event that signals the completion of a
                                timepoint, triggering the device's main loop.
        thread (DeviceThread): The main thread for this device.
        dict_location (dict): A shared dictionary mapping locations to locks,
                              ensuring exclusive access.
        barrier (ReusableBarrierCond): A shared barrier for synchronizing all
                                       devices at the end of a timepoint.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes a Device instance."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []


        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.dict_location = {}
        self.barrier = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes shared resources for all devices in the simulation.

        This method is intended to be called on a single device (typically the
        one with device_id 0) to set up the shared barrier and location-lock
        dictionary for all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """

        # This setup is centralized and performed by one device (e.g., device 0)
        # to initialize shared state for all other devices.
        if self.device_id == 0:
            num_threads = len(devices)
            
            # The barrier synchronizes all device threads at the end of a timepoint.
            self.barrier = cond_barrier.ReusableBarrierCond(num_threads)
            for device in devices:
                device.barrier = self.barrier
                device.dict_location = self.dict_location

    def assign_script(self, script, location):
        """Assigns a script to be executed at a specific location.

        If a script is provided, it is added to the device's script queue. If
        the script is None, it signals the end of the current timepoint.

        Args:
            script: The script to be executed.
            location: The location associated with the script.
        """
        
        # A lock is created for a location if one does not already exist. This
        # ensures that worker threads can later acquire a lock for that location.
        if location not in self.dict_location:
            self.dict_location[location] = Lock()

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals that all scripts for the current timepoint
            # have been assigned, allowing the device thread to proceed.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location.

        Args:
            location: The location for which to retrieve data.

        Returns:
            The sensor data at the given location, or None if not available.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """Updates sensor data at a given location.

        Args:
            location: The location at which to update data.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class Worker(Thread):
    """A thread that executes scripts on sensor data.

    Workers are created by a DeviceThread to perform the actual data processing.
    Each worker can be assigned multiple scripts to execute. It acquires a lock
    for the script's location before processing to prevent race conditions.
    """

    def __init__(self, worker_id, neighbours, device, dict_location):
        """Initializes a Worker thread."""
        Thread.__init__(self, name="Worker Thread %d" % worker_id)
        self.worker_id = worker_id
        self.neighbours = neighbours
        self.device = device
        self.dict_location = dict_location
        self.scripts = []
        self.location = []

    def addwork(self, script, location):
        """Adds a script to the worker's queue.

        Args:
            script: The script to be executed.
            location: The location associated with the script.
        """
        self.scripts.append(script)
        self.location.append(location)

    def run(self):
        """The main execution loop for the worker thread.

        Iterates through its assigned scripts, acquires the necessary lock,
        gathers data from its device and its neighbors, runs the script, and
        updates the data on all relevant devices.
        """
        i = 0
        for script in self.scripts:
            # Pre-condition: Acquire lock to ensure exclusive access to the
            # sensor data at this location.
            self.dict_location[self.location[i]].acquire()
            script_data = []
            
            # Gather data from all neighboring devices at the specified location.
            for device in self.neighbours:
                data = device.get_data(self.location[i])
                if data is not None:
                    script_data.append(data)
            
            # Include the device's own data for that location.
            data = self.device.get_data(self.location[i])
            if data is not None:
                script_data.append(data)

            if script_data != []:
                # The script is executed with the collected data.
                result = script.run(script_data)

                
                # The result of the script is broadcast back to the neighbors
                # and the device itself, updating the state for the location.
                for device in self.neighbours:
                    device.set_data(self.location[i], result)
                
                self.device.set_data(self.location[i], result)
            # Post-condition: Release the lock after processing is complete.
            self.dict_location[self.location[i]].release()
            i = i + 1

class DeviceThread(Thread):
    """The main control thread for a single Device.

    This thread orchestrates the device's participation in the simulation. It
    waits for a timepoint to begin, spawns worker threads to execute scripts,
    and then synchronizes with all other devices using a shared barrier before
    starting the next timepoint.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main execution loop for the device thread."""
        
        # Loop indefinitely, processing one timepoint per iteration.
        while True:
            
            # Get the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value for neighbours signals the end of the simulation.
                break

            # Block until the supervisor signals the start of a new timepoint
            # by calling assign_script with a None script.
            self.device.timepoint_done.wait()

            nr_worker = 0
            num_threads = 8 # Defines the size of the worker thread pool.
            workers = []

            for i in range(num_threads):
                
                
                lock_loc = self.device.dict_location
                workers.append(Worker(i, neighbours, self.device, lock_loc))

            # Distribute the assigned scripts among the worker threads in a
            # round-robin fashion.
            for (script, location) in self.device.scripts:
                workers[nr_worker].addwork(script, location)
                nr_worker = nr_worker + 1
                if nr_worker == 8:
                    nr_worker = 0

            
            # Start all worker threads to execute the scripts in parallel.
            for i in range(num_threads):
                workers[i].start()
            # Wait for all worker threads to complete their assigned tasks.
            for i in range(num_threads):
                workers[i].join()

            # Clear the event to prepare for the next timepoint.
            self.device.timepoint_done.clear()
            
            # Synchronize with all other devices. No device will proceed to
            # the next timepoint until all have reached this barrier.
            self.device.barrier.wait()
