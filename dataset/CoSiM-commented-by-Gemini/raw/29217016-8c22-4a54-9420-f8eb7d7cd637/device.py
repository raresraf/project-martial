"""
A simulation framework for a network of data-processing devices.

This module defines a `Device` class that operates within a synchronized,
time-stepped simulation. Each device runs its own control thread (`DeviceThread`)
which, at each time step, spawns a pool of worker threads to execute
dynamically assigned scripts. The system uses shared locks and a reusable
barrier to coordinate data access and synchronize all devices between time steps.
"""

from threading import Thread, Event, Lock
from barrier import ReusableBarrier

class Device(object):
    """
    Represents a single device, managing its data, state, and execution.
    
    Each device is initialized with its own sensor data and a connection to a
    supervisor. It participates in a coordinated setup process and then enters
    a loop of executing scripts in synchronized timepoints.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of the device's internal sensor data.
            supervisor: The central supervisor managing the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None
        self.locations = []
        self.locks = []


    def __str__(self):
        """Returns the string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes and distributes shared resources across all devices.

        This method implements a two-phase setup:
        1. The device with `device_id == 0` acts as a master, discovering all
           unique data locations from all devices and creating a corresponding
           set of locks and a shared `ReusableBarrier`.
        2. All other devices find the master device and copy the references to
           these shared `locations`, `locks`, and `barrier` objects.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            # Block Logic: Master device (ID 0) discovers all data locations.
            for device in devices:
                for data in device.sensor_data:
                    if data not in self.locations:
                        self.locations.append(data)
            self.locations.sort()

            # Create a shared lock for each unique location.
            for i in xrange(len(self.locations)):
                self.locks.append(Lock())
            
            # Create a barrier to synchronize all devices at the end of a timepoint.
            self.barrier = ReusableBarrier(len(devices))
        else:
            # Block Logic: All other devices find and copy the shared resources
            # from the master device.
            for device in devices:
                if device.device_id == 0:
                    self.locations = device.locations
                    self.locks = device.locks
                    self.barrier = device.barrier


    def assign_script(self, script, location):
        """
        Receives a script from the supervisor for the current timepoint.

        If a `None` script is received, it signals the end of the assignment
        phase for the current timepoint by setting the `timepoint_done` event.

        Args:
            script: The script object to execute.
            location: The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a specific location."""
        return self.sensor_data[location] \
        if location in self.sensor_data else None

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Waits for the main device thread to complete."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control thread for a device, orchestrating its lifecycle.
    
    This thread manages the device's execution within synchronized timepoints.
    In each timepoint, it creates a temporary pool of worker threads that
    compete to execute tasks from a shared script queue.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        Args:
            device (Device): The parent Device instance this thread controls.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.scripts = []
        self.neighbours = []

    def run(self):
        """
        The main execution loop, organized into discrete timepoints.
        """
        while True:
            # Get the current set of neighbours from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()

            # A `None` response signals the end of the simulation.
            if self.neighbours is None:
                break

            # Wait until the supervisor has finished assigning all scripts for this timepoint.
            self.device.timepoint_done.wait()
            num_scripts = len(self.device.scripts)
            self.scripts.extend(self.device.scripts)

            # Determine the size of the worker thread pool for this timepoint.
            if num_scripts < 8:
                num_threads = num_scripts
            else:
                num_threads = 8

            # Block Logic: Create and start a pool of worker threads.
            threads = []
            for i in xrange(num_threads):
                thread = Thread(target=self.run_script)
                thread.start()
                threads.append(thread)

            # Wait for all worker threads in the pool to complete their tasks.
            for i in xrange(len(threads)):
                threads[i].join()

            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()
            # Invariant: All devices must wait at the barrier, ensuring they all
            # complete the current timepoint before any can proceed to the next.
            self.device.barrier.wait()

    def run_script(self):
        """
        The target function for worker threads.
        
        Workers repeatedly pull scripts from a shared queue (`self.scripts`)
        until the queue is empty. For each script, a worker acquires the
        appropriate lock, processes data, and broadcasts the result.
        """
        # Block Logic: Implements a work-stealing pattern where threads from the pool
        # compete to pop scripts from the shared list until it's empty.
        while not self.scripts == []:
            (script, location) = self.scripts.pop()

            # Acquire the lock for the specific data location to prevent race conditions.
            self.device.locks[location].acquire()
            script_data = []
            # Block Logic: Aggregate data from neighbouring devices.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)

            # Add the device's own data to the set.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Invariant: Only run the script if there is data to process.
            if script_data != []:
                result = script.run(script_data)

                # Block Logic: Broadcast the result to all neighbours and self.
                for device in self.neighbours:
                    device.set_data(location, result)
                self.device.set_data(location, result)
            self.device.locks[location].release()