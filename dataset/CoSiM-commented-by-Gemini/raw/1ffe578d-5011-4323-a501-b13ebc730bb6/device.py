
"""
Models a distributed system of devices processing sensor data in parallel.

This module provides a framework for simulating a network of devices that
execute computational scripts on sensor data. The simulation progresses in
synchronized time-steps. Unlike other implementations, this version spawns a new
"slave" thread for each script to be executed in a timepoint and uses a custom
semaphore-based reusable barrier for synchronization.
"""

from threading import *


class Device(object):
    """Represents a single device in the distributed simulation.

    Each device holds sensor data and is assigned scripts by a supervisor.
    It manages a main `DeviceThread` which in turn spawns `SlaveThread`s for
    executing scripts.

    Attributes:
        device_id (int): Unique identifier for the device.
        sensor_data (dict): A dictionary mapping locations to sensor values.
        supervisor: An external object that provides neighbor information.
        scripts (list): A list of (script, location) tuples to be executed.
        timepoint_done (Event): An event signaling that all scripts for the
                                current timepoint have been assigned.
        lock_location (list): A list of locks, indexed by location, to ensure
                              exclusive access to sensor data.
        time_barrier (ReusableBarrierSem): A shared barrier to synchronize all
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

        self.lock_data = Lock()
        self.lock_location = []
        self.time_barrier = None

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """Initializes shared resources for all devices in the simulation.

        This method, intended to be called on a single master device (id 0),
        creates and distributes the shared time barrier and location-based locks
        to all devices in the simulation.

        Args:
            devices (list): A list of all Device objects.
        """
        
        if self.device_id == 0:
            # Create a reusable barrier for all device threads.
            self.time_barrier = ReusableBarrierSem(len(devices))

            for device in devices:
                device.time_barrier = self.time_barrier

            # Determine the total number of unique locations to create locks for.
            loc_num = 0
            for device in devices:
                for location in device.sensor_data:
                    loc_num = max(loc_num, location)
            # Create a list of locks, one for each location.
            for i in range(loc_num + 1):
                self.lock_location.append(Lock())

            # Distribute the shared list of locks to all devices.
            for device in devices:
                device.lock_location = self.lock_location

    def assign_script(self, script, location):
        """Assigns a script to the device for a specific location.

        Args:
            script: The script to execute. If None, it signals the end of the
                    current timepoint's script assignments.
            location: The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location.

        Args:
            location (int): The location identifier.

        Returns:
            The sensor data value, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """Thread-safely updates sensor data at a given location.

        Args:
            location (int): The location identifier.
            data: The new data value.
        """
        with self.lock_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the device by joining its main thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main control thread for a Device.

    This thread orchestrates the device's work within each timepoint. It waits
    for scripts, spawns a new `SlaveThread` for each script, waits for them to
    complete, and then synchronizes with other devices at the barrier.
    """

    def __init__(self, device):
        """Initializes the DeviceThread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """Main execution loop for the device."""
        while True:
            slaves = []
            
            neighbours = self.device.supervisor.get_neighbours()
            # A None value for neighbours signals the simulation's end.
            if neighbours is None:
                break

            # Wait until the supervisor signals all scripts for the timepoint are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # For each script, spawn a new SlaveThread to execute it.
            # This is less efficient than a fixed worker pool due to thread
            # creation/destruction overhead.
            for (script, location) in self.device.scripts:
                slave = SlaveThread(script, location, neighbours, self.device)
                slaves.append(slave)
                slave.start()

            # Wait for all slave threads for this timepoint to complete.
            for i in range(len(slaves)):
                slaves.pop().join()
            
            # Wait at the barrier to synchronize with all other devices
            # before proceeding to the next timepoint.
            self.device.time_barrier.wait()

class SlaveThread(Thread):
    """A short-lived worker thread to execute a single script."""
    def __init__(self, script, location, neighbours, device):
        """Initializes the SlaveThread."""
        Thread.__init__(self, name="Slave Thread of Device %d" % device.device_id)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """Executes a single script."""
        device = self.device
        script = self.script
        location = self.location
        neighbours = self.neighbours
        
        data = device.get_data(location)
        input_data = []
        # Get the specific lock for the location this script operates on.
        this_lock = device.lock_location[location]

        if data is not None:
            input_data.append(data)

        # Pre-condition: Acquire the lock for this location to ensure exclusive
        # access to the data of all devices at this location.
        with this_lock:
            # Aggregate data from all neighbors.
            for neighbour in neighbours:
                temp = neighbour.get_data(location)

                if temp is not None:
                    input_data.append(temp)
            
            # If any data was found, run the script.
            if input_data != []:
                result = script.run(input_data)

                # Broadcast the result back to all neighbors and the device itself.
                for neighbour in neighbours:
                    neighbour.set_data(location, result)

                device.set_data(location, result)


class ReusableBarrierSem():
    """A custom reusable barrier implemented using semaphores.

    This barrier synchronizes multiple threads at a point, and can be used
    repeatedly. It works in two phases to prevent threads from one iteration
    from passing the barrier before all threads from the previous iteration
    have finished.
    """
    
    def __init__(self, num_threads):
        """Initializes the reusable barrier."""
        self.num_threads = num_threads
        # Counter for threads arriving at the first phase.
        self.count_threads1 = self.num_threads
        # Counter for threads arriving at the second phase.
        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock()
        # Semaphores to block threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            # The last thread to arrive releases all other waiting threads.
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset for the next iteration's phase 2.
                self.count_threads2 = self.num_threads
         
        # All threads will block here until the last thread releases the semaphore.
        self.threads_sem1.acquire()
         
    def phase2(self):
        """Second synchronization phase to ensure reusability."""
        with self.counter_lock:
            self.count_threads2 -= 1
            # The last thread to arrive at phase 2 releases all for the next round.
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset for the next iteration's phase 1.
                self.count_threads1 = self.num_threads
         
        # All threads block here, completing the two-phase synchronization.
        self.threads_sem2.acquire()
