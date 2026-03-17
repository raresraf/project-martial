"""
This module implements a distributed device simulation using a master-worker
threading model and a two-phase reusable barrier for synchronization.

The simulation is initialized by a master device (ID 0) which creates and
distributes shared synchronization objects: a semaphore-based reusable barrier
and a list of locks indexed by data location. Each device runs a main
`DeviceThread` that orchestrates work for discrete timepoints. For each
timepoint, it spawns a new `SlaveThread` for every assigned script, which
handles the computation and data distribution for that single script.
"""

from threading import *


class Device(object):
    """
    Represents a single device in the simulated network.

    This class holds the device's state, including its sensor data and assigned
    scripts, and manages a main orchestrator thread (`DeviceThread`).

    Attributes:
        device_id (int): The unique identifier for the device.
        sensor_data (dict): A dictionary holding the device's local sensor readings.
        supervisor: The central entity managing the network topology.
        lock_location (list): A shared list of Lock objects, indexed by location.
        time_barrier (ReusableBarrierSem): A shared barrier for timepoint synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes the Device instance.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor: The central supervisor for the simulation.
        """
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
        """
        Initializes and distributes shared synchronization objects for the network.

        This method must be called on the master device (ID 0). It determines the
        total number of unique data locations, creates a lock for each, and
        creates a shared barrier. These objects are then assigned to all devices.
        """
        
        # Block Logic: Only the device with ID 0 can coordinate this setup.
        if self.device_id == 0:
            self.time_barrier = ReusableBarrierSem(len(devices)) 

            for device in devices:
                device.time_barrier = self.time_barrier

            loc_num = 0
            # Determine the maximum location ID to size the lock list.
            for device in devices:
                for location in device.sensor_data:
                    loc_num = max(loc_num, location) 
            for i in range(loc_num + 1):
                self.lock_location.append(Lock()) 

            # Distribute the list of location-based locks to all devices.
            for device in devices:
                device.lock_location = self.lock_location 

    def assign_script(self, script, location):
        """
        Assigns a script to the device for the current timepoint.

        Args:
            script: The script object to execute.
            location (int): The data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script marks the end of script assignment for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location index.
        
        Returns:
            The data for the location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data at a given location in a thread-safe manner.

        Args:
            location (int): The location index.
            data: The new data value.
        """
        with self.lock_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the main device thread."""
        self.thread.join()


class DeviceThread(Thread):
    """The main orchestrator thread for a Device, managing timepoint cycles."""

    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main control loop, executing timepoint by timepoint."""
        while True:
            slaves = []
            
            neighbours = self.device.supervisor.get_neighbours()
            # A None value for neighbors is the shutdown signal.
            if neighbours is None:
                break

            # Waits until all scripts for the current timepoint are assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() 

            
            # Block Logic: Spawns a new worker thread for each assigned script.
            for (script, location) in self.device.scripts:
                slave = SlaveThread(script, location, neighbours, self.device) 
                slaves.append(slave)
                slave.start()

            # Block Logic: Waits for all spawned slave threads to complete.
            # BUG: This loop only joins half of the threads. `slaves.pop()` modifies
            # the list while it's being implicitly iterated over by `range(len(slaves))`.
            for i in range(len(slaves)):
                slaves.pop().join()

            # === SYNC BARRIER ===
            # Waits for all devices to finish their timepoint before starting the next.
            self.device.time_barrier.wait() 

class SlaveThread(Thread):
    """A short-lived worker thread that executes a single script."""
    def __init__(self, script, location, neighbours, device):
        Thread.__init__(self, name="Slave Thread of Device %d" % device.device_id)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """
        Executes the script logic: lock, gather data, run, and distribute results.
        """
        device = self.device
        script = self.script
        location = self.location
        neighbours = self.neighbours
        
        data = device.get_data(location)
        input_data = []
        this_lock = device.lock_location[location]

        if data is not None:
            input_data.append(data) 

        # Block Logic: Acquires a lock for the specific location to ensure that
        # data from neighbors is read and results are written atomically.
        with this_lock: 
            for neighbour in neighbours:
                temp = neighbour.get_data(location) 

                if temp is not None:
                    input_data.append(temp)

            if input_data != []: 
                result = script.run(input_data) 

                # Distributes the result to all neighbors and the parent device.
                for neighbour in neighbours:
                    neighbour.set_data(location, result) 

                device.set_data(location, result) 


class ReusableBarrierSem():
    """
    A reusable barrier implemented with semaphores, using a two-phase protocol.

    This ensures that threads from a new `wait()` cycle cannot interfere with
    threads from the previous cycle that are still being released.
    """

    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock()
        # The first semaphore blocks threads until all have arrived.
        self.threads_sem1 = Semaphore(0) 
        # The second semaphore ensures all threads have left phase 1 before resetting.
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        """Causes a thread to wait at the barrier."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase: Threads arrive and wait."""
        with self.counter_lock:
            self.count_threads1 -= 1
            # The last thread to arrive opens the gate for all waiting threads.
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire()
         
    def phase2(self):
        """Second phase: Ensures all threads have passed before resetting."""
        with self.counter_lock:
            self.count_threads2 -= 1
            # The last thread to pass phase 2 resets the barrier for the next use.
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            self.count_threads1 = self.num_threads
         
        self.threads_sem2.acquire()
