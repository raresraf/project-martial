"""
This module provides a framework for simulating a network of interconnected devices.

It defines a `Device` class that operates as a node in the network, processing
computational scripts based on its own and its neighbors' sensor data. The
simulation proceeds in discrete timepoints. Synchronization between devices is
managed by a custom `ReusableBarrier` and a series of location-based locks,
ensuring data consistency. Each script is executed in its own thread, with a
semaphore limiting the total number of concurrent script executions per device.
"""

from threading import Thread, Lock, Event, Condition, Semaphore

class ReusableBarrier():
    """
    A simple reusable barrier implementation using a Condition variable.

    This barrier allows a set number of threads to wait for each other to reach
    a certain point of execution before proceeding. After all threads have
    arrived, the barrier resets itself for the next use.

    Attributes:
        num_threads (int): The number of threads that must wait at the barrier.
    """
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The number of threads to wait for.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """

        Causes a thread to wait at the barrier.

        When the required number of threads have called wait(), all of them are
        notified and released to continue execution.
        """
        self.cond.acquire()
        self.count_threads -= 1
        # Block Logic: If this is the last thread to arrive, notify all waiting
        # threads and reset the barrier for the next cycle.
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            self.cond.wait()
        self.cond.release()

class Device(object):
    """
    Represents a device in the simulated network.

    Each device has an orchestrator thread (`DeviceThread`) that manages its
    lifecycle. For each assigned script, it spawns a temporary `NewThreadScript`
    to execute the computation. It relies on shared synchronization objects
    (barrier, locks) distributed by a designated master device (ID 0).

    Attributes:
        device_id (int): The unique identifier for the device.
        sensor_data (dict): Local data store for the device.
        supervisor: The central entity managing the network topology.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): The unique ID for this device.
            sensor_data (dict): The initial sensor data for this device.
            supervisor: The central supervisor for the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Event to signal that all scripts for the current timepoint are assigned.
        self.timepoint_done = Event()
        # Event to signal that the initial device setup is complete.
        self.setup_event = Event()

        # A list of locks, one for each data location, shared across all devices.
        self.lock_location = []
        self.lock_n = Lock()
        self.barrier = None

        self.thread_script = []
        self.num_thread = 0
        # Semaphore to limit the number of concurrent script threads to 8.
        self.sem = Semaphore(value=8)

        # The main orchestrator thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared synchronization objects for the entire network.

        This method must be called on one designated "master" device (ID 0).
        It creates a shared barrier and a set of location-based locks and
        distributes them to all other devices in the network.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        
        # Block Logic: Only the device with ID 0 acts as the coordinator for setup.
        if self.device_id == 0:
            barrier = ReusableBarrier(len(devices))
            # Assumption: There are at most 25 distinct data locations.
            for _ in xrange(25):
                self.lock_location.append(Lock())

            # Injects the shared barrier and locks into every device.
            for dev in devices:
                dev.barrier = barrier
                dev.lock_location = self.lock_location
                dev.setup_event.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device for the current timepoint.

        Args:
            script: The script object to execute.
            location (int): The data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script signifies the end of script assignment for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Args:
            location (int): The location index.
        
        Returns:
            The data for the location, or None if not found.
        """
        return self.sensor_data[location] if location in 
            self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates sensor data at a given location.

        Args:
            location (int): The location index.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Shuts down the main device thread."""
        self.thread.join()

    def shutdown_script(self):
        """
        Cleans up all script-execution threads spawned during a timepoint.
        """
        for i in xrange(self.num_thread):
            self.thread_script[i].join()

        # Clears the list of thread objects.
        for i in xrange(self.num_thread):
            del self.thread_script[-1]

        self.num_thread = 0

class NewThreadScript(Thread):
    """
    A short-lived thread to execute a single script for a single location.
    
    It handles acquiring the location-specific lock, gathering data, running
    the script, and distributing the results.
    """
    def __init__(self, parent, neighbours, location, script):
        Thread.__init__(self)
        self.neighbours = neighbours
        self.parent = parent
        self.location = location
        self.script = script

    def run(self):
        """
        The main execution logic for the script thread.
        
        Acquires a location-specific lock to ensure exclusive access to the
        data at that location during computation.
        """
        # Block Logic: The 'with' statement ensures that the lock for this
        # specific location is acquired before processing and released after,
        # preventing race conditions.
        with self.parent.lock_location[self.location]:
            script_data = []
            
            # Gathers data from all neighboring devices.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Gathers data from the parent device.
            data = self.parent.get_data(self.location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                # Executes the script with the aggregated data.
                result = self.script.run(script_data)

                
                # Distributes the result back to neighbors and the parent.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                self.parent.set_data(self.location, result)
        
        # Releases the semaphore, allowing another script thread to be spawned.
        self.parent.sem.release()

class DeviceThread(Thread):
    """The main orchestrator thread for a Device."""

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """The main control loop, executing timepoint by timepoint."""
        
        # Block Logic: Waits until the master device has completed initial setup.
        self.device.setup_event.wait()

        while True:
            
            # Fetches the list of neighbors for the current timepoint.
            with self.device.lock_n:
                neighbours = self.device.supervisor.get_neighbours()
                # A None value for neighbors is the shutdown signal.
                if neighbours is None:
                    break

            
            # Waits until all scripts for the timepoint have been assigned.
            self.device.timepoint_done.wait()

            
            # Block Logic: Spawns a new thread for each script, respecting the semaphore limit.
            for (script, location) in self.device.scripts:
                # Acquires the semaphore, blocking if the concurrent thread limit is reached.
                self.device.sem.acquire()
                self.device.thread_script.append(NewThreadScript 
                    (self.device, neighbours, location, script))

                self.device.num_thread = self.device.num_thread + 1
                self.device.thread_script[-1].start()



            
            # === SYNC BARRIER 1 ===
            # Waits for all devices to finish spawning their script threads.
            self.device.barrier.wait()
            
            # Joins all script threads to ensure they have completed execution.
            self.device.shutdown_script()
            
            # Resets the event for the next timepoint.
            self.device.timepoint_done.clear()
            
            # === SYNC BARRIER 2 ===
            # Waits for all devices to complete their cleanup before starting the next cycle.
            self.device.barrier.wait()
