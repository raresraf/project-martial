"""
Models a distributed system of communicating devices for a sensor network simulation.

This module defines the core components for a simulation of a network of devices
that process sensor data collaboratively. It includes a custom reusable barrier for
thread synchronization and a multi-threaded device model where each device runs
scripts on data gathered from itself and its neighbors.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    A reusable barrier for synchronizing a fixed number of threads.

    This implementation uses a two-phase protocol to ensure that threads wait at the
    barrier until all participating threads have arrived. Once all have arrived, they
    are released and the barrier resets for the next use. This prevents threads from
    one iteration from proceeding before all threads from the *previous* iteration
    have been released.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads

        # Lock to protect access to the internal thread counters.
        self.counter_lock = Lock()
        # Semaphores to block and release threads for each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """Causes a thread to wait at the barrier until all threads have arrived."""
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the two-phase barrier."""
        with self.counter_lock:
            self.count_threads1 -= 1
            # If this is the last thread to arrive...
            if self.count_threads1 == 0:
                # ...release all waiting threads for phase 1...
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # ...and reset the counter for the next use of this phase.
                self.count_threads1 = self.num_threads
        
        # All threads wait here until the last thread releases the semaphore.
        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase of the two-phase barrier."""
        with self.counter_lock:
            self.count_threads2 -= 1
            # If this is the last thread to arrive...
            if self.count_threads2 == 0:
                # ...release all waiting threads for phase 2...
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # ...and reset the counter for the next use of this phase.
                self.count_threads2 = self.num_threads
        
        # All threads wait here, ensuring no thread from this iteration
        # wraps around to phase1 before all have completed phase2.
        self.threads_sem2.acquire()


class Device(object):
    """
    Represents a single device in the simulated network.

    Each device manages its own sensor data and executes scripts based on data from
    itself and its neighbors. It operates on a multi-threaded model, with one main
    `DeviceThread` for control and multiple `ThreadAux` workers for computation.

    Class Attributes:
        bar1 (ReusableBarrier): A global barrier to synchronize all devices.
        event1 (Event): A global event to signal the start of the simulation.
        locck (list): A list of locks, one for each data location, to ensure
                      atomic updates across the network.
    """
    bar1 = ReusableBarrier(1)
    event1 = Event()
    locck = []

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device and starts its threads.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary of sensor data, mapping location to value.
            supervisor (object): The supervisor object that manages the network topology.
        """
        # Event to signal the completion of a timepoint's script assignments.
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = []

        # A list of events to synchronize the main thread with worker threads.
        self.event = [Event() for _ in xrange(11)]

        self.nr_threads_device = 8
        # Counter for round-robin script assignment to worker threads.
        self.nr_thread_atribuire = 0
        
        # A barrier to synchronize this device's main thread and its workers.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device + 1)

        # The main control thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

        # The pool of worker threads for this device.
        self.threads = [ThreadAux(self) for _ in xrange(self.nr_threads_device)]
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        """String representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs global setup for the simulation. Called once.

        Device 0 is responsible for initializing shared resources like the global
        locks and the main barrier based on the total number of devices.
        
        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        self.devices = devices
        
        # Device 0 acts as the master for initialization.
        if self.device_id == 0:
            # Initialize a lock for each of the 30 possible data locations.
            Device.locck = [Lock() for _ in xrange(30)]
            # Initialize the global barrier for all devices.
            Device.bar1 = ReusableBarrier(len(devices))
            # Signal that setup is complete and the simulation can start.
            Device.event1.set()

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by a worker thread in a round-robin fashion.

        Args:
            script (object): The script object to execute. Must have a `run` method.
            location (int): The data location the script operates on.
        """
        if script is not None:
            # Assign the script and location to the next available worker thread.
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            self.nr_thread_atribuire = (self.nr_thread_atribuire + 1) % self.nr_threads_device
        else:
            # A `None` script indicates the end of script assignments for this timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data from a specific location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data at a specific location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Joins all threads to ensure a clean shutdown."""
        self.thread.join()
        for threadd in self.threads:
            threadd.join()


class DeviceThread(Thread):
    """The main control thread for a Device."""
    def __init__(self, device):
        """Initializes the device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None
        # Counter to advance through the per-timepoint events.
        self.contor = 0

    def run(self):
        """
        The main execution loop for the device's control logic.

        This loop orchestrates the device's participation in each timepoint of
        the simulation. It retrieves neighbors, waits for scripts to be assigned,
        signals workers to start, and synchronizes with both its own workers and
        all other devices in the network.
        """
        # Wait for the initial global setup to complete.
        Device.event1.wait()

        while True:
            # At each timepoint, get the current set of neighbors from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()

            # A `None` neighbor list is the signal to shut down.
            if self.neighbours is None:
                # Signal workers to also shut down.
                self.device.event[self.contor].set()
                break

            # Wait until the supervisor has finished assigning scripts for this timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Signal this device's worker threads that they can start processing.
            self.device.event[self.contor].set()
            self.contor += 1
            
            # Wait at the local barrier until all of this device's workers have finished.
            self.device.bar_threads_device.wait()

            # Wait at the global barrier until all devices in the network are finished.
            Device.bar1.wait()

class ThreadAux(Thread):
    """A worker thread that executes computational scripts for a Device."""
    def __init__(self, device):
        """Initializes the worker thread."""
        Thread.__init__(self)
        self.device = device
        # A dictionary mapping assigned scripts to their target locations.
        self.script_loc = {}
        # Counter to advance through the per-timepoint events.
        self.contor = 0

    def run(self):
        """
        The main execution loop for the worker thread.
        
        The thread waits for a signal from its parent `DeviceThread`, then executes
        all its assigned scripts. For each script, it gathers data from its own device
        and its neighbors, runs the script, and distributes the results back.
        """
        while True:
            # Wait for the signal to start processing for the current timepoint.
            self.device.event[self.contor].wait()
            self.contor += 1

            # Check for the shutdown signal.
            neigh = self.device.thread.neighbours
            if neigh is None:
                break

            # Execute all scripts assigned for this timepoint.
            for script, location in self.script_loc.items():
                # Acquire the global lock for this location to ensure atomic data access.
                Device.locck[location].acquire()
                script_data = []

                # Gather data from all neighbors at the specified location.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Gather data from the local device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Run the script and update data if any was collected.
                if script_data:
                    result = script.run(script_data)
                    # Propagate the result to all neighbors and the local device.
                    for device in neigh:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the lock for the location.
                Device.locck[location].release()

            # Wait at the local barrier to signal completion to the main DeviceThread.
            self.device.bar_threads_device.wait()
