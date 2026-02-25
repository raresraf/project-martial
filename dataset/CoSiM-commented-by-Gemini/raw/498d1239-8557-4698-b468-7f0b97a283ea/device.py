"""
Models a network of interconnected devices that execute data-parallel computations
in discrete, synchronized time steps. This system is designed for simulating
distributed algorithms, such as those found in sensor networks or grid computing,
where nodes (devices) perform computations based on their own state and the state
of their immediate neighbors.

The simulation uses a multi-level barrier synchronization strategy to ensure that
all devices and their internal worker threads complete a time step before any
proceed to the next, guaranteeing data consistency across simulation ticks.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    A reusable, two-phase barrier for synchronizing a fixed number of threads.

    This barrier implementation ensures that all participating threads reach the
    barrier before any of them are allowed to proceed. It uses two separate
    semaphores to manage two distinct synchronization phases, making it safe
    for threads to pass through it multiple times in a loop.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a specified number of threads.

        Args:
            num_threads (int): The number of threads that must wait at the barrier.
        """
        self.num_threads = num_threads
        # Counters for threads arriving at each phase of the barrier.
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads

        # Lock to protect access to the counters.
        self.counter_lock = Lock()
        # Semaphores to block threads during each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have arrived.
        The synchronization process is split into two distinct phases.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """Manages the first synchronization phase."""
        with self.counter_lock:
            self.count_threads1 -= 1
            # The last thread to arrive is responsible for releasing all waiting threads.
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset counter for the next use of the barrier.
                self.count_threads1 = self.num_threads

        self.threads_sem1.acquire()

    def phase2(self):
        """Manages the second synchronization phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            # The last thread to arrive releases all threads for the second phase.
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset counter for the next use.
                self.count_threads2 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a single node in the distributed simulation network.

    Each device manages its own sensor data, a pool of worker threads for
    computation, and a main control thread. It communicates with neighboring
    devices as defined by a central supervisor.
    """

    # Class-level variables shared across all Device instances.
    bar1 = ReusableBarrier(1) # A global barrier for synchronizing all devices.
    event1 = Event()            # A global event to signal simulation start.
    locck = []                  # A list of locks for location-based data access.

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary holding the device's initial sensor data.
            supervisor (Supervisor): A central object that manages the network topology.
        """
        # Event to signal that script assignment for a timepoint is complete.
        self.timepoint_done = Event()
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = []

        # A series of events for intra-device, inter-thread synchronization across timepoints.
        self.event = []
        for _ in xrange(11):
            self.event.append(Event())

        # Setup a pool of worker threads for parallel script execution.
        self.nr_threads_device = 8
        self.nr_thread_atribuire = 0 # Used for round-robin script assignment.
        
        # A barrier to synchronize the main device thread and its worker threads.
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device + 1)

        # Start the main control thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Start the pool of auxiliary worker threads.
        self.threads = []
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs one-time setup for shared, simulation-wide resources.
        This method should be called on one device after all devices are created.
        """
        self.devices = devices
        
        # The device with ID 0 is responsible for initializing global resources.
        if self.device_id == 0:
            # Initialize a global array of locks for fine-grained data synchronization.
            for _ in xrange(30):
                Device.locck.append(Lock())
            # Initialize the global barrier for all devices in the simulation.
            Device.bar1 = ReusableBarrier(len(devices))
            
            # Signal all devices to begin their main loops.
            Device.event1.set()

    def assign_script(self, script, location):
        """
        Assigns a computational script to a worker thread using a round-robin strategy.

        Args:
            script (Script): The script object to be executed.
            location (int): The data location the script will operate on.
        """
        if script is not None:
            # Assign script to the next available worker thread.
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            self.nr_thread_atribuire = (self.nr_thread_atribuire + 1) % self.nr_threads_device
        else:
            # A None script signals the end of assignments for the current timepoint.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        return self.sensor_data.get(location)

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Terminates all threads associated with this device."""
        self.thread.join()
        for threadd in self.threads:
            threadd.join()

class DeviceThread(Thread):
    """The main control thread for a single Device."""

    def __init__(self, device):
        """
        Initializes the main thread for a device.

        Args:
            device (Device): The parent device object.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None
        self.contor = 0 # Timepoint counter.

    def run(self):
        """
        The main control loop for the device, executed for each simulation timepoint.
        """
        # Wait for the global start signal.
        Device.event1.wait()

        while True:
            # At the beginning of each timepoint, get the current network topology.
            self.neighbours = self.device.supervisor.get_neighbours()

            # A None value for neighbours signals the end of the simulation.
            if self.neighbours is None:
                # Wake up worker threads so they can terminate gracefully.
                self.device.event[self.contor].set()
                break

            # Wait until all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            # Signal this device's worker threads to begin execution for this timepoint.
            self.device.event[self.contor].set()
            self.contor += 1

            # Wait for all of this device's worker threads to finish their computations.
            self.device.bar_threads_device.wait()

            # Wait for all other devices in the simulation to finish the current timepoint.
            Device.bar1.wait()

class ThreadAux(Thread):
    """A worker thread that executes computational scripts for a Device."""
    
    def __init__(self, device):
        """
        Initializes the auxiliary worker thread.

        Args:
            device (Device): The parent device object.
        """
        Thread.__init__(self)
        self.device = device
        self.script_loc = {} # Dictionary to store assigned scripts and their locations.
        self.contor = 0 # Timepoint counter.

    def run(self):
        """
        The main loop for a worker thread, executed for each timepoint.
        """
        while True:
            # Wait for the signal from the main DeviceThread to start the timepoint.
            self.device.event[self.contor].wait()
            self.contor += 1

            # Check for the simulation termination signal.
            neigh = self.device.thread.neighbours
            if neigh is None:
                break

            # Execute all assigned scripts for this timepoint.
            for script, location in self.script_loc.items():
                # Acquire a location-specific lock to ensure data consistency
                # across all devices operating on this data location.
                Device.locck[location].acquire()
                
                script_data = []
                # Gather data from all neighboring devices for the specified location.
                for device in neigh:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Include the device's own data.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Run the script and broadcast the result back to all participants.
                if script_data:
                    result = script.run(script_data)
                    for device in neigh:
                        device.set_data(location, result)
                    self.device.set_data(location, result)

                # Release the lock for the location.
                Device.locck[location].release()

            # Synchronize with the main DeviceThread and other worker threads on the same device.
            self.device.bar_threads_device.wait()
