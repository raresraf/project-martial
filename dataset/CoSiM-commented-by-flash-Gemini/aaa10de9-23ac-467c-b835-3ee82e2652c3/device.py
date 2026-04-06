"""
This module implements a simulated distributed system, modeling a network of
interconnected 'devices' that collaboratively process sensor data. It features
a custom reusable barrier for inter-thread synchronization and a multi-threaded
architecture within each device for concurrent script execution.

The key components include:
- `ReusableBarrier`: A custom implementation of a barrier synchronization primitive
  that allows a fixed number of threads to wait for each other at a common point
  and then proceed. Designed for reusability across multiple synchronization cycles.
- `Device`: Represents an individual node in the distributed system. Each device
  manages its own sensor data, holds a queue of scripts to execute, and orchestrates
  its main processing thread (`DeviceThread`) along with multiple auxiliary worker
  threads (`ThreadAux`). It utilizes class-level static synchronization objects
  for global coordination.
- `DeviceThread`: The primary execution thread for each `Device`. It is responsible
  for discovering neighbors, managing timepoint progression, and distributing
  script execution responsibilities to its auxiliary threads.
- `ThreadAux`: Auxiliary worker threads spawned by each `Device`. These threads
  are dedicated to executing individual scripts or batches of scripts, including
  data collection from local sensors and neighbors, script execution, and updating
  sensor data across the network. They rely on shared locks for data consistency.

The system aims to simulate a complex cooperative data processing environment
where devices need to synchronize their operations and safely exchange data
to perform computational tasks.
"""


from threading import Event, Thread, Lock, Semaphore, Lock

class ReusableBarrier(object):
    """
    Implements a reusable barrier synchronization primitive for a fixed number of threads.

    This barrier allows a group of `num_threads` to halt execution at a specific
    point (`wait` method) until all threads in the group have arrived. Once all
    threads have reached the barrier, they are all released simultaneously.
    The barrier can then be reset and reused for subsequent synchronization points.

    The implementation uses a two-phase approach (via `phase1` and `phase2` methods)
    to ensure reusability without deadlock. Each phase involves:
    - A `counter_lock` to atomically decrement a thread counter.
    - A `Semaphore` to block arriving threads and release them when the counter reaches zero.

    Attributes:
        num_threads (int): The total number of threads expected to participate in the barrier.
        count_threads1 (int): Internal counter for the first phase of the barrier.
        count_threads2 (int): Internal counter for the second phase of the barrier.
        counter_lock (Lock): A lock to protect access to the thread counters during decrement/reset.
        threads_sem1 (Semaphore): Semaphore for the first phase of thread synchronization.
        threads_sem2 (Semaphore): Semaphore for the second phase of thread synchronization.
    """
    
    def __init__(self, num_threads):
        """
        Initializes a new `ReusableBarrier` instance.

        Sets up the internal state required for barrier synchronization,
        including thread counters, a lock for atomic updates, and two
        semaphores for managing thread blocking and release across two phases.

        Args:
            num_threads (int): The total number of threads that will
                                participate in this barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads


        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait until all other threads (`num_threads`)
        have also called this method.

        This method orchestrates the two-phase barrier synchronization. A thread
        must successfully pass through both `phase1` and `phase2` before being
        allowed to proceed. This design prevents a "last thread in" from potentially
        getting ahead of "first thread out" in a subsequent barrier use.
        """
        
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        Manages the first phase of the barrier synchronization.

        Threads decrement a shared counter, and the last thread to reach zero
        releases all waiting threads for this phase. All other threads block.
        """
        
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for reusability

        self.threads_sem1.acquire() # Acquire (block) until released by the last thread

    def phase2(self):
        """
        Manages the second phase of the barrier synchronization.

        Similar to `phase1`, threads decrement a shared counter, and the last
        thread to reach zero releases all waiting threads for this phase.
        All other threads block. This second phase is crucial for ensuring
        reusability without allowing threads to prematurely pass the barrier.
        """
        
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:


                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for reusability

        self.threads_sem2.acquire() # Acquire (block) until released by the last thread


class Device(object):
    """
    Represents an individual node (device) in the simulated distributed system.

    Each `Device` manages its own sensor data, holds scripts for execution, and
    orchestrates its processing threads. It plays a central role in coordinating
    data exchange and synchronized task execution with other devices and a
    central supervisor. This class utilizes both instance-level and class-level
    synchronization primitives for complex multi-threaded coordination.

    Class Attributes:
        bar1 (ReusableBarrier): A class-level `ReusableBarrier` used for global
                                synchronization across all `DeviceThread`s in the system.
        event1 (Event): A class-level `Event` used to signal a global state change,
                        often related to setup completion.
        locck (list): A class-level list of `Lock` objects, where each lock protects
                      access to a specific sensor `location` across all devices.

    Instance Attributes:
        timepoint_done (Event): An `Event` object used to signal when a timepoint's
                                script assignments have been fully processed for this device.
        device_id (int): A unique integer identifier for this device.
        sensor_data (dict): A dictionary holding the device's local sensor readings.
                            Keys are sensor locations/types, and values are the corresponding data.
        supervisor (object): A reference to the central `Supervisor` orchestrating the devices.
        devices (list): A list of all `Device` objects in the system (populated only for device_id 0).
        event (list): A list of `Event` objects (currently 11), likely used for internal
                      synchronization specific to `DeviceThread` or `ThreadAux`.
        nr_threads_device (int): The number of auxiliary worker threads (`ThreadAux`)
                                 to be spawned by this device (default 8).
        nr_thread_atribuire (int): An index used to distribute scripts among the
                                   `ThreadAux` instances in a round-robin fashion.
        bar_threads_device (ReusableBarrier): A `ReusableBarrier` for synchronizing
                                              the main `DeviceThread` with its `ThreadAux` workers.
        thread (DeviceThread): The main `DeviceThread` instance for this device.
        threads (list): A list of `ThreadAux` instances spawned by this device.
    """
    
    
    bar1 = ReusableBarrier(1) # Global barrier for all devices
    event1 = Event() # Global event for initial synchronization
    locck = [] # Global list of locks for sensor locations

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new `Device` instance.

        Sets up the device's unique identifier, its local sensor data, and a reference
        to the central `supervisor`. It also initializes internal state variables,
        creates synchronization primitives specific to this device, and starts
        its dedicated `DeviceThread` and a pool of `ThreadAux` worker threads.

        Args:
            device_id (int): A unique integer identifier for this device.
            sensor_data (dict): A dictionary representing the sensor readings
                                collected by this device. Keys are sensor locations
                                or types, and values are the corresponding data.
            supervisor (object): An object representing the central supervisor
                                 that orchestrates the devices.
        """
        
        
        self.timepoint_done = Event() # Signals script assignments for a timepoint are complete
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.devices = [] # Populated by `setup_devices` for master device

        
        # A list of events, possibly for inter-thread communication or signaling.
        self.event = []
        for _ in xrange(11):
            self.event.append(Event())

        
        self.nr_threads_device = 8 # Number of auxiliary worker threads
        
        self.nr_thread_atribuire = 0 # Index for round-robin script assignment to worker threads
        
        # Barrier for synchronizing the main DeviceThread with its ThreadAux workers
        self.bar_threads_device = ReusableBarrier(self.nr_threads_device+1)

        
        # Start the main DeviceThread for this device
        self.thread = DeviceThread(self)
        self.thread.start()

        
        # Spawn and start auxiliary worker threads
        self.threads = []
        for _ in xrange(self.nr_threads_device):
            self.threads.append(ThreadAux(self))
        for threadd in self.threads:
            threadd.start()

    def __str__(self):
        """
        Provides a human-readable string representation of the Device object.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the device with the list of all devices in the simulation
        and initializes global synchronization primitives if this is the master device.

        This method is critical for establishing the system's topology and setting
        up shared resources like the global barrier and location-specific locks.

        Args:
            devices (list): A list of all `Device` objects participating in the simulation.
        """
        
        self.devices = devices
        
        if self.device_id == 0: # Only the master device (ID 0) performs global setup
            # Initialize global location locks for up to 30 sensor locations.
            for _ in xrange(30):
                Device.locck.append(Lock())
            # Initialize the global barrier for all devices.
            Device.bar1 = ReusableBarrier(len(devices))
            
            # Signal that global setup is complete.
            Device.event1.set()

    def assign_script(self, script, location):
        """
        Assigns a script to an auxiliary worker thread for execution at a specific location.

        Scripts are assigned in a round-robin fashion to the device's `ThreadAux` workers.
        A `None` script acts as a signal to indicate the completion of script assignments
        for a timepoint, triggering the `timepoint_done` event.

        Args:
            script (object or None): The script object to be executed, or `None` for signaling.
            location (int): The sensor location associated with the script.
        """
        
        if script is not None:
            # Assign script to a worker thread's script_loc dictionary.
            self.threads[self.nr_thread_atribuire].script_loc[script] = location
            
            # Move to the next worker thread in a round-robin manner.
            self.nr_thread_atribuire = (self.nr_thread_atribuire+1)%
            self.nr_threads_device
        else:
            # Signal that script assignments for the current timepoint are complete.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location from the device's local store.

        Args:
            location (int): The identifier for the sensor location whose data is to be retrieved.

        Returns:
            Any: The sensor data associated with the given `location`, or `None` if
                 the `location` is not found in `self.sensor_data`.
        """
        
        return self.sensor_data[location] if location in 
        self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location in the device's local store.

        The update only occurs if the specified `location` already exists in
        `self.sensor_data`, preventing the creation of new sensor locations
        through this method.

        Args:
            location (int): The identifier for the sensor location whose data is to be updated.
            data (Any): The new data value to be stored for the given `location`.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates a graceful shutdown of the device by joining its main
        `DeviceThread` and all auxiliary `ThreadAux` worker threads.

        This ensures that all processing within the device completes cleanly
        before the device object is fully decommissioned.
        """
        
        self.thread.join()
        for threadd in self.threads:
            threadd.join()


class DeviceThread(Thread):
    """
    The main execution thread for a `Device` instance.

    This thread manages the overall operation of a device within the simulated
    distributed system. Its primary responsibilities include:
    - Discovering and updating its list of neighboring devices from the `supervisor`.
    - Waiting for and signaling `timepoint_done` events to coordinate script assignments.
    - Synchronizing with its `ThreadAux` workers and other `DeviceThread`s using barriers.
    - Handling termination signals from the `supervisor`.
    """
    

    def __init__(self, device):
        """
        Initializes a new `DeviceThread` instance.

        Sets up the thread with a descriptive name and establishes a strong
        reference to the `Device` object it is responsible for managing.

        Args:
            device (Device): The `Device` instance that this thread will serve.
                             This reference allows the thread to access and
                             manipulate the parent device's state and resources.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = None # Stores the current list of neighbors
        self.contor = 0 # Counter, likely for managing events or specific phases.

    def run(self):
        """
        The main execution loop for the `DeviceThread`.

        This method defines the continuous operation of a device within the
        simulated distributed system. It performs the following sequence of operations
        in a loop until a termination condition is met:

        1.  **Global Setup Wait**: Waits for `Device.event1` to be set, indicating
            that global device setup (like barrier initialization) is complete.
        2.  **Neighbor Discovery**: Fetches the latest list of neighbors from the
            `supervisor`. If `None` is returned, it signals a termination,
            breaks the loop, and ends the thread.
        3.  **Script Assignment Synchronization**: Waits for `self.device.timepoint_done`
            to be set, indicating that all scripts for the current timepoint have
            been assigned to the `ThreadAux` workers.
        4.  **Signal Workers**: Sets an event (`self.device.event[self.contor]`) to
            signal its auxiliary workers that new neighbors are available and scripts
            can be processed. The `contor` is incremented.
        5.  **Worker Barrier**: Waits at `self.device.bar_threads_device` to synchronize
            with all its `ThreadAux` workers, ensuring all workers have completed
            their tasks for the current timepoint.
        6.  **Global Barrier**: Waits at the global `Device.bar1` to synchronize with
            all other `DeviceThread`s in the system, ensuring that all devices complete
            their timepoint processing before proceeding to the next.
        """
        Device.event1.wait() # Wait for global setup to complete.

        while True:
            
            # Fetch neighbors from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()

            if self.neighbours is None:
                # If no neighbors, signal termination to workers and break.
                self.device.event[self.contor].set() # Signal to worker threads that neighbors are None (for termination)
                break

            
            # Wait for all scripts to be assigned for the current timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear() # Clear the event for the next timepoint.

            
            # Signal worker threads to start processing with the new neighbor list.
            self.device.event[self.contor].set()
            self.contor += 1 # Advance event counter, possibly for next event slot.

            
            # Synchronize with auxiliary worker threads.
            self.device.bar_threads_device.wait()

            
            # Global synchronization with all other DeviceThreads.
            Device.bar1.wait()