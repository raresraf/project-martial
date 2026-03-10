"""
This module implements a simulation framework for distributed devices,
focusing on concurrent execution of scripts and synchronized data processing.
It includes a reusable barrier (`ReusableBarrierSem`) for thread synchronization,
a `Device` class for device simulation, a `DeviceThread` for main device logic,
and a `SlaveThread` for executing individual scripts.
"""


from threading import *


class Device(object):
    """
    Represents a simulated device within a distributed system.

    Each device manages its sensor data, interacts with a central supervisor,
    and dispatches scripts for execution. It launches a dedicated `DeviceThread`
    to handle control flow and participates in global synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor readings
                                (e.g., {location_id: data_value}).
            supervisor (object): An object representing the central supervisor,
                                 used for coordination (e.g., getting neighbors).
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a script has been received (though its usage with `assign_script` is specific).
        self.script_received = Event()
        # List to temporarily store scripts assigned to this device before dispatching.
        self.scripts = []
        # Event to signal when the current timepoint's script assignments are done.
        self.timepoint_done = Event()
        # The main thread for the device, responsible for supervisor interaction and dispatching scripts.
        self.thread = DeviceThread(self)
        self.thread.start()

        # Lock to protect `sensor_data` during updates in `set_data`.
        self.lock_data = Lock()
        # List of locks, one for each data location, to protect specific sensor data points.
        self.lock_location = []
        # Reference to the global barrier used for synchronizing all devices.
        self.time_barrier = None

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up global synchronization primitives (barrier and data locks)
        and distributes them among all devices.

        This method is typically called once during the initialization phase of the simulation.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Block Logic: Only device with ID 0 acts as the coordinator for setup.
        if self.device_id == 0:
            # Functional Utility: Creates a global barrier that all devices will use for synchronization.
            self.time_barrier = ReusableBarrierSem(len(devices))

            # Distributes the same barrier instance to all other devices.
            for device in devices:
                device.time_barrier = self.time_barrier

            # Block Logic: Determines the maximum location ID to create a sufficient number of location-specific locks.
            loc_num = 0
            for device in devices:
                for location in device.sensor_data:
                    loc_num = max(loc_num, location)
            # Initializes a list of `Lock` objects, one for each possible data location ID.
            for i in range(loc_num + 1):
                self.lock_location.append(Lock())

            # Distributes the same list of location-specific locks to all devices.
            for device in devices:
                device.lock_location = self.lock_location

    def assign_script(self, script, location):
        """
        Assigns a script for execution at a specific data location.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the sensor data location the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
            # Functional Utility: Sets the `script_received` event.
            # Note: This event is immediately cleared within the same function
            # in the original code, suggesting a specific, possibly single-shot, signaling pattern.
            self.script_received.set()
        else:
            # If script is None, it signifies the end of script assignments for the current timepoint.
            # Functional Utility: Signals that all scripts for the current timepoint have been assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a given location.

        Note: This method does NOT acquire a lock for the data location.
        Locking for data access is expected to be handled by the caller (`SlaveThread`).

        Args:
            location (int): The identifier for the sensor data location.

        Returns:
            Any: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates sensor data for a given location, protected by a global data lock.

        Args:
            location (int): The identifier for the sensor data location.
            data (Any): The new sensor data to set.
        """
        # Functional Utility: Acquires a global lock (`self.lock_data`) to protect
        # the entire `sensor_data` dictionary during modification, ensuring atomicity.
        with self.lock_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        Performs a graceful shutdown of the main device thread.
        """
        # Waits for the main device thread (`DeviceThread`) to complete its execution.
        self.thread.join()


class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for managing its interaction
    with the supervisor and coordinating the execution of scripts by
    launching individual `SlaveThread` instances for each script.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The `Device` instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        Main execution loop for the DeviceThread.

        Architectural Intent: Continuously fetches neighbor information from the supervisor.
        It orchestrates the execution of assigned scripts by launching a new `SlaveThread`
        for each, waits for them to complete, and then participates in a global barrier
        for timepoint synchronization.
        """
        while True:
            slaves = [] # List to keep track of active SlaveThread instances.
            
            # Block Logic: Fetches the current list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Termination Condition: If no neighbors are returned (None), it signifies
            # that the simulation is ending for this device, and the loop breaks.
            if neighbours is None:
                break

            # Block Logic: Waits until the current timepoint's script assignments are done.
            self.device.timepoint_done.wait()
            # Functional Utility: Clears the event to prepare for the next timepoint.
            self.device.timepoint_done.clear()

            
            # Block Logic: For each assigned script, a new SlaveThread is created and started.
            # This allows scripts to potentially run concurrently.
            for (script, location) in self.device.scripts:
                slave = SlaveThread(script, location, neighbours, self.device)
                slaves.append(slave)
                slave.start()

            # Block Logic: Waits for all launched SlaveThread instances to complete their execution.
            # Inline: `slaves.pop().join()` waits for each slave thread, processing them from last to first.
            for i in range(len(slaves)):
                slaves.pop().join()

            # Functional Utility: Clears the list of scripts after they've been processed.
            self.device.scripts = []

            # Block Logic: Waits at the global barrier for all devices to complete their
            # current timepoint processing before proceeding to the next simulation step.
            self.device.time_barrier.wait()

class SlaveThread(Thread):
    """
    A worker thread responsible for executing a single script on a specific data location.

    This thread collects data from its associated device and its neighbors,
    executes the provided script, and updates the relevant sensor data,
    ensuring data consistency via location-specific locks.
    """
    def __init__(self, script, location, neighbours, device):
        """
        Initializes a SlaveThread.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the sensor data location the script operates on.
            neighbours (list): A list of neighboring Device objects.
            device (Device): The `Device` instance this thread is associated with.
        """
        Thread.__init__(self, name="Slave Thread of Device %d" % device.device_id)
        self.script = script
        self.location = location
        self.neighbours = neighbours
        self.device = device

    def run(self):
        """
        Main execution logic for the SlaveThread.

        Architectural Intent: Collects relevant sensor data, acquires a location-specific lock,
        executes the assigned script, and updates data on both the local device and its neighbors,
        all while ensuring thread safety for the specific data location.
        """
        
        device = self.device
        script = self.script
        location = self.location
        neighbours = self.neighbours
        
        # Block Logic: Retrieves data from the local device for the specified location.
        data = device.get_data(location)
        input_data = [] # List to store all data relevant to the script.
        # Functional Utility: Retrieves the location-specific lock for the current data point.
        this_lock = device.lock_location[location]

        if data is not None:
            input_data.append(data) # Adds local device data if available.

        # Functional Utility: Acquires a lock for the specific data location, ensuring
        # exclusive access to this data point across devices during script processing.
        with this_lock:
            # Block Logic: Collects data from neighboring devices for the same location.
            for neighbour in neighbours:
                temp = neighbour.get_data(location) # Note: neighbour.get_data does not acquire a lock itself.

                if temp is not None:
                    input_data.append(temp)

            # Block Logic: If any data was collected, executes the script and updates data.
            if input_data != []:
                # Functional Utility: Executes the assigned script with the collected data.
                result = script.run(input_data)

                # Updates data on neighboring devices.
                for neighbour in neighbours:
                    neighbour.set_data(location, result) # Note: neighbour.set_data acquires device.lock_data.

                # Updates data on its own device.
                device.set_data(location, result) # Note: device.set_data acquires device.lock_data.


class ReusableBarrierSem():
    """
    A reusable barrier synchronization primitive implemented using semaphores.

    This barrier allows a fixed number of threads (`num_threads`) to wait for
    each other to reach a common point before any can proceed. It uses a
    double-barrier approach to ensure reusability for subsequent synchronization points.
    """
    
    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrierSem.

        Args:
            num_threads (int): The total number of threads that must arrive
                               at the barrier before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for the first phase of the barrier.


        self.count_threads2 = self.num_threads # Counter for the second phase of the barrier.
        
        self.counter_lock = Lock() # Lock to protect the counters during updates.
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase, initialized to block all threads.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase, initialized to block all threads.

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        `num_threads` threads have also called `wait()`.
        """
        self.phase1() # Executes the first synchronization phase.
        self.phase2() # Executes the second synchronization phase, enabling reusability.

    def phase1(self):
        """
        Manages the first phase of the double-barrier synchronization.

        Block Logic: Decrements a shared counter. When it reaches zero, all threads
                     have arrived at the barrier for this phase. It then releases
                     all waiting threads via the first semaphore and resets the
                     counter for the second phase.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release() # Releases all threads blocked on `threads_sem1`.
            # Note: The reset for `count_threads2` here seems unusual for a standard double barrier.
            # In a typical double barrier, `count_threads1` would be reset in phase2, and vice versa.
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire() # Blocks the current thread until it's released by the semaphore.
         
    def phase2(self):
        """
        Manages the second phase of the double-barrier synchronization.

        Block Logic: Decrements a shared counter. When it reaches zero, all threads
                     have passed through the first phase. It then releases all
                     waiting threads via the second semaphore and resets the
                     counter for the first phase, making the barrier reusable.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release() # Releases all threads blocked on `threads_sem2`.
            # Note: The reset for `count_threads1` here seems unusual for a standard double barrier.
            # In a typical double barrier, `count_threads2` would be reset in phase1, and vice versa.
            self.count_threads1 = self.num_threads
         
        self.threads_sem2.acquire() # Blocks the current thread until it's released by the semaphore.
