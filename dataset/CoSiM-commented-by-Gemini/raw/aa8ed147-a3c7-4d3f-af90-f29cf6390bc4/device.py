"""
A simulation framework for a network of communicating devices.

This script defines a system of Devices that can execute assigned 'scripts'
concurrently. The devices operate in synchronized time-steps, managed by a
reusable barrier. Each device runs in its own thread and can execute multiple
scripts, which may involve gathering data from neighboring devices, performing
a computation, and updating data on itself and its neighbors.

The main components are:
- ReusableBarrierSem: A synchronization primitive to ensure all devices
  complete a phase before starting the next.
- Device: Represents a node in the network, holding sensor data and managing
  script execution.
- DeviceThread: The main control loop for a Device, handling synchronization
  and script management.
- MyScriptThread: A thread dedicated to executing a single script on a device,
  including communication with neighbors.
"""
from threading import Event, Semaphore, Lock, Thread


class ReusableBarrierSem(object):
    """
    Implements a reusable barrier using two semaphores.

    This barrier forces a specified number of threads to wait until all of them
    have called the `wait()` method. It is "reusable" because it can be used
    multiple times. It employs a two-phase signaling protocol to prevent
    race conditions where fast threads could loop around and enter the barrier
    a second time before slow threads have left it from the first time.

    Attributes:
        num_threads (int): The number of threads that must wait at the barrier.
        count_threads1 (int): Counter for threads entering the first phase.
        count_threads2 (int): Counter for threads entering the second phase.
        counter_lock (Lock): A mutex to protect access to the counters.
        threads_sem1 (Semaphore): Semaphore for the first synchronization phase.
        threads_sem2 (Semaphore): Semaphore for the second synchronization phase.
    """
    def __init__(self, num_threads):
        """Initializes the reusable barrier for a given number of threads."""
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        
        self.counter_lock = Lock()
        
        # The first semaphore is initially locked. All threads will block on it.
        self.threads_sem1 = Semaphore(0)
        
        # The second semaphore is also locked, used for the second phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to wait at the barrier until all threads have arrived.
        """
        # The barrier consists of two phases to ensure reusability.
        self.phase1()
        self.phase2()

    def phase1(self):
        """First phase of the barrier synchronization."""
        with self.counter_lock:
            self.count_threads1 -= 1
            # The last thread to arrive at the barrier...
            if self.count_threads1 == 0:
                # ...releases all waiting threads by signaling the semaphore num_threads times.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset the counter for the next use of the barrier.
                self.count_threads1 = self.num_threads
        # All threads wait here until the last one releases them.
        self.threads_sem1.acquire()

    def phase2(self):
        """Second phase to ensure all threads have passed the first phase."""
        with self.counter_lock:
            self.count_threads2 -= 1
            # The last thread to leave the first phase...
            if self.count_threads2 == 0:
                # ...releases all threads for the next full wait cycle.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset the counter for the next use.
                self.count_threads2 = self.num_threads
        # This ensures no thread proceeds until all have completed phase1.
        self.threads_sem2.acquire()


class Device(object):
    """
    Represents a single device in the distributed system simulation.

    Each device runs its own thread, holds sensor data, and can execute
    scripts that may interact with its neighbors.

    Attributes:
        device_id (int): A unique identifier for the device.
        sensor_data (dict): A dictionary storing local sensor readings.
        supervisor (object): An external object that manages the simulation.
        script_received (Event): An event to signal when scripts are assigned.
        scripts (list): A list of (script, location) tuples to be executed.
        my_lock (Lock): A lock to protect access to this device's data.
        barrier (ReusableBarrierSem): The synchronization barrier.
        timepoint_done (Event): Signals that the device has finished its work
                                for the current timepoint.
        thread (DeviceThread): The main execution thread for this device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """Initializes the device and starts its main thread."""
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        
        self.my_lock = Lock()
        self.barrier = ReusableBarrierSem(0)
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """Returns a string representation of the device."""
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the synchronization barrier for a group of devices.

        This method is intended to be called once. Device 0 creates the
        barrier, and all other devices share it.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            # All devices share the barrier created by device 0.
            self.barrier = devices[0].barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device.

        If the script is None, it signals that no more scripts are coming for
        this timepoint.

        Args:
            script (object): The script object to run.
            location (any): The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A None script is a signal from the supervisor that all scripts
            # for this timepoint have been assigned.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (any): The key for the desired sensor data.

        Returns:
            The sensor data, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Updates sensor data for a specific location.

        This method is expected to be called with `my_lock` held.

        Args:
            location (any): The key for the sensor data to update.
            data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """Stops the device's thread and waits for it to terminate."""
        # This method assumes the main loop in DeviceThread will terminate.
        self.thread.join()


class MyScriptThread(Thread):
    """
    A thread that executes a single script on a device.

    This thread is responsible for gathering data from the device and its
    neighbors, running the script's logic, and then disseminating the
    results back to the device and its neighbors.

    Attributes:
        script (object): The script to execute, with a `run` method.
        location (any): The context/location for this script execution.
        device (Device): The parent device running this script.
        neighbours (list): A list of neighboring Device objects.
    """

    def __init__(self, script, location, device, neighbours):
        """Initializes the script thread."""
        Thread.__init__(self)
        self.script = script
        self.location = location
        self.device = device
        self.neighbours = neighbours

    def run(self):
        """The main execution logic for the script thread."""
        script_data = []

        # Block Logic: Gather data from neighbors.
        # Pre-condition: Neighbors are available.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Gather data from the local device.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Perform the computation defined by the script.
            result = self.script.run(script_data)

            # Block Logic: Disseminate the result to all neighbors.
            # The lock on each device is acquired before setting data.
            # Invariant: Data consistency is maintained through locking.
            for device in self.neighbours:
                device.my_lock.acquire()
                device.set_data(self.location, result)
                device.my_lock.release()

            # Disseminate the result to the local device.
            self.device.my_lock.acquire()
            self.device.set_data(self.location, result)
            self.device.my_lock.release()


class DeviceThread(Thread):
    """
    The main control thread for a single Device.

    This thread orchestrates the device's participation in the simulation's
    time-stepped execution cycle.

    Attributes:
        device (Device): The device this thread belongs to.
    """

    def __init__(self, device):
        """Initializes the device thread."""
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main simulation loop for the device.

        The cycle is as follows:
        1. Get neighbor information from the supervisor.
        2. Wait at a barrier for all devices to be ready (synchronization).
        3. Wait for scripts to be assigned by the supervisor.
        4. Spawn and run threads for each assigned script.
        5. Wait for another signal indicating the timepoint is complete.
        6. Wait at the barrier again to ensure all devices finished the step.
        7. Clear state for the next iteration.
        """
        while True:
            # Get the list of neighbors for the current simulation step.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Supervisor signals termination by returning None.
                break
            
            # --- Synchronization Point 1 ---
            # All devices wait here before starting script execution.
            self.device.barrier.wait()

            # Wait until the supervisor signals that all scripts are assigned.
            self.device.script_received.wait()
            
            script_threads = []
            # Create a thread for each assigned script.
            for (script, location) in self.device.scripts:
                script_threads.append(MyScriptThread(script,
                    location, self.device, neighbours))
            
            # Start and join all script threads, effectively parallelizing
            # script execution within a single device's time step.
            for thread in script_threads:
                thread.start()
            for thread in script_threads:
                thread.join()
            
            # Wait for the supervisor's signal that the timepoint is done.
            self.device.timepoint_done.wait()
            
            # --- Synchronization Point 2 ---
            # All devices wait here to ensure everyone has finished their
            # work for this timepoint before proceeding to the next.
            self.device.barrier.wait()
            
            # Reset for the next time step.
            self.device.script_received.clear()
