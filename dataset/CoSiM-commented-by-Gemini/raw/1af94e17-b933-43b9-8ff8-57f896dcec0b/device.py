"""
This module implements a simulation of a distributed device network.

It defines a system of concurrent 'Device' objects that execute computational
scripts on sensor data, simulating a sensor network or a similar distributed system.
The simulation proceeds in discrete, synchronized time steps, managed by a
two-phase reusable barrier and a set of location-based locks.
"""

from threading import Thread, Semaphore, Event, Lock

class ReusableBarrierSem(object):
    """
    A reusable, two-phase barrier implemented using semaphores.

    This barrier allows a fixed number of threads to synchronize at a point in
    their execution. It is "reusable" because threads can wait on it multiple
    times. The implementation uses two separate semaphores to manage two distinct
    synchronization phases, preventing threads from one cycle from interfering
    with threads in the next.
    """
    def __init__(self, num_threads):
        """
        Initializes the barrier for a given number of threads.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                               before any of them can proceed.
        """
        self.num_threads = num_threads
        # Counter for threads arriving at the first phase.
        self.count_threads1 = self.num_threads
        # Counter for threads arriving at the second phase.
        self.count_threads2 = self.num_threads

        # Lock to protect access to the shared counters.
        self.counter_lock = Lock()
        # Semaphore for the first synchronization phase.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for the second synchronization phase.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes a thread to block until all threads have called this method.

        This is the main entry point for the barrier and orchestrates the two-
        phase synchronization protocol.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        Handles the first synchronization phase.

        A thread entering this phase decrements a counter. The last thread to
        arrive (when the counter hits zero) releases enough permits for all
        waiting threads to pass the first semaphore and resets the counter
        for the barrier's next use.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # The last thread releases all other waiting threads.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset the counter for the next synchronization event.
                self.count_threads1 = self.num_threads
        # All threads will block here until the last thread releases them.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        Handles the second synchronization phase.

        This phase ensures that no thread can start a new 'wait' cycle before
        all threads have left the first phase, preventing race conditions.
        The logic is identical to phase1 but uses a separate set of counters
        and semaphores.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads
        self.threads_sem2.acquire()

class Device(object):
    """
    Represents a single device in the distributed network simulation.

    Each device has an ID, local sensor data, and a main thread that manages
    its lifecycle and script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary representing the device's local data,
                                mapping locations to values.
            supervisor (Supervisor): A reference to the central supervisor object
                                     that manages the simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to the device.
        self.script_received = Event()
        self.scripts = []
        # Event to signal that all scripts for the current timepoint are assigned.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # Shared barrier, initialized by device 0.
        self.barrier = None
        # Shared locks for data locations, initialized by device 0.
        self.locks = []
        
        # Determine the maximum lock index needed based on sensor data locations.
        self.nrlocks = max(sensor_data) if sensor_data else -1

    def __str__(self):
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs centralized setup for all devices in the simulation.

        This method should only be called by one "master" device (device_id == 0).
        It creates and distributes the shared ReusableBarrierSem and a list of
        shared Locks to all devices.

        Args:
            devices (list): A list of all Device objects in the simulation.
        """
        # Pre-condition: This block should only be executed by a single master device.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            # Distribute the shared barrier instance to all devices.
            for _, device in enumerate(devices):
                device.barrier = self.barrier
        
        # Pre-condition: This block should only be executed by a single master device.
        if self.device_id == 0:
            listmaxim = []
            for _, device in enumerate(devices):
                if device.nrlocks > -1:
                    listmaxim.append(device.nrlocks)
            
            number = max(listmaxim) if listmaxim else -1
            
            # Create a shared lock for each potential data location.
            for _ in range(number + 1):
                self.locks.append(Lock())
            
            # Distribute the list of shared locks to all devices.
            for _, device in enumerate(devices):
                device.locks = self.locks

    def assign_script(self, script, location):
        """
        Assigns a computational script to be executed at a specific location.

        Args:
            script (Script): The script object to be executed. If None, it signals
                             the end of script assignments for the current timepoint.
            location (int): The data location the script will operate on.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script indicates that the work for this timepoint is defined.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves data from a specific sensor location on this device.

        Args:
            location (int): The data location to read from.

        Returns:
            The sensor data at the given location, or None if the location
            is not present on this device.
        """
        if location in self.sensor_data:
            data = self.sensor_data[location]
        else: data = None
        return data

    def set_data(self, location, data):
        """
        Updates data at a specific sensor location on this device.

        Args:
            location (int): The data location to write to.
            data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Gracefully shuts down the device's main thread.
        """
        self.thread.join()

class MiniDeviceThread(Thread):
    """
    A short-lived thread designed to execute a single script.

    This thread encapsulates the logic for gathering data, executing a script,
    and disseminating the results for a single location.
    """
    def __init__(self, device, script, location, neighbours):
        """
        Initializes the script-execution thread.

        Args:
            device (Device): The parent device that owns this thread.
            script (Script): The script object to execute.
            location (int): The data location the script will operate on.
            neighbours (list): A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        The core logic for script execution.

        It acquires a location-specific lock, gathers data from its parent
        device and its neighbors, runs the script, and writes the result
        back to all involved devices before releasing the lock.
        """
        # Invariant: Acquire a lock to ensure exclusive access to the data at this location
        # across all devices, preventing race conditions during read-modify-write.
        self.device.locks[self.location].acquire()
        script_data = []
        
        # Block Logic: Aggregate input data for the script.
        # It collects data from the same location from all neighbors.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        data = self.device.get_data(self.location)
        
        if data is not None:
            script_data.append(data)

        # Block Logic: Execute the script and disseminate the result.
        # Pre-condition: Only run if there is data to process.
        if script_data != []:
            # The script performs a computation on the aggregated data.
            result = self.script.run(script_data)
            
            # The result is written back to the parent device and all neighbors,
            # effectively updating the state at this location across the local neighborhood.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            self.device.set_data(self.location, result)
        
        self.device.locks[self.location].release()

class DeviceThread(Thread):
    """
    The main, long-running thread for a Device.

    This thread orchestrates the device's participation in the simulation's
    synchronized time steps.
    """
    def __init__(self, device):
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.nr_iter = None

    def run(self):
        """
        The main control loop for the device.
        """
        # The loop continues as long as the supervisor provides neighbors,
        # indicating the simulation is ongoing.
        while True:
            # Fetches the current set of neighbors from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # A None value signals the end of the simulation.
                break
            
            # Block until the supervisor signals that all scripts for the
            # current timepoint have been assigned.
            self.device.timepoint_done.wait()
            
            # This logic partitions the assigned scripts into batches of 8.
            # This appears to be a strategy to manage thread creation overhead
            # or resource contention, but the fixed size of 8 is arbitrary.
            self.nr_iter = len(self.device.scripts) / 8
            
            if self.nr_iter == 0:
                # If there are fewer than 8 scripts, run them all at once.
                scriptthreads = []
                for (script, location) in self.device.scripts:
                    scriptthreads.append(MiniDeviceThread(self.device, script, location, neighbours))
                
                for _, thread in enumerate(scriptthreads):
                    thread.start()
                
                for _, thread in enumerate(scriptthreads):
                    thread.join()
            
            else:
                # If there are 8 or more scripts, run them in batches of 8.
                count = 0
                size = 8
                for _ in range(self.nr_iter):
                    scriptthreads = []
                    for idx in range(count, size):
                        script = self.device.scripts[idx][0]
                        location = self.device.scripts[idx][1]
                        scriptthreads.append(MiniDeviceThread(self.device, script, location, neighbours))
                    
                    for _, thread in enumerate(scriptthreads):
                        thread.start()
	                
                    for _, thread in enumerate(scriptthreads):
                        thread.join()

                    count = count + 8
                    # Adjust the size for the final, potentially smaller, batch.
                    if size + 8 > len(self.device.scripts):
                        size = len(self.device.scripts) - size
                    else:
                        size = size + 8
            
            # Invariant: Wait at the barrier to synchronize with all other devices.
            # This ensures that no device begins the next timepoint until all
            # devices have completed their work for the current one.
            self.device.barrier.wait()
            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()
