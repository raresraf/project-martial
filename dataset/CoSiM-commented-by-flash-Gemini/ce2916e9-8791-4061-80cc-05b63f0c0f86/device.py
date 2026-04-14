"""
@ce2916e9-8791-4061-80cc-05b63f0c0f86/device.py
@brief Implements a distributed device simulation framework with thread-based script execution and a reusable barrier for synchronization.

This module defines the core components for simulating distributed devices that process sensor data
through scripts. It includes a `Device` class representing an individual device, a `DeviceThread`
for managing script execution on each device, a `SolveScript` thread for executing individual scripts,
and a `ReusableBarrierSem` for synchronizing multiple threads.

Domain: Distributed Systems Simulation, Concurrency, Thread Synchronization, Sensor Networks.
"""

from threading import Event, Semaphore, Lock, Thread
from Queue import Queue

class Device(object):
    """
    @brief Represents a simulated device in a distributed system, capable of processing sensor data.

    Each device has a unique ID, sensor data, and a supervisor for coordination. It manages
    the execution of assigned scripts in a dedicated thread and uses events for synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor readings, keyed by location.
        @param supervisor: A reference to the supervisor object managing this device.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that scripts have been received for the current timepoint.
        self.scripts_received = Event()
        self.scripts = []
        # Event to signal that processing for the current timepoint is complete.
        self.timepoint_done = Event()
        # Dedicated thread for the device's operational logic.
        self.thread = DeviceThread(self)
        self.thread.start()
        # Number of worker threads to process scripts concurrently.
        self.no_th = 8

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up synchronization primitives and shared resources among devices.

        This method is primarily intended for device_id 0 to initialize a shared barrier
        and location-specific locks, which are then distributed to other devices.
        
        @param devices: A list of all Device instances in the simulation.
        """
        # Block Logic: Initializes shared synchronization primitives if this is the first device.
        # Invariant: Only device with ID 0 initializes the barrier and locks to prevent redundant setup.
        if self.device_id == 0:
            # Reusable barrier for synchronizing all devices at the end of a timepoint.
            barrier = ReusableBarrierSem(len(devices))
            # Dictionary to hold locks for each sensor data location, ensuring exclusive access.
            lock_for_loct = {}
            # Propagate the barrier to all devices.
            for device in devices:
                device.barrier = barrier
                # Pre-condition: Iterate through each device's sensor data locations.
                # Invariant: A lock is created for each unique location if it doesn't already exist.
                for location in device.sensor_data:
                    if location not in lock_for_loct:
                        lock_for_loct[location] = Lock()
                # Propagate the location-specific locks to all devices.
                device.lock_for_loct = lock_for_loct

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed on this device for a specific location.

        If `script` is None, it signals that all scripts for the current timepoint have been assigned.

        @param script: The script object to assign, or None to signal completion.
        @param location: The sensor data location this script pertains to.
        """
        # Block Logic: Appends a script and its location to the device's script list.
        # Pre-condition: The script object is not None.
        if script is not None:
            self.scripts.append((script, location))
        # Block Logic: Signals that all scripts for the current timepoint have been received.
        else:
            self.scripts_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.

        @param location: The location for which to retrieve sensor data.
        @return: The sensor data for the given location, or None if the location is not found.
        """
        # Functional Utility: Safely retrieves sensor data, returning None for missing keys.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data, source=None):
        """
        @brief Sets or updates sensor data for a specified location.

        @param location: The location for which to set or update sensor data.
        @param data: The new sensor data value.
        @param source: Optional; the source device of the data (not currently used but provided for context).
        """
        # Block Logic: Updates sensor data if the location already exists.
        # Invariant: Existing sensor data for the location is overwritten with the new data.
        if location in self.sensor_data:
            self.sensor_data[location] = data


    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its processing thread to complete.
        """
        # Functional Utility: Ensures the device's thread finishes its execution before the program exits.
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the lifecycle of script execution on a Device.

    This thread continuously fetches neighboring device information, waits for script assignments,
    and dispatches them to worker threads (`SolveScript`) for parallel processing. It also
    synchronizes with other DeviceThreads using a barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread.

        @param device: The Device instance this thread is managing.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = [] # Queue of scripts to be processed.
        self.neighbours = [] # List of neighboring devices.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Continuously fetches neighbors, waits for scripts, dispatches SolveScript threads,
        and synchronizes with other device threads.
        """
        # Block Logic: Main loop for device operations.
        # Invariant: Continues until the supervisor signals termination (neighbours is None).
        while True:
            # Functional Utility: Retrieves the current list of neighboring devices from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If no neighbors are returned, it signifies termination.
            if self.neighbours is None:
                break

            # Block Logic: Waits for new script assignments for the current timepoint.
            selfion: Clears the event so it can be set again for the next timepoint.
            self.device.scripts_received.wait()
            self.device.scripts_received.clear()
            
            # Functional Utility: Initializes a new Queue for the current set of scripts.
            self.queue = Queue()
            # Block Logic: Populates the queue with all assigned scripts.
            # Invariant: All scripts for the current timepoint are added to the queue.
            for script in self.device.scripts:
                self.queue.put_nowait(script)

            # Block Logic: Spawns multiple worker threads to concurrently solve scripts.
            # Invariant: 'no_th' number of SolveScript threads are started to process the queue.
            for _ in range(self.device.no_th):
                SolveScript(self.device, self.neighbours, self.queue).start()
            
            # Functional Utility: Blocks until all items in the queue have been processed by SolveScript threads.
            self.queue.join()
            
            # Functional Utility: Synchronizes with other DeviceThreads, waiting for all to complete their timepoint processing.
            self.device.barrier.wait()

class SolveScript(Thread):
    """
    @brief A worker thread responsible for executing a single script associated with a sensor data location.

    It acquires a lock for the specific location, retrieves data from neighboring devices and
    its own device, executes the script, and then updates data on neighbors and itself.
    """

    def __init__(self, device, neighbours, queue):
        """
        @brief Initializes a new SolveScript thread.

        @param device: The parent Device instance.
        @param neighbours: A list of neighboring Device instances.
        @param queue: The Queue containing scripts to be processed.
        """
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.queue = queue

    def run(self):
        """
        @brief The main execution logic for a SolveScript thread.

        It continuously fetches scripts from the queue, processes them, and handles synchronization.
        """
        # Block Logic: Ensures that any exceptions during script processing do not crash the worker thread.
        try:
            # Block Logic: Iterates through the scripts assigned to the device for the current timepoint.
            # Pre-condition: The queue contains (script, location) tuples.
            # Invariant: Each script is processed with appropriate locking and data exchange.
            for (script, location) in self.device.scripts: # This loop seems redundant with queue.get(False)
                # Functional Utility: Retrieves a script and its associated location from the queue without blocking.
                (script, location) = self.queue.get(False)
                
                # Functional Utility: Acquires a lock for the specific sensor data location to prevent race conditions.
                self.device.lock_for_loct[location].acquire()

                script_data = []
                # Block Logic: Gathers sensor data from neighboring devices for the current location.
                # Invariant: 'script_data' contains valid data from neighbors for the given location.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Functional Utility: Gathers sensor data from its own device for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if valid data has been collected.
                # Pre-condition: 'script_data' is not empty.
                if script_data != []:
                    # Functional Utility: Executes the assigned script with the collected data.
                    result = script.run(script_data)
                    
                    # Block Logic: Updates the sensor data on all neighboring devices with the script's result.
                    # Invariant: All neighbors' data for the location is consistent with the script's output.
                    for device in self.neighbours:
                        device.set_data(location, result)
                    
                    # Functional Utility: Updates the sensor data on its own device with the script's result.
                    self.device.set_data(location, result)

                # Functional Utility: Releases the lock for the sensor data location, allowing other threads access.
                self.device.lock_for_loct[location].release()
                
                # Functional Utility: Signals that this task (script processing) is done for the queue.
                self.queue.task_done()
        # Block Logic: Catches any exceptions during script execution and prevents thread termination.
        # Invariant: The thread continues its operation even if an individual script fails.
        except:
            pass

class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier synchronization mechanism using semaphores and a lock.

    This barrier allows a fixed number of threads to wait for each other at a synchronization
    point and then proceed together, and can be reused multiple times.
    Algorithm: Double-phase semaphore-based barrier.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes a ReusableBarrierSem instance.

        @param num_threads: The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        # Counter for threads in the first phase of the barrier.
        self.count_threads1 = self.num_threads
        # Counter for threads in the second phase of the barrier.
        self.count_threads2 = self.num_threads
        # Lock to protect the shared thread counters.
        self.counter_lock = Lock()               
        # Semaphore for the first phase, initialized to 0 (all threads block initially).
        self.threads_sem1 = Semaphore(0)         
        # Semaphore for the second phase, initialized to 0.
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this point.

        Executes both phases of the barrier.
        """
        # Functional Utility: Orchestrates the two-phase synchronization.
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief The first phase of the barrier.

        Threads decrement a counter and the last thread releases all waiting threads for phase 1.
        """
        # Block Logic: Atomically decrements the counter and releases threads if this is the last one.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Pre-condition: This is the last thread to arrive at the barrier for phase 1.
            if self.count_threads1 == 0:
                # Functional Utility: Releases all threads waiting on threads_sem1.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Functional Utility: Resets the counter for future reuse of the barrier.
                self.count_threads1 = self.num_threads

        # Functional Utility: Blocks the current thread until it is released by the last thread in phase 1.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief The second phase of the barrier.

        Threads decrement a second counter and the last thread releases all waiting threads for phase 2.
        This design allows the barrier to be reusable.
        """
        # Block Logic: Atomically decrements the counter and releases threads if this is the last one.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Pre-condition: This is the last thread to arrive at the barrier for phase 2.
            if self.count_threads2 == 0:
                # Functional Utility: Releases all threads waiting on threads_sem2.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Functional Utility: Resets the counter for future reuse of the barrier.
                self.count_threads2 = self.num_threads

        # Functional Utility: Blocks the current thread until it is released by the last thread in phase 2.
        self.threads_sem2.acquire()
