

"""
@950c3d7d-afad-490b-ad6c-379e73da4ca1/device.py
@brief Implements a multi-threaded simulation for distributed sensor devices using a custom reusable barrier.

This module defines the core components for simulating a network of sensor devices,
each capable of executing scripts, managing local sensor data, and interacting
with a central supervisor. This version features a `ReusableBarrier` for
efficient synchronization of multiple threads across different phases,
and dynamically allocated `RLock` objects for location-specific data protection.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate with neighbors under the
guidance of a supervisor.

Classes:
- ReusableBarrier: A custom barrier implementation for multi-threaded synchronization.
- Device: Represents a single simulated sensor device.
- DeviceThread: Manages the lifecycle and operation of a Device, including coordinating
                `MyThread` instances.
- MyThread: A worker thread responsible for executing a single script for a specific location.

Domain: Distributed Systems Simulation, Concurrent Programming, Parallel Processing, Custom Synchronization Primitives.
"""

from threading import Event, Thread, Lock, Semaphore, RLock


class ReusableBarrier(object):
    """
    @brief A custom barrier implementation that allows multiple threads to wait
           until all have reached a common point, and then can be reused.

    This barrier uses a two-phase semaphore approach to ensure correct synchronization
    without deadlocks and allows for repeated use in a simulation loop.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier.

        @param num_threads: The total number of threads that will participate in this barrier.
        """
        self.num_threads = num_threads
        # Tracks the count of threads in the first phase.
        self.count_threads1 = [self.num_threads]
        # Tracks the count of threads in the second phase.
        self.count_threads2 = [self.num_threads]
        # Lock to protect access to thread counts.
        self.count_lock = Lock()
        # Semaphore for the first phase of the barrier.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for the second phase of the barrier.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all
               participating threads have called `wait()`.
        """
        # Block Logic: Executes the first phase of the barrier.
        self.phase(self.count_threads1, self.threads_sem1)
        # Block Logic: Executes the second phase of the barrier. This two-phase
        # approach ensures the barrier is reusable.
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Implements a single phase of the reusable barrier.

        This method decrements a counter and, when it reaches zero, releases all
        waiting threads. It then resets the counter for the next use.

        @param count_threads: A list (to allow pass-by-reference) containing the current thread count for this phase.
        @param threads_sem: The Semaphore associated with this phase.
        """
        with self.count_lock:
            # Pre-condition: Lock is acquired, ensuring atomic decrement of thread count.
            # Invariant: `count_threads[0]` accurately reflects the number of threads yet to reach the barrier.
            count_threads[0] -= 1
            if count_threads[0] == 0:
                # Block Logic: All threads have reached the barrier. Release all waiting threads.
                i = 0
                while i < self.num_threads:
                    threads_sem.release()
                    i += 1
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # Block Logic: All threads wait on the semaphore until it's released by the last thread.
        threads_sem.acquire()


class Device(object):
    """
    @brief Represents a single simulated sensor device in a distributed network.

    Each device manages its own sensor data, interacts with a supervisor,
    and executes assigned scripts in a multi-threaded environment. This version
    uses a global `ReusableBarrier` and dynamically allocated `RLock` objects
    for location-specific data protection.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data, keyed by location.
        @param supervisor: A reference to the central supervisor managing all devices.
        """
        # Unique identifier for this device.
        self.device_id = device_id
        # Dictionary storing sensor data, keyed by location.
        self.sensor_data = sensor_data
        # Reference to the central supervisor.
        self.supervisor = supervisor
        # List to store assigned scripts, each being a tuple of (script, location).
        self.scripts = []
        # A list containing all device instances in the simulation.
        self.devices = []
        # Event to signal that a script has been received and processed for the current timepoint.
        self.script_received = Event()
        # Event to signal that all scripts for the current timepoint have been processed.
        self.timepoint_done = Event()
        # The main thread responsible for the device's lifecycle.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up global synchronization resources for all devices.

        This method populates the `self.devices` list with all device instances.
        It also initializes a global `ReusableBarrier` and a dictionary for
        location-specific `RLock` objects on the first device (device 0).

        @param devices: A list of all Device instances in the simulation.
        """
        # Block Logic: Populates the `self.devices` list with all participating devices.
        for device in devices:
            self.devices.append(device)
        # Block Logic: Initializes the global `ReusableBarrier` and `locations_lock` dictionary
        # on the first device (device 0) to ensure shared access.
        self.devices[0].barrier = ReusableBarrier(len(self.devices))
        self.devices[0].locations_lock = {}

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed for a specific location on this device.

        If `script` is None, it signals that the device has received all scripts for
        the current timepoint and is ready to process.

        @param script: The script object to be executed, or None to signal completion.
        @param location: The location pertinent to the script execution.
        """
        if script is not None:
            # Pre-condition: `script` is not None.
            # Invariant: The script is added to the device's list of pending scripts.
            self.scripts.append((script, location))
        else:
            # Pre-condition: `script` is None, signaling end of script assignment for this timepoint.
            # Invariant: `timepoint_done` and `script_received` events are set to allow processing to begin.
            self.timepoint_done.set()
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The location for which to retrieve data.
        @return: The sensor data for the specified location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location: The location for which to set data.
        @param data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's main thread.

        Waits for the device's main thread to complete its execution.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the execution lifecycle for a single Device instance in a dedicated thread.

    This thread is responsible for handling timepoint progression, spawning `MyThread`
    instances for script execution, and synchronizing with other devices using a global barrier.
    It manages a fixed number of concurrent `MyThread` instances to process scripts in batches.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.

        @param device: The Device instance that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # The maximum number of `MyThread` instances to run concurrently.
        self.num_threads = 8

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        Pre-condition: The device's synchronization mechanisms are properly set up.
        Invariant: The device continuously processes timepoints, executes assigned scripts
                   in parallel batches, and synchronizes with other devices until a shutdown signal is received.
        """
        while True:
            # List to store `MyThread` instances for batch execution.
            threads = []
            
            # Block Logic: Fetches the current neighbors of this device from the supervisor.
            # This allows for dynamic network topology changes between timepoints.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Pre-condition: `neighbours` is None, indicating a termination signal from the supervisor.
                # Invariant: The loop breaks, leading to thread termination.
                break
            
            # Block Logic: Waits for the supervisor to signal that all scripts for the current
            # timepoint have been assigned.
            self.device.script_received.wait()

            # Block Logic: Creates `MyThread` instances for each assigned script.
            for (script, location) in self.device.scripts:
                thread = MyThread(self, script, location, neighbours)
                threads.append(thread)

            # Calculates the number of full batches and remaining scripts.
            rounds = len(self.device.scripts) / self.num_threads
            leftovers = len(self.device.scripts) % self.num_threads
            
            # Block Logic: Executes scripts in full batches of `self.num_threads`.
            # Each batch of threads is started, waited upon, and then cleared.
            while rounds > 0:
                for j in xrange(self.num_threads):
                    threads[j].start()
                for j in xrange(self.num_threads):
                    threads[j].join()
                for j in xrange(self.num_threads):
                    threads.pop(0) # Removes processed threads from the list.
                rounds -= 1
            
            # Block Logic: Executes any remaining scripts that didn't form a full batch.
            for j in xrange(leftovers):
                threads[j].start()
            for j in xrange(leftovers):
                threads[j].join()
            for j in xrange(leftovers):
                threads.pop(0) # Removes processed threads from the list.

            # Clears the entire list of `MyThread` instances for the next timepoint.
            del threads[:]
            
            # Block Logic: Synchronizes with all other DeviceThreads using the global barrier.
            # This ensures all devices have finished processing their scripts before proceeding.
            self.device.devices[0].barrier.wait()
            
            # Resets the `script_received` event, preparing for the next timepoint.
            self.device.script_received.clear()


class MyThread(Thread):
    """
    @brief A worker thread responsible for executing a single script for a specific location.

    This thread ensures that data access for its assigned location is synchronized
    using a `RLock` object, preventing race conditions during script execution
    and data updates across devices.
    """

    def __init__(self, device_thread, script, location, neighbours):
        """
        @brief Initializes the MyThread.

        @param device_thread: A reference to the parent DeviceThread instance.
        @param script: The script object to be executed.
        @param location: The specific location for which the script will be executed.
        @param neighbours: A list of neighboring Device instances to interact with.
        """
        Thread.__init__(self)
        self.device_thread = device_thread
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution method for the MyThread.

        Pre-condition: The script, neighbors, and location are properly initialized.
        Invariant: The script is executed, and relevant data is updated while holding
                   the location-specific `RLock`.
        """
        # Block Logic: Dynamically creates an `RLock` for the current location if one doesn't exist.
        # This ensures that only one thread can modify data for this location at a time.
        if self.location not in self.device_thread.device.devices[0].locations_lock:
            self.device_thread.device.devices[0].locations_lock[self.location] = RLock()
        
        # Critical Section: Acquires the `RLock` for the specific location.
        # Ensures exclusive access to the data associated with this location during script
        # execution and data updates, preventing race conditions.
        with self.device_thread.device.devices[0].locations_lock[self.location]:
            script_data = []
            
            # Block Logic: Gathers data from neighboring devices for the current location.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)
            
            # Gathers data from its own sensor_data for the current location.
            data = self.device_thread.device.get_data(self.location)
            if data is not None:
                script_data.append(data)
            
            if script_data != []:
                # Executes the script with the collected data.
                result = self.script.run(script_data)
                
                # Block Logic: Updates data on neighboring devices with the script's result.
                for device in self.neighbours:
                    device.set_data(self.location, result)
                
                # Updates its own data with the script's result.
                self.device_thread.device.set_data(self.location, result)

