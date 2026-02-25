


"""
@3b22b207-999d-4edd-b65f-815a9f122d23/device.py
@brief Implements a simulated device for a distributed sensor network,
       including multithreaded script execution and synchronized data processing.

This module defines the architecture for individual devices in a sensor
network. Each device manages its sensor data, receives and executes scripts,
and collaborates with a supervisor and neighboring devices. It leverages
threading primitives like Events, Threads, Locks, and Semaphores for concurrent
script processing and a reusable barrier for global synchronization across devices.
"""


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier(object):
    """
    @brief A reusable barrier synchronization primitive for coordinating
           multiple threads.

    This barrier allows a fixed number of threads to wait until all have
    reached a certain point, then releases them all simultaneously. It can
    be reused multiple times after all threads have passed.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes a new instance of the ReusableBarrier.

        @param num_threads: The total number of threads that must reach
                            the barrier before it releases.
        """
        self.num_threads = num_threads
        # Two counters are used to manage the two phases of the barrier,
        # preventing lost wake-ups in a reusable scenario.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        # Mutex to protect access to the thread counters.
        self.count_lock = Lock()
        # Semaphores to block and release threads in each phase.
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` threads
               have reached this barrier across two phases.

        This method ensures that all participating threads synchronize twice,
        which is crucial for reusing the barrier safely.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Manages a single synchronization phase of the barrier.

        Decrements a counter and, if it reaches zero, releases all waiting
        threads via a semaphore. Otherwise, the thread waits.

        @param count_threads: A list containing the current count of threads
                              waiting in this phase. (List is used for mutability)
        @param threads_sem: The semaphore used to block and release threads
                            in this specific phase.
        """
        with self.count_lock:
            # Atomically decrement the count of threads waiting.
            count_threads[0] -= 1
            # Check if this is the last thread to arrive at the barrier.
            if count_threads[0] == 0:
                n_threads = self.num_threads
                # Block Logic: Release all waiting threads.
                while n_threads > 0:
                    threads_sem.release()
                    n_threads -= 1
                # Reset the counter for the next use of this phase.
                count_threads[0] = self.num_threads
        # Acquire the semaphore; if not the last thread, it blocks here.
        threads_sem.acquire()


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.

    Each device has a unique ID, stores sensor data, executes assigned scripts,
    and collaborates with other devices for synchronized data processing.
    It manages its own script queue and communicates its state to a supervisor.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's sensor readings,
                            keyed by location.
        @param supervisor: A reference to the supervisor object for inter-device
                           communication and network topology information.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to the device.
        self.script_received = Event()
        # List to store assigned scripts, each being a (script_object, location) tuple.
        self.scripts = []
        # Event to signal when the current timepoint's script processing is done.
        self.timepoint_done = Event()
        # The main processing thread for this device.
        self.thread = DeviceThread(self)
        # List of all devices in the network.
        self.devices = []
        # Reference to the shared barrier for device synchronization.
        self.barrier = None
        # List to hold active ScriptWorker threads.
        self.workers = []
        # Initialize location-specific barriers (locks) for thread-safe access.
        # This creates a lock for each location up to 60.
        keys = range(60)
        self.loc_barrier = {key: None for key in keys}
        self.thread.start() # Start the main device processing thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared synchronization barrier for a group of devices.

        This method ensures that all devices share the same `ReusableBarrier` instance.
        It also populates the device's internal list of all network devices.

        @param devices: A list of all Device objects in the network.
        """
        # Block Logic: Initializes a shared barrier if not already set.
        # Invariant: All devices in the 'devices' list will be assigned the same barrier.
        if self.barrier is None:
            barrier = ReusableBarrier(len(devices))
            self.barrier = barrier
            for device in devices:
                if device.barrier is None:
                    device.barrier = barrier

        # Populate the device's list of all devices in the network.
        for device in devices:
            if device is not None:
                self.devices.append(device)

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.

        If a script is provided, it's added to the device's script queue. Location-specific
        locks are managed: if a lock for the location doesn't exist on this device, it
        attempts to use one from another device or creates a new one. An event is set
        to notify the device's processing thread that new scripts are available.
        If `script` is None, it signals the end of the current timepoint for script assignment.

        @param script: The script object to be executed, or None.
        @param location: The data location relevant to the script.
        """

        if script is not None:
            self.scripts.append((script, location))
            # Block Logic: Manages location-specific locks.
            # Ensures that a lock exists for the current location, potentially reusing one.
            if self.loc_barrier[location] is None:
                # Search for an existing lock for this location among other devices.
                for device in self.devices:
                    if device.loc_barrier[location] is not None:
                        self.loc_barrier[location] = device.loc_barrier[location]
                        break

            # If no existing lock found, create a new one.
            if self.loc_barrier[location] is None:
                self.loc_barrier[location] = Lock()
            self.script_received.set() # Signal the DeviceThread that new scripts arrived.
        else:
            self.timepoint_done.set() # Signal that no more scripts are coming for this timepoint.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The location for which to retrieve data.
        @return: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location: The location at which to set the data.
        @param data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its processing thread to complete.
        """
        self.thread.join()


class ScriptWorker(Thread):
    """
    @brief A worker thread responsible for executing a single script for a Device.

    ScriptWorkers gather data, run the script's logic, and update sensor data
    for the local device and its neighbors, ensuring thread-safe access to locations
    using location-specific locks.
    """
    
    def __init__(self, device, neighbours, script, location):
        """
        @brief Initializes a ScriptWorker thread.

        @param device: The parent Device object.
        @param neighbours: A list of neighboring Device objects from which to collect data.
        @param script: The script object to execute.
        @param location: The data location pertinent to this script.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief The main execution logic for the ScriptWorker thread.

        It acquires a lock for the specific data location, collects data from
        the local device and its neighbors, runs the assigned script, updates
        the sensor data across relevant devices, and then releases the lock.
        """
        self.device.loc_barrier[self.location].acquire() # Acquire lock for the specific location.
        script_data = [] # List to hold data collected for the script.
        
        # Block Logic: Collect data from neighboring devices at the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Collect data from the local device at the specified location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        # Block Logic: If any script data was collected, execute the script and update data.
        if script_data != []:
            result = self.script.run(script_data) # Execute the script with collected data.
            # Update sensor data for all neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            # Update sensor data for the local device.
            self.device.set_data(self.location, result)
        self.device.loc_barrier[self.location].release() # Release lock for the specific location.


class DeviceThread(Thread):
    """
    @brief The main processing thread for a Device.

    This thread manages script execution for its associated device. It coordinates
    with other devices using a barrier, fetches neighbors' data, and dispatches
    scripts to `ScriptWorker` threads for processing.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread with a reference to its parent device.

        @param device: The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        It continuously orchestrates script processing: it waits for a timepoint
        to be done, launches `ScriptWorker` threads for each assigned script,
        waits for all workers to complete, and then synchronizes with other
        device threads using a global barrier. The loop terminates if the
        supervisor signals no more neighbors (end of simulation).
        """
        # Block Logic: Main loop for continuous operation of the DeviceThread.
        while True:
            # Get the current list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If supervisor returns None for neighbors, it indicates simulation end.
            if neighbours is None:
                break # Exit the main device thread loop if no more neighbors.

            self.device.timepoint_done.wait() # Wait until the current timepoint is marked as done.

            # Block Logic: For each assigned script, create and store a ScriptWorker thread.
            for (script, location) in self.device.scripts:
                thread = ScriptWorker(self.device, neighbours, script, location)
                self.device.workers.append(thread)

            # Block Logic: Start all ScriptWorker threads.
            for worker in self.device.workers:
                worker.start()

            # Block Logic: Wait for all ScriptWorker threads to complete their execution.
            for worker in self.device.workers:
                worker.join()

            self.device.workers = [] # Clear the list of workers for the next timepoint.
            self.device.timepoint_done.clear() # Clear timepoint_done for the next cycle.
            # Synchronization Point: Wait for all devices to finish their timepoint processing.
            self.device.barrier.wait()
