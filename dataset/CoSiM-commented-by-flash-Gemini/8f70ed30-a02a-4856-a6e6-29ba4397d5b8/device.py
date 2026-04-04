


"""
@8f70ed30-a02a-4856-a6e6-29ba4397d5b8/device.py
@brief Implements a multi-threaded simulation for distributed sensor devices with dynamic worker thread management.

This module defines the core components for simulating a network of sensor devices,
each capable of executing scripts, managing local sensor data, and interacting
with a central supervisor. This version features a dynamic pool of worker threads
(`MyThread`) spawned by a `DeviceThread` to execute scripts, and a custom
`ReusableBarrierSem` for synchronization across devices.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate with neighbors under the
guidance of a supervisor.

Classes:
- Device: Represents a single simulated sensor device.
- MyThread: A worker thread responsible for executing individual scripts.
- DeviceThread: Manages the lifecycle and operation of a Device, including spawning
                and managing worker threads.
- ReusableBarrierSem: A custom barrier implementation for thread synchronization.

Domain: Distributed Systems Simulation, Concurrent Programming, Dynamic Thread Pooling, Sensor Networks.
"""

from threading import Event, Thread, Lock, Semaphore

class Device(object):
    """
    @brief Represents a single simulated sensor device in a distributed network.

    Each device manages its own sensor data, interacts with a supervisor,
    and executes assigned scripts in a multi-threaded environment. It uses
    a dedicated `DeviceThread` to manage worker threads (`MyThread`) for script
    execution and relies on a `ReusableBarrierSem` for timepoint synchronization.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        Sets up the device's unique identifier, its initial sensor data,
        a reference to the central supervisor, and initializes various
        synchronization primitives and state variables required for
        multi-threaded operation.

        @param device_id: A unique integer identifier for the device.
        @param sensor_data: A dictionary containing the device's initial sensor readings.
                            Keys are location IDs, values are sensor data.
        @param supervisor: A reference to the Supervisor object managing the device network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Synchronization primitive: Event to signal that a new script has been assigned to the device.
        self.script_received = Event()
        self.scripts = [] # Stores scripts assigned to this device for execution. Each script is (script_object, location_id).
        # Synchronization primitive: Event to signal that all scripts for the current timepoint have been processed.
        self.timepoint_done = Event()
        # Synchronization primitive: A dictionary of locks, one for each location, to protect concurrent
        # access to sensor data at specific locations.
        self.lock = {}
        # Synchronization primitive: A barrier to synchronize all devices at specific points in the simulation.
        self.barrier = None
        self.devices = [] # A list of all Device objects in the simulation.
        # Spawns a dedicated thread to manage this device's operations.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the device's awareness of all other devices in the simulation
               and initializes shared synchronization primitives.

        This method is called once at the beginning of the simulation. It
        initializes a shared `ReusableBarrierSem` and a dictionary of `Lock`
        objects for thread-safe access to sensor data locations across all devices.

        @param devices: A list of all Device objects participating in the simulation.
        """
        self.devices = devices
        # Functional Utility: Initializes a reusable barrier for all devices in the simulation.
        self.barrier = ReusableBarrierSem(len(self.devices))

        # Block Logic: Initializes a Lock for each unique sensor data location across all devices.
        # This ensures that any access to a specific location's data is thread-safe.
        for location in self.sensor_data:
            self.lock[location] = Lock()
        for device in devices:
            for location in device.sensor_data:
                self.lock[location] = Lock()

        # Block Logic: Propagates the initialized shared barrier and location locks to all devices.
        # This centralizes synchronization and resource management.
        for i in xrange(len(self.devices)):
            self.devices[i].barrier = self.barrier
            self.devices[i].lock = self.lock

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device to be executed at a specific location
               or signals the completion of a timepoint if no script is provided.

        @param script: The script object to be executed, or None if the timepoint is done.
        @param location: The location ID associated with the script, or irrelevant if script is None.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Block Logic: If no script is provided (script is None), it signifies that
            # all scripts for the current timepoint have been assigned.
            self.script_received.set() # Signals that script assignments are complete for the timepoint.
            self.timepoint_done.set()  # Signals that the timepoint processing is logically done (no more scripts to assign).

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The location ID for which to retrieve data.
        @return The sensor data for the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.

        This method directly updates the `sensor_data` dictionary for the given location.

        @param location: The location ID for which to set data.
        @param data: The new sensor data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown process for the device by joining its associated thread.

        This ensures that the `DeviceThread` completes its execution before
        the program exits.
        """
        self.thread.join()


class MyThread(Thread):
    """
    @brief A worker thread (`MyThread`) responsible for executing a single script for a Device.

    These threads are dynamically created and managed by `DeviceThread` to
    process scripts. Each `MyThread` acquires a lock for its assigned location,
    gathers data from neighbors and its own device, executes the script, and
    updates the sensor data for all relevant devices.
    """

    def __init__(self, my_id, device, neighbours, lock, script, location):
        """
        @brief Initializes a new MyThread instance.

        @param my_id: A unique integer identifier for this worker thread.
        @param device: The Device object that this worker thread belongs to.
        @param neighbours: A list of neighboring Device objects from which to collect data.
        @param lock: The shared dictionary of locks for sensor data locations.
        @param script: The script object to be executed.
        @param location: The location ID associated with the script.
        """
        Thread.__init__(self, name="Thread %d from device %d" % (my_id, device.device_id))
        self.device = device
        self.my_id = my_id
        self.neighbours = neighbours
        self.lock = lock
        self.script = script
        self.location = location

    
    def run(self):
        """
        @brief The main execution logic for the MyThread.

        This method acquires a lock for its assigned location, collects sensor data
        from the device and its neighbors, executes the script with this data,
        and then updates the sensor data in all relevant devices.
        """
        # Block Logic: Acquires a location-specific lock to ensure exclusive access
        # to the sensor data at this particular location during script execution and data update.
        # Invariant: Only one thread can operate on data at a given 'location' at any time.
        with self.lock[self.location]:
            script_data = []
            
            # Block Logic: Gathers sensor data from neighboring devices for the specified location.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            # Block Logic: Gathers sensor data from the current device for the specified location.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Pre-condition: If valid script data was collected (either from neighbors or self).
            if script_data != []:
                # Functional Utility: Executes the assigned script with the collected sensor data.
                # The script's `run` method presumably contains the core logic for data processing.
                result = self.script.run(script_data)

                # Block Logic: Updates the sensor data for all neighboring devices with the script's result.
                for device in self.neighbours:
                    device.set_data(self.location, result)

                # Block Logic: Updates the sensor data for the current device with the script's result.
                self.device.set_data(self.location, result)

    def shutdown(self):
        """
        @brief Joins the thread, ensuring its completion.
        """
        self.join()


class DeviceThread(Thread):
    """
    @brief Manages the lifecycle and operation of a Device, including spawning
           and managing worker threads (`MyThread`).

    This thread is responsible for orchestrating the device's activities across
    timepoints, including retrieving neighbor information from the supervisor,
    dynamically creating and joining worker threads to execute scripts, and
    synchronizing with other devices via a shared barrier.
    """
    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)

        self.device = device
        self.numThreads = 0 # Counter for assigning unique IDs to dynamically created worker threads.

        # List to keep track of currently active and completed worker threads (`MyThread` instances).
        self.listThreads = []

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        It continuously retrieves neighbor information, waits for scripts to be
        assigned, dynamically manages worker threads to execute these scripts,
        and then synchronizes with other devices at the end of each timepoint.
        """
        while True:
            # Block Logic: Retrieves information about neighboring devices from the supervisor.
            # This information is crucial for scripts that need to interact with or gather
            # data from adjacent devices in the simulated environment.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If there are no more neighbors (end of simulation), break the loop.
            if neighbours is None:
                break

            # Pre-condition: Waits until the `Device` signals that new scripts have been assigned
            # for the current timepoint.
            self.device.script_received.wait()

            # Block Logic: Iterates through assigned scripts and dispatches them to worker threads.
            # It dynamically manages a pool of worker threads, reusing them when possible.
            for (script, location) in self.device.scripts:
                # Invariant: Limits the number of concurrently running worker threads to 8.
                if len(self.listThreads) < 8:
                    # Functional Utility: Creates and starts a new worker thread to execute the script.
                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.append(thread)
                    thread.start()
                    self.numThreads += 1
                else:
                    # Block Logic: If the thread pool is full, it finds a completed thread to reuse its slot.
                    index = -1
                    for i in xrange(len(self.listThreads)):
                        if not self.listThreads[i].is_alive():
                            self.listThreads[i].join() # Joins the completed thread to reclaim resources.
                            index = i
                            break # Found a slot, exit loop.

                    self.listThreads.pop(index) # Removes the old thread object from the list.
                    # Functional Utility: Creates and starts a new worker thread to execute the script in the reclaimed slot.
                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.insert(index,thread) # Inserts the new thread into the now-empty slot.
                    self.listThreads[index].start()
                    self.numThreads += 1

            # Block Logic: Waits for all currently active worker threads to complete their tasks.
            for i in xrange(len(self.listThreads)):
                self.listThreads[i].join()

            # Pre-condition: Waits until the `Device` signals that the current timepoint's processing is done.
            self.device.timepoint_done.wait()
            
            # Post-condition: Clears the events for the next timepoint.
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            
            # Synchronization point: All devices wait here until every other device
            # has completed its current timepoint processing. This ensures that
            # all devices are synchronized before moving to the next timepoint.
            self.device.barrier.wait()



class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier using semaphores for synchronizing a fixed number of threads.

    This barrier allows a specified number of threads to wait at a synchronization
    point and then proceed together. It is designed to be reusable for multiple
    synchronization cycles.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrierSem instance.

        @param num_threads: The total number of threads that must reach the barrier
                            before any of them can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for the first phase of the barrier.
        self.count_threads2 = self.num_threads # Counter for the second phase of the barrier.
        self.counter_lock = Lock()               # Lock to protect access to the counters.
        self.threads_sem1 = Semaphore(0)         # Semaphore for the first phase, initially blocking all threads.
        self.threads_sem2 = Semaphore(0)         # Semaphore for the second phase, initially blocking all threads.

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this point.

        This method orchestrates the two-phase synchronization, ensuring all threads
        are released only after all have arrived at the barrier.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief First phase of the barrier synchronization.

        Threads acquire the `counter_lock` to decrement `count_threads1`. The last thread
        to reach this phase (when `count_threads1` becomes 0) releases all semaphores
        for the first phase, allowing all waiting threads to proceed.
        """
        # Block Logic: Ensures atomic decrement of the counter for the first phase.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Pre-condition: If this is the last thread to reach the barrier in phase 1.
            if self.count_threads1 == 0:
                # Functional Utility: Releases all `num_threads` from the `threads_sem1` semaphore.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Resets the counter for the next use.

        # Functional Utility: Blocks until released by the last thread in phase 1.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Second phase of the barrier synchronization.

        Threads acquire the `counter_lock` to decrement `count_threads2`. The last thread
        to reach this phase (when `count_threads2` becomes 0) releases all semaphores
        for the second phase, allowing all waiting threads to proceed. This two-phase
        approach prevents issues where a fast thread might re-enter the barrier before
        all slow threads have left it from the previous cycle.
        """
        # Block Logic: Ensures atomic decrement of the counter for the second phase.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Pre-condition: If this is the last thread to reach the barrier in phase 2.
            if self.count_threads2 == 0:
                # Functional Utility: Releases all `num_threads` from the `threads_sem2` semaphore.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Resets the counter for the next use.

        # Functional Utility: Blocks until released by the last thread in phase 2.
        self.threads_sem2.acquire()
