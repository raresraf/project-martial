

"""
This module implements a multi-threaded device simulation environment.
It includes a reusable barrier for synchronization (`ReusableBarrierSem`),
a worker thread for script execution (`MyThread`), and classes representing
`Device` entities and their main execution threads (`DeviceThread`).
Shared locks are used to protect data access for specific locations.

Domain: Concurrency, Multi-threading, Distributed Systems, Sensor Networks.
"""

from threading import Event, Semaphore, Lock, Thread

class ReusableBarrierSem(object):
    """
    @brief Implements a reusable barrier for thread synchronization using semaphores.

    This barrier allows a fixed number of threads to wait until all have reached a certain point
    before any are allowed to proceed. It uses a two-phase approach to ensure reusability
    without deadlocks.
    """
    

    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrierSem with a specified number of threads.

        @param num_threads: The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # @brief Counter for the first phase of the barrier.
        self.count_threads1 = self.num_threads
        # @brief Counter for the second phase of the barrier.
        self.count_threads2 = self.num_threads


        # @brief Lock to protect access to the thread counters.
        self.counter_lock = Lock()               
        # @brief Semaphore for the first phase of waiting threads.
        self.threads_sem1 = Semaphore(0)         
        # @brief Semaphore for the second phase of waiting threads.
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """
        @brief Blocks until all participating threads have reached this point.

        This method orchestrates the two phases of the barrier to ensure all threads
        synchronize before proceeding, allowing for reusability.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief First phase of the barrier synchronization.

        Threads decrement a counter. The last thread to reach zero releases all
        waiting threads in this phase.
        Invariant: All threads must pass through this phase before any can proceed to phase 2.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Block Logic: Release all threads waiting in phase 1.
                for _ in range(self.num_threads):


                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Inline: Reset counter for next cycle.

        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Second phase of the barrier synchronization.

        Similar to phase 1, threads decrement a counter, and the last thread releases
        all waiting threads for this phase, effectively resetting the barrier for reuse.
        Invariant: All threads must pass through phase 1 and this phase for full synchronization.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Block Logic: Release all threads waiting in phase 2.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Inline: Reset counter for next cycle.

        self.threads_sem2.acquire()


class MyThread(Thread):
    """
    @brief A worker thread that executes a script and updates shared data.

    This thread is responsible for running a given script with its data,
    and then applying the result to the device's and its neighbors' sensor data.
    It uses a location-specific lock for data consistency during modification.
    """
    

    def __init__(self, device, location, neighbours, script):
        """
        @brief Initializes a new MyThread instance.

        @param device: The Device object this thread is associated with.
        @param location: The data location (index) this thread will operate on.
        @param neighbours: A list of neighboring Device objects.
        @param script: The script object to be executed.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.neighbours = neighbours
        self.script = script

    def run(self):
        """
        @brief Executes the script, applies the result, and manages data access.

        This method acquires a lock for the specific data location (`self.device.locks[self.location]`),
        collects data from the device and its neighbors, executes the script,
        and then updates their sensor data with the script's result. Finally, it releases
        the location-specific lock.
        """
        # Block Logic: Acquire a lock specific to the data location to ensure exclusive access.
        self.device.locks[self.location].acquire()

        script_data = [] # @brief List to store collected sensor data for the script.
        
        # Block Logic: Collect data from neighboring devices for the current location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Collect data from the current device for the current location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Block Logic: Execute the script with the collected data.
            result = self.script.run(script_data)

            # Block Logic: Disseminate the script's result to neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Block Logic: Update the current device's data with the script's result.
            self.device.set_data(self.location, result)

        self.device.locks[self.location].release() # @brief Release the lock for the data location.

def get_locations(devices):
    """
    @brief Determines the maximum location ID across all devices in the network.

    Iterates through all devices and their sensor data to find the highest
    location key, then returns one greater than that to represent the total
    number of unique locations.

    @param devices: A list of all Device objects in the network.
    @return The total number of unique locations (max_location_id + 1).
    """
    no_loc = 0 # @brief Variable to track the maximum location ID found.

    # Block Logic: Iterate through all devices and their sensor data to find the maximum location ID.
    for i in xrange(len(devices)):
        # Inline: Get the maximum location key from the current device's sensor data.
        maxx = int(max(devices[i].sensor_data.keys()))
        if maxx > no_loc:
            no_loc = maxx
    return no_loc + 1 # @brief Return the total number of unique locations.

class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.

    This class manages sensor data, stores assigned scripts, and coordinates
    synchronization and data access using shared locks (per location) and
    a shared barrier for timepoint processing.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's sensor readings.
                            Keys are locations, values are data.
        @param supervisor: A reference to the supervisor object that manages the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # @brief Event to signal that a script has been received (used in DeviceThread's run logic for an initial wait).
        self.script_received = Event()
        # @brief List to store assigned scripts and their locations.
        self.scripts = []
        # @brief Event to signal that processing for a timepoint is done (used for synchronization).
        self.timepoint_done = Event()
        # @brief The thread responsible for running the device's main logic.
        self.thread = DeviceThread(self)
        
        # @brief List of Lock objects, one for each location, shared across all devices.
        self.locks = []
        
        # @brief Synchronization barrier for coordinating timepoints across devices.
        self.barrier = None
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up and distributes the shared synchronization primitives (barrier and locks).

        Only the device with `device_id == 0` (or minimum ID) initializes the
        `ReusableBarrierSem` and the list of `Lock` objects. These are then
        distributed to all other devices in the network.

        @param devices: A list of all Device objects in the network.
        """
        # Block Logic: Only Device 0 (minimum ID) initializes the shared barrier and locks.
        if self.device_id == 0:
            # Block Logic: Initialize a new ReusableBarrierSem and distribute it to all devices.
            barrier = ReusableBarrierSem(len(devices))
            for i in xrange(len(devices)):
                devices[i].barrier = barrier

            # Block Logic: Determine the total number of locations across all devices.
            no_loc = get_locations(devices)
            # Block Logic: Create a Lock for each unique location.
            for i in xrange(no_loc):
                lock = Lock()              
                self.locks.append(lock)    

            # Block Logic: Distribute the created locks to all devices.
            for i in xrange(no_loc):
                for j in xrange(len(devices)):
                    devices[j].locks.append(self.locks[i])

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script is provided, it's added to the list of scripts.
        If no script is provided (None), it signals that script assignment for the
        current timepoint is complete by setting `script_received` and `timepoint_done` events.

        @param script: The script object to be executed, or None to signal completion.
        @param location: The data location (index) the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set() # Inline: Signal that scripts have been assigned.
            self.timepoint_done.set() # Inline: Signal that the timepoint's script assignment is done.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        Note: Locks for data access are managed by `MyThread` in this implementation,
              using the shared `self.locks` list.

        @param location: The data location (index) to retrieve data from.
        @return The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.

        Note: Locks for data access are managed by `MyThread` in this implementation,
              using the shared `self.locks` list.

        @param location: The data location (index) to set data for.
        @param data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's main thread, ensuring proper termination.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main execution thread for a Device.

    This thread manages the device's operational lifecycle, including fetching
    neighbor information, coordinating script execution using worker threads (`MyThread`),
    and synchronizing with other devices using barriers and events.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device: The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the device thread.

        This loop continuously performs the following actions:
        1. Fetches current neighbor information from the supervisor.
        2. Terminates if no neighbors are returned (simulation end).
        3. Waits on `device.script_received` event, indicating scripts are assigned for the timepoint.
        4. Creates and starts `MyThread` instances for each assigned script.
        5. Waits on `device.barrier` to synchronize with all other devices.
        6. Joins all `MyThread` instances to ensure their completion.
        7. Waits on `device.timepoint_done` event (signaling end of script assignment for this timepoint)
           and clears it for the next cycle.
        Invariant: The device processes data in discrete timepoints, synchronizing with the network
                   after each timepoint.
        """
        while True:
            # Block Logic: Fetch updated neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Inline: Terminate thread if no more neighbors (simulation end).

            my_threads = [] # @brief List to hold MyThread worker instances for the current timepoint.

            # Block Logic: Wait until scripts for the current timepoint have been assigned.
            self.device.script_received.wait()
            # Inline: Clear the event for the next timepoint.
            self.device.script_received.clear()

            # Block Logic: Create and start MyThread instances for each assigned script.
            for (script, location) in self.device.scripts:
                # Inline: Create a new MyThread for the script and add it to the list.
                my_threads.append(MyThread(self.device, location, neighbours, script))
                my_threads[-1].start() # Inline: Start the newly created MyThread.

            # Block Logic: Wait on the barrier to synchronize with all other devices after starting worker threads.
            self.device.barrier.wait()

            # Block Logic: Wait for all MyThread instances to complete their execution.
            for i in xrange(len(my_threads)):
                my_threads[i].join()

            # Block Logic: Wait for the `timepoint_done` event to be set (signaling end of script assignment).
            # Then clear the event for the next timepoint.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
