"""
@5c4ec176-5588-497f-9381-e5f463b97d34/device.py
@brief Implements a simulated device for a distributed sensor network, with sequential script execution and a custom reusable semaphore-based barrier.
This module defines a `Device` that processes sensor data and executes scripts.
It features a `DeviceThread` for operational logic and uses a `ReusableBarrierSem`
for global time-step synchronization. Scripts are processed sequentially by the `DeviceThread`.
Data access to `sensor_data` is not explicitly locked, which could lead to race conditions.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    @brief Implements a reusable barrier for synchronizing a fixed number of threads using a Lock and Semaphores.
    This barrier ensures that all participating threads wait at a synchronization point
    until every thread has reached it, after which all are released simultaneously.
    It uses a two-phase semaphore approach for reusability.
    """
    

    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that will participate in this barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for the first phase of the barrier.
        self.count_threads2 = self.num_threads # Counter for the second phase of the barrier.
        self.counter_lock = Lock()               # Lock to protect access to thread counters.
        self.threads_sem1 = Semaphore(0)         # Semaphore for releasing threads from the first phase.
        self.threads_sem2 = Semaphore(0)         # Semaphore for releasing threads from the second phase.

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have completed both phases of the barrier.
        Orchestrates the two-phase synchronization process.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief The first phase of the barrier synchronization.
        Threads decrement a counter; the last thread to decrement releases all others for this phase.
        Invariant: All threads are held at `threads_sem1.acquire()` until `count_threads1` reaches zero.
        """
        # Block Logic: Atomically decrements the counter for phase 1.
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Block Logic: The last thread to reach this point releases all waiting threads from phase 1.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Inline: Resets the counter for phase 1 for subsequent uses.
                self.count_threads1 = self.num_threads

        # Block Logic: Threads wait here until released by the last thread of phase 1.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief The second phase of the barrier synchronization, allowing for reuse.
        Threads decrement a second counter; the last thread to decrement releases all others.
        Invariant: All threads are held at `threads_sem2.acquire()` until `count_threads2` reaches zero.
        """
        # Block Logic: Atomically decrements the counter for phase 2.
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Block Logic: The last thread to reach this point releases all waiting threads from phase 2.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Inline: Resets the counter for phase 2 for subsequent uses.
                self.count_threads2 = self.num_threads

        # Block Logic: Threads wait here until released by the last thread of phase 2.
        self.threads_sem2.acquire()


class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread and a shared `ReusableBarrierSem`.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing the device's local sensor readings.
        @param supervisor: The supervisor object responsible for managing the overall simulation.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        
        self.scripts = [] # List to store assigned scripts.
        self.timepoint_done = Event() # Event to signal completion of a timepoint's processing.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.barrier = None # Shared barrier for global time step synchronization.
        self.devices = None # Will store a reference to all devices in the simulation.


    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared `ReusableBarrierSem` for synchronization among all devices.
        If not already set, it initializes the global barrier and distributes it to all devices.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        # Block Logic: Stores a reference to all devices.
        self.devices = devices
        # Block Logic: Initializes the shared barrier if this is the first device (device_id == 0) and distributes it.
        # Invariant: A single `ReusableBarrierSem` instance is created and shared across all devices.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            for dev in devices:
                dev.barrier = self.barrier
                

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        If a script is provided, it's added to the queue.
        If no script (i.e., `None`) is provided, it signals that the timepoint is done.
        @param script: The script object to assign.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            
        else:
            # Block Logic: Signals completion of the timepoint if no script is assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        Note: This method does not acquire any locks, which could lead to race conditions
        if `sensor_data` is modified concurrently by another thread without external synchronization.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
        Note: This method does not acquire any locks, which could lead to race conditions
        if `sensor_data` is modified concurrently by another thread without external synchronization.
        @param location: The key for the sensor data to be modified.
        @param data: The new data value to store.
        Precondition: `location` must be a valid key in `self.sensor_data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread, waiting for its graceful completion.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a `Device` instance.
    This thread manages the device's operational cycle, including fetching neighbor data,
    executing scripts sequentially, and coordinating with other device threads using
    a shared `ReusableBarrierSem`.
    Time Complexity: O(T * S * (N * D_access + D_script_run)) where T is the number of timepoints,
    S is the number of scripts per device, N is the number of neighbors, D_access is data access
    time, and D_script_run is script execution time.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a `DeviceThread` instance.
        @param device: The `Device` instance that this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main loop for the device's operational thread.
        Block Logic:
        1. Continuously fetches neighbor information from the supervisor.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        2. Waits for the `timepoint_done` event to be set, indicating that scripts are ready to be processed.
        3. Clears the `timepoint_done` event for the next cycle.
        4. Processes each assigned script: it collects data from neighbors and itself,
           runs the script, and then updates data on neighbors and itself.
           Invariant: Each script retrieves data from neighbors and itself, executes, and updates data.
        5. Synchronizes with all other device threads using a shared `ReusableBarrierSem`.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits until the device's timepoint is marked as done (e.g., all scripts assigned).
            self.device.timepoint_done.wait()


            # Block Logic: Clears the `timepoint_done` event for the next timepoint cycle.
            self.device.timepoint_done.clear()

            # Block Logic: Processes each script assigned to the device for the current timepoint.
            # Invariant: Each script retrieves data from neighbors and itself, executes, and updates data.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Block Logic: Collects data from neighboring devices for the specified location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collects data from its own device for the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if any data was collected and propagates the result.
                if script_data != []:
                    
                    result = script.run(script_data)

                    # Block Logic: Updates neighboring devices with the script's result.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Updates its own device's data with the script's result.
                    self.device.set_data(location, result)

            # Block Logic: Synchronizes with other device threads using a shared barrier,
            # ensuring all devices complete their processing before proceeding.
            self.device.barrier.wait()
