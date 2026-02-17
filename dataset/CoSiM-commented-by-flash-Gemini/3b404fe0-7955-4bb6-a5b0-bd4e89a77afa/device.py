

"""
@3b404fe0-7955-4bb6-a5b0-bd4e89a77afa/device.py
@brief Implements a simulated device for a distributed sensor network,
including its thread of execution and a custom reusable barrier for synchronization.
This module models device behavior where devices process sensor data,
interact with neighbors, and execute scripts, with a focus on synchronized timepoints.
"""

from threading import *


class Device(object):
    """
    @brief Represents a single device in the distributed system simulation.
    Manages its sensor data, assigned scripts, and coordinates its operation
    through a dedicated thread and a shared barrier mechanism.
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
        self.script_received = Event()
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Provides a string representation of the device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the reusable barrier for synchronization across all devices.
        This method ensures that only the first device initializes the barrier,
        which is then implicitly shared among all `DeviceThread` instances.
        @param devices: A list of all Device instances in the simulation.
        Precondition: This method is called once during system setup.
        """
        self.devices=devices
        # Block Logic: Initializes a shared barrier if this is the first device, ensuring one barrier per group.
        # Invariant: A single `MyReusableBarrier` instance is created and implicitly shared for all devices.
        if self==devices[0]:
            self.bar = MyReusableBarrier(len(devices))
        
        pass

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific data `location`.
        If a script is provided, it's added to the queue, and a signal is set.
        If no script is provided (i.e., `None`), it signals that the timepoint is done.
        @param script: The script object to be executed, or `None` to signal completion.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Block Logic: Signals completion of a timepoint if no script is assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key identifying the sensor data.
        @return The data associated with the location, or `None` if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location.
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
    executing scripts, and synchronizing with other device threads using a barrier.
    Time Complexity: O(T * S * (N + D)) where T is the number of timepoints, S is the number of scripts per device,
    N is the number of neighbors, and D is the data retrieval/setting operations.
    """

    def __init__(self, device):
        """
        @brief Initializes a `DeviceThread` instance.
        @param device: The `Device` instance that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main loop for the device's operational thread.
        Block Logic:
        1. Continuously retrieves neighbor information from the supervisor.
           Invariant: The loop terminates if `neighbours` is `None`, signaling the end of the simulation.
        2. Waits for a signal that the current timepoint is ready for processing.
           Precondition: `self.device.timepoint_done` is set by the supervisor or `assign_script`.
        3. Clears the `timepoint_done` signal for the next cycle.
        4. Iterates through assigned scripts, collecting data from neighbors and its own device,
           executing the script, and propagating the results.
           Invariant: For each script, all necessary data is gathered, the script is run,
           and results are disseminated to relevant devices.
        5. Synchronizes with all other device threads using a shared barrier before proceeding.
           Invariant: All active `DeviceThread` instances must reach this barrier before any can
           progress to the next timepoint, ensuring synchronized advancement of the simulation.
        """
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break

            # Block Logic: Waits for the `timepoint_done` event to be set, indicating that
            # all scripts for the current timepoint have been assigned or processed.
            self.device.timepoint_done.wait()
            # Block Logic: Resets the `timepoint_done` event for the next timepoint.
            self.device.timepoint_done.clear()

            # Block Logic: Processes each script assigned to the device for the current timepoint.
            # Invariant: Each script retrieves data from neighbors and itself, executes, and updates data.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Block Logic: Collects data from neighboring devices for the specified location.
                # Note: This implementation does not use explicit locks for neighbor data access,
                # which might lead to race conditions if not handled by higher-level mechanisms.
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

            # Block Logic: Waits on a shared barrier, ensuring all device threads complete their
            # processing for the current timepoint before any proceeds to the next.
            self.device.devices[0].bar.wait()


class MyReusableBarrier():
    """
    @brief Implements a custom reusable barrier for synchronizing multiple threads.
    This barrier uses two phases of semaphores and a lock to ensure all threads
    reach a synchronization point before any are allowed to proceed, and then
    resets itself for subsequent synchronizations.
    """
    
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that will participate in this barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for the first phase of the barrier.
        self.count_threads2 = self.num_threads # Counter for the second phase of the barrier.
        
        self.counter_lock = Lock()       # Lock to protect access to thread counters.
        self.threads_sem1 = Semaphore(0) # Semaphore for releasing threads from the first phase.
        self.threads_sem2 = Semaphore(0) # Semaphore for releasing threads from the second phase.

    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this point.
        This method orchestrates the two-phase synchronization.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief The first phase of the barrier synchronization.
        Threads decrement a counter and the last thread releases all others.
        Invariant: All threads are held at `threads_sem1.acquire()` until `count_threads1` reaches zero.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Block Logic: The last thread to reach this point releases all waiting threads from phase 1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            # Inline: Reset count for phase 2 immediately after the last thread has passed phase 1 check.
            self.count_threads2 = self.num_threads
         
        # Block Logic: Threads wait here until the last thread releases them from phase 1.
        self.threads_sem1.acquire()
         
    def phase2(self):
        """
        @brief The second phase of the barrier synchronization, allowing for reuse.
        Threads decrement a second counter and the last thread releases all others.
        Invariant: All threads are held at `threads_sem2.acquire()` until `count_threads2` reaches zero.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Block Logic: The last thread to reach this point releases all waiting threads from phase 2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            # Inline: Reset count for phase 1 immediately after the last thread has passed phase 2 check.
            self.count_threads1 = self.num_threads
         
        # Block Logic: Threads wait here until the last thread releases them from phase 2.
        self.threads_sem2.acquire()

