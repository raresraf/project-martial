


"""
@file device.py
@brief Implements a simulated distributed device with multithreaded task execution and barrier synchronization.

This module defines classes for a distributed system:
- `Device`: Represents an individual processing unit managing sensor data, scripts, and synchronization.
- `MyThread`: A worker thread responsible for executing a single script on sensor data.
- `DeviceThread`: Orchestrates the overall lifecycle of a device, including synchronization, script execution, and worker management.
- `ReusableBarrierSem`: A reusable barrier implementation using semaphores for synchronizing multiple threads in phases.

Functional Utility: Provides a framework for simulating complex parallel and distributed computing scenarios,
handling task distribution, data access synchronization, and multi-phase execution.
"""

from threading import Event, Thread, Lock, Semaphore

class Device(object):
    """
    @class Device
    @brief Represents a simulated device in a distributed system.

    Functional Utility: Manages its unique identifier, local sensor data, and a reference
    to a supervisor entity. It orchestrates script reception and execution,
    and synchronizes its state with other devices through events and a barrier.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id: A unique integer identifier for the device.
        @param sensor_data: A dictionary containing the sensor data managed by this device.
        @param supervisor: A reference to the supervisor object that manages this device.
        Functional Utility: Sets up the device's identity, its data, and establishes
        communication channels (Events) for coordinating with other components.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a script has been received and assigned to the device.
        self.script_received = Event()
        # List to store assigned scripts and their locations.
        self.scripts = []
        # Event to signal that processing for a timepoint is complete.
        self.timepoint_done = Event()
        # Dictionary to store Locks for each data location, ensuring synchronized access.
        self.lock = {}
        # Reference to a shared barrier for synchronizing with other devices.
        self.barrier = None
        # List of all devices in the system, including itself.
        self.devices = []
        # The main thread responsible for this device's operations.
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        Functional Utility: Provides a human-readable identifier for the device.
        @return: A string in the format "Device %d" % self.device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared resources and synchronization mechanisms for a collection of devices.
        @param devices: A list of Device objects, including this one, in the system.
        Functional Utility: Initializes a reusable barrier for global synchronization,
        creates a lock for each unique data location across all devices, and
        distributes these shared resources (barrier and locks) to all participating devices.
        """
        # Functional Utility: Stores the list of all devices.
        self.devices = devices
        # Functional Utility: Initializes a reusable barrier for synchronization among all devices.
        self.barrier = ReusableBarrierSem(len(self.devices))

        # Block Logic: Create a Lock for each sensor data location on this device.
        for location in self.sensor_data:
            self.lock[location] = Lock()
        # Block Logic: Create a Lock for each sensor data location across all devices.
        # This ensures that all devices share the same set of locks for data consistency.
        for device in devices:
            for location in device.sensor_data:
                self.lock[location] = Lock()

        # Block Logic: Distribute the shared barrier and locks to all devices.
        for i in xrange(len(self.devices)):
            self.devices[i].barrier = self.barrier
            self.devices[i].lock = self.lock

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed on a specific data location.
        @param script: The script object to be executed. If None, it acts as a signal to complete script assignment for the timepoint.
        @param location: The data location (key in sensor_data) where the script should operate.
        Functional Utility: Adds a script and its target location to the device's queue
        for later processing. If `script` is None, it sets the `script_received` and
        `timepoint_done` events, indicating completion of script assignment for the current timepoint.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Functional Utility: Signal that script assignment is complete for the timepoint.
            self.script_received.set()
            # Functional Utility: Signal that processing for this timepoint is done (used for final barrier wait).
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The key corresponding to the desired sensor data.
        @return: The sensor data at the specified location, or None if the location is not found.
        Functional Utility: Provides read access to the device's local sensor data store.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.
        @param location: The key corresponding to the sensor data to be updated.
        @param data: The new data value to set.
        Functional Utility: Provides write access to the device's local sensor data store.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's associated thread.
        Functional Utility: Ensures proper termination and cleanup of the DeviceThread.
        """
        self.thread.join()


class MyThread(Thread):
    """
    @class MyThread
    @brief A worker thread responsible for executing a single script on sensor data.

    Functional Utility: This thread fetches relevant sensor data from its own device
    and its neighbors for a specific location, executes a provided script with this data,
    and then updates the sensor data on the respective devices. It uses locks to
    ensure exclusive access to data locations during script execution.
    """

    def __init__(self, my_id, device, neighbours, lock, script, location):
        """
        @brief Initializes a new MyThread worker.
        @param my_id: A unique identifier for this thread.
        @param device: The parent Device object this thread belongs to.
        @param neighbours: A list of neighboring Device objects.
        @param lock: A dictionary of Locks for various data locations.
        @param script: The script object to be executed.
        @param location: The data location (key in sensor_data) where the script should operate.
        Functional Utility: Sets up the worker thread with all necessary context and
        references to perform its task of script execution and data update.
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
        @brief The main execution logic for the MyThread worker.
        Functional Utility: Acquires a lock for the target data location, collects
        sensor data from the current device and its neighbors, executes the script,
        and then updates the sensor data on all relevant devices. Ensures data
        consistency through locking.
        """
        # Block Logic: Acquire lock for the specific data location to ensure exclusive access.
        # Pre-condition: `self.location` is a valid key in `self.lock`.
        with self.lock[self.location]:
            script_data = []
            # Block Logic: Collect sensor data from neighboring devices for script execution.
            for device in self.neighbours:
                data = device.get_data(self.location)
                if data is not None:
                    script_data.append(data)

            # Functional Utility: Collect sensor data from the current device.
            data = self.device.get_data(self.location)
            if data is not None:
                script_data.append(data)

            # Block Logic: If script_data is available, execute the script and update data.
            if script_data != []:
                # Functional Utility: Execute the script with the collected data.
                result = self.script.run(script_data)

                # Block Logic: Update sensor data on neighboring devices with the script's result.
                for device in self.neighbours:
                    device.set_data(self.location, result)

                # Functional Utility: Update sensor data on the current device with the script's result.
                self.device.set_data(self.location, result)

    def shutdown(self):
        """
        @brief Waits for the thread to complete its execution.
        Functional Utility: Ensures the graceful termination of the worker thread.
        """
        self.join()


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief Manages the lifecycle and task execution for a single Device.

    Functional Utility: Orchestrates the device's main loop, including
    synchronization with the supervisor, distribution of assigned scripts
    to a pool of `MyThread` workers, and global barrier synchronization
    for timepoint progression. It handles dynamic management of worker threads.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread.
        @param device: The Device object that this thread manages.
        Functional Utility: Configures the thread with its associated device
        and initializes variables for managing worker threads.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)

        self.device = device
        self.numThreads = 0
        self.listThreads = []

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        Functional Utility: Coordinates the device's operations through different phases:
        (1) periodically checks for shutdown signal from the supervisor,
        (2) waits for scripts to be assigned,
        (3) dispatches scripts to a limited pool of worker threads,
        (4) waits for all worker threads to complete their tasks for the current timepoint,
        and (5) synchronizes with a global barrier before proceeding to the next timepoint.
        It manages a dynamic pool of worker threads to avoid overwhelming the system.
        """
        while True:
            # Functional Utility: Get information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Block Logic: Check if the supervisor has signaled a shutdown.
            # Pre-condition: `neighbours` will be None if shutdown is initiated.
            if neighbours is None:
                break

            # Block Logic: Wait for scripts to be assigned by the supervisor for the current timepoint.
            # Invariant: The device will not proceed until `script_received` event is set.
            self.device.script_received.wait()

            
            
            for (script, location) in self.device.scripts:
                if len(self.listThreads) < 8: # If there are fewer than 8 threads currently managed.
                    # Functional Utility: Create and start a new worker thread.
                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.append(thread)
                    thread.start()
                    self.numThreads += 1 # Increment total count of threads ever created (not necessarily active).
                else:
                    index = -1
                    # Block Logic: Search for an inactive worker thread to reuse its slot.
                    # Pre-condition: `len(self.listThreads)` is 8.
                    # Invariant: If an inactive thread is found, `index` will store its position, and the loop breaks.
                    for i in xrange(len(self.listThreads)):
                        if not self.listThreads[i].is_alive():
                            self.listThreads[i].join() # Functional Utility: Ensure the old thread is fully terminated.
                            index = i
                            break # Inline: Exit after finding the first inactive thread for reuse.

                    # Block Logic: Handle thread replacement.
                    # If `index` remains -1 after the loop, it implies all 8 threads were still busy.
                    # In Python, `list.remove(list[index])` with `index = -1` removes the last element.
                    # `list.insert(index, thread)` with `index = -1` inserts before the last element.
                    # Thus, if all 8 threads are still running, the *last* currently running thread
                    # is effectively removed and a new one is inserted just before the last position.
                    # This behavior potentially interrupts a live thread, which may be an unintended side-effect
                    # or a design choice for forceful task reassignment.
                    self.listThreads.remove(self.listThreads[index]) # Functional Utility: Remove the identified (or last) thread from the list.

                    # Functional Utility: Create a new worker thread and insert it into the list at the chosen position.
                    thread = MyThread(self.numThreads, self.device, neighbours, self.device.lock, script, location)
                    self.listThreads.insert(index,thread) # Functional Utility: Insert at the found index to maintain structure.
                    self.listThreads[index].start()
                    self.numThreads += 1

            # Block Logic: Wait for all currently dispatched worker threads to complete.
            # Invariant: All tasks for the current timepoint are finished before proceeding.
            for i in xrange(len(self.listThreads)):


                self.listThreads[i].join()

            # Block Logic: Wait for the timepoint to be explicitly marked as done.
            # Invariant: This confirms all scripts for the timepoint have been assigned and processed.
            self.device.timepoint_done.wait()
            
            # Functional Utility: Clear events for the next timepoint.
            self.device.script_received.clear()
            self.device.timepoint_done.clear()
            
            # Block Logic: Synchronize all device threads at the barrier before starting a new timepoint.
            # Invariant: All devices in the system must reach this barrier before any can proceed to the next timepoint.
            self.device.barrier.wait()



class ReusableBarrierSem():



    """



    @class ReusableBarrierSem



    @brief Implements a reusable barrier for synchronizing a fixed number of threads in two phases.







    Functional Utility: Ensures that all participating threads wait at a synchronization point



    until every thread has arrived. Once all threads have passed, the barrier resets and



    can be reused for subsequent synchronization points. This implementation uses two semaphores



    to manage the two phases of the barrier.



    """







    def __init__(self, num_threads):



        """



        @brief Initializes a new ReusableBarrierSem.



        @param num_threads: The total number of threads that must reach the barrier.



        Functional Utility: Sets up the internal counters and semaphores required



        for managing thread synchronization across two phases.



        """



        self.num_threads = num_threads



        # Counter for the first phase of the barrier.



        self.count_threads1 = self.num_threads



        # Counter for the second phase of the barrier.



        self.count_threads2 = self.num_threads



        # Lock to protect access to the counters.               



        self.counter_lock = Lock()               



        # Semaphore for the first synchronization point (threads wait here).         



        self.threads_sem1 = Semaphore(0)         



        # Semaphore for the second synchronization point (used to release threads from phase 1).         



        self.threads_sem2 = Semaphore(0)         







    def wait(self):



        """



        @brief Blocks the calling thread until all `num_threads` threads have reached this point.



        Functional Utility: Orchestrates the two-phase synchronization mechanism, ensuring



        all threads complete phase 1 before any can enter phase 2, and all complete phase 2



        before the barrier resets.



        """



        self.phase1()



        self.phase2()







    def phase1(self):



        """



        @brief First phase of the barrier synchronization.



        Functional Utility: Threads decrement a counter, and the last thread to arrive



        releases all waiting threads from `threads_sem1`, allowing them to proceed to phase 2.



        """



        # Block Logic: Atomically decrement the counter and check if this is the last thread.



        with self.counter_lock:



            self.count_threads1 -= 1



            # Block Logic: If this is the last thread to arrive, release all waiting threads.



            if self.count_threads1 == 0:



                for i in range(self.num_threads):



                    self.threads_sem1.release() # Functional Utility: Release all threads from the first semaphore.



                self.count_threads1 = self.num_threads # Functional Utility: Reset the counter for reuse.







        self.threads_sem1.acquire() # Functional Utility: Threads block here until released by the last thread.







    def phase2(self):



        """



        @brief Second phase of the barrier synchronization.



        Functional Utility: Similar to phase 1, but uses `threads_sem2` to ensure all



        threads have cleared the first phase before resetting the second phase's counter.



        """



        # Block Logic: Atomically decrement the counter and check if this is the last thread.



        with self.counter_lock:



            self.count_threads2 -= 1



            # Block Logic: If this is the last thread to arrive, release all waiting threads.



            if self.count_threads2 == 0:



                for i in range(self.num_threads):



                    self.threads_sem2.release() # Functional Utility: Release all threads from the second semaphore.



                self.count_threads2 = self.num_threads # Functional Utility: Reset the counter for reuse.







        self.threads_sem2.acquire() # Functional Utility: Threads block here until released by the last thread.
