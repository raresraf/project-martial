"""
@file device.py
@brief Implements a device simulation with multi-threaded script execution and a reusable barrier for synchronization.

This module defines three key classes:
- `ReusableBarrier`: Provides a thread synchronization mechanism allowing a fixed number of threads to wait for each other.
- `Device`: Simulates an IoT-like device capable of processing sensor data through assigned scripts, interacting with a supervisor and neighboring devices.
- `DeviceThread`: Represents a worker thread associated with a `Device` instance, responsible for executing parts of the assigned scripts.

The system uses a two-phase reusable barrier to ensure all worker threads synchronize correctly before and after processing.
"""


from threading import Thread, Event, Lock, Semaphore

class ReusableBarrier():
    """
    Implements a reusable barrier synchronization mechanism using semaphores.
    This barrier allows a fixed number of threads (`num_threads`) to wait for each other
    at a synchronization point, and then allows them to proceed. It is "reusable"
    because it can be used multiple times without reinitialization.
    It uses a two-phase approach to prevent threads from "slipping" through the barrier
    if they arrive too early for the next cycle.
    """


    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                before any of them can proceed.
        """
        self.num_threads = num_threads
        # Counter for threads in phase 1.
        self.count_threads1 = self.num_threads
        # Counter for threads in phase 2.
        self.count_threads2 = self.num_threads
        # Lock to protect access to the thread counters.
        self.counter_lock = Lock()
        # Semaphore for threads waiting in phase 1. Initialized to 0, so all threads
        # will block until released.
        self.threads_sem1 = Semaphore(0)
        # Semaphore for threads waiting in phase 2. Initialized to 0.
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all `num_threads`
        have also called `wait()`. Once all threads have arrived, they are all released.
        This method executes both phase1 and phase2 of the barrier.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        The first phase of the barrier. Threads decrement a counter and the last thread
        to arrive releases all waiting threads for phase 1.
        
        Pre-condition: All threads from the previous cycle (if any) have passed through phase2.
        Invariant: `count_threads1` accurately reflects the number of threads yet to reach the barrier in this phase.
        """
        with self.counter_lock: # Protect access to the counter.
            self.count_threads1 -= 1
            # Conditional Logic: If this is the last thread to arrive at phase 1.
            # Functional Utility: The final thread arriving at the barrier signals all other waiting threads to proceed.
            if self.count_threads1 == 0:
                # Release all `num_threads` from the first semaphore.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                # Reset the counter for the next use of phase 1.
                self.count_threads1 = self.num_threads
        # Functional Utility: Blocks the current thread until all threads have reached and been released from phase 1.
        self.threads_sem1.acquire() 

    def phase2(self):
        """
        The second phase of the barrier. Threads decrement a counter and the last thread
        to arrive releases all waiting threads for phase 2. This phase ensures reusability.

        Pre-condition: All threads have successfully passed through phase 1 of the current cycle.
        Invariant: `count_threads2` accurately reflects the number of threads yet to reach the barrier in this phase.
        Functional Utility: This second phase is crucial for the reusability of the barrier, ensuring that
        threads don't re-enter phase1 before all threads have completed the current cycle.
        """
        with self.counter_lock: # Protect access to the counter.
            self.count_threads2 -= 1
            # Conditional Logic: If this is the last thread to arrive at phase 2.
            # Functional Utility: The final thread arriving at the second phase signals all other waiting threads to proceed, enabling reuse.
            if self.count_threads2 == 0:
                # Release all `num_threads` from the second semaphore.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                # Reset the counter for the next use of phase 2.
                self.count_threads2 = self.num_threads

        # Functional Utility: Blocks the current thread until all threads have reached and been released from phase 2.
        self.threads_sem2.acquire()


class Device(object):
    """
    Represents a simulated device in an IoT-like environment, managing sensor data,
    executing scripts, and interacting with a supervisor and neighboring devices.
    Each device operates with multiple worker threads for parallel script execution.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing sensor readings, keyed by location.
            supervisor (Supervisor): An object responsible for coordinating device activities.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal that a new script has been assigned to the device.
        self.script_received = Event()
        # Event to signal that all threads have completed processing for the current timepoint.
        self.timepoint_done = Event()
        # List to store assigned scripts and their target locations.
        self.scripts = []

        # Reusable barrier for synchronizing worker threads within this device.
        # The number 8 suggests a fixed pool of worker threads per device.
        self.barrier_worker = ReusableBarrier(8)
        # Event used during the setup phase to coordinate device initialization.
        self.setup_event = Event()
        # List of other Device instances this device can interact with.
        self.devices = []
        # Shared locks for managing access to specific sensor data locations across devices.
        self.locks = None
        # List of neighboring devices, typically retrieved from the supervisor.
        self.neighbours = []
        # Global barrier for synchronizing all devices.
        self.barrier = None
        # List of DeviceThread instances associated with this device.
        self.threads = []

        # Block Logic: Initializes and starts a fixed number of worker threads for the device.
        for i in range(8):
            self.threads.append(DeviceThread(self, i))

        # Block Logic: Starts all worker threads.
        for thr in self.threads:
            thr.start()

        # List of locks, one for each sensor data location, to manage concurrent access.
        self.location_lock = []

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the device and other devices in the system, typically called by a supervisor.
        This method is primarily handled by device_id 0 to orchestrate the setup.

        Args:
            devices (list): A list of all Device instances in the simulation.

        Pre-condition: This method is expected to be called by a coordinating entity (e.g., supervisor).
        Invariant: After execution, all devices will share a common global barrier and location locks.
        """
        
        # Conditional Logic: Only the device with ID 0 performs the global setup.
        if self.device_id == 0:
            # Functional Utility: Creates a global reusable barrier for all devices' worker threads.
            barrier = ReusableBarrier(len(devices)*8)
            self.barrier = barrier
            location_max = 0
            # Block Logic: Iterates through all devices to assign the global barrier and determine the maximum location ID.
            for device in devices:
                device.barrier = barrier
                # Block Logic: Determines the maximum sensor data location across all devices.
                for location, data in device.sensor_data.iteritems():
                    # Conditional Logic: Updates location_max if a larger location ID is found.
                    if location > location_max:
                        location_max = location
                # Functional Utility: Signals the current device to proceed with its setup.
                device.setup_event.set()
            # Functional Utility: Signals device 0 to proceed with its setup.
            self.setup_event.set()

            # Functional Utility: Initializes a list of locks for each sensor data location.
            self.location_lock = [None] * (location_max + 1)

            # Block Logic: Assigns the globally initialized location locks to all devices.
            for device in devices:
                device.location_lock = self.location_lock
                # Functional Utility: Signals the current device to proceed with its setup after receiving location locks.
                device.setup_event.set()
            # Functional Utility: Signals device 0 to complete its setup.
            self.setup_event.set()

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific sensor data location.

        Args:
            script (Script): The script object to be executed.
            location (int): The sensor data location (key) the script operates on.

        Pre-condition: The script and location are valid.
        Post-condition: The script is added to the device's script queue, and `script_received` event is set.
        """
        busy = 0
        # Conditional Logic: Processes the script if it is not None.
        if script is not None:
            # Functional Utility: Adds the script and its target location to the device's internal queue.
            self.scripts.append((script, location))
            # Conditional Logic: Checks if a lock for the current location needs to be initialized or assigned.
            if self.location_lock[location] is None:
                # Block Logic: Iterates through other devices to find an existing lock for the location.
                for device in self.devices:
                    # Conditional Logic: If another device already has a lock for this location, share it.
                    if device.location_lock[location] is not None:
                        self.location_lock[location] = device.location_lock[location]
                        busy = 1
                        break

                # Conditional Logic: If no existing lock was found, create a new one.
                if busy == 0:
                    self.location_lock[location] = Lock()
            # Functional Utility: Signals worker threads that a new script has been received.
            self.script_received.set()
        else:
            # Functional Utility: Signals that processing for the current timepoint is complete if no script is assigned.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (int): The key corresponding to the sensor data.

        Returns:
            any: The sensor data at the specified location, or None if the location does not exist.
        """
        # Conditional Logic: Checks if the location exists in the sensor data.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        Args:
            location (int): The key corresponding to the sensor data.
            data (any): The new data to set for the location.

        Pre-condition: The location must already exist in `sensor_data` to be updated.
        """
        # Conditional Logic: Updates the sensor data only if the location already exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device by joining all worker threads.
        """
        # Block Logic: Waits for all worker threads to complete their execution before shutting down.
        for thr in self.threads:
            thr.join()


class DeviceThread(Thread):
    """
    Represents a worker thread that belongs to a Device. Each thread
    is responsible for executing a subset of the assigned scripts.
    """

    def __init__(self, device, idd):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent Device instance this thread belongs to.
            idd (int): A unique identifier for this thread within its device (0-7).
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.idd = idd

    def run(self):
        """
        The main execution loop for the DeviceThread. It waits for device setup,
        synchronizes with other worker threads, retrieves neighbor information,
        and processes assigned scripts.
        """
        
        # Functional Utility: Blocks until the parent device has completed its initial setup.
        self.device.setup_event.wait()

        # Invariant: The loop continues as long as the device is not signaled for shutdown
        # (implicitly through `self.device.neighbours` becoming None).
        while True:
            
            # Conditional Logic: Only the first thread (idd == 0) retrieves neighbor information from the supervisor.
            if self.idd == 0:
                # Functional Utility: Fetches the list of neighboring devices from the supervisor.
                neighbours = self.device.supervisor.get_neighbours()
                self.device.neighbours = neighbours

            # Functional Utility: Synchronizes all 8 worker threads within the device using the device's internal barrier.
            self.device.barrier_worker.wait()

            # Conditional Logic: Breaks the loop and terminates the thread if no neighbors are found, indicating shutdown or end of simulation.
            if self.device.neighbours is None:
                break

            # Functional Utility: Waits until all worker threads signal completion of the current timepoint's processing.
            self.device.timepoint_done.wait()
            # Functional Utility: Synchronizes all 8 worker threads again after timepoint completion, before proceeding to script execution.
            self.device.barrier_worker.wait()

            i = 0
            
            # Block Logic: Iterates through the scripts assigned to the parent device.
            # Invariant: Each script is processed by a specific thread based on its index modulo 8.
            for (script, location) in self.device.scripts:
                # Conditional Logic: Ensures that each thread processes only its assigned subset of scripts.
                if i % 8 == self.idd:
                    # Functional Utility: Acquires a lock for the specific sensor data location to prevent race conditions during script execution.
                    with self.device.location_lock[location]:
                        script_data = []
                        
                        # Block Logic: Collects sensor data from neighboring devices for the current location.
                        for device in self.device.neighbours:
                            data = device.get_data(location)
                            # Conditional Logic: Adds data to script_data only if it is not None.
                            if data is not None:
                                script_data.append(data)
                        
                        # Functional Utility: Collects sensor data from the current device itself for the specified location.
                        data = self.device.get_data(location)
                        # Conditional Logic: Adds data to script_data only if it is not None.
                        if data is not None:
                            script_data.append(data)

                        # Conditional Logic: Executes the script only if relevant sensor data was collected.
                        if script_data != []:
                            
                            # Functional Utility: Executes the assigned script with the collected sensor data.
                            result = script.run(script_data)

                            
                            # Block Logic: Propagates the script's result back to all neighboring devices.
                            for device in self.device.neighbours:
                                device.set_data(location, result)
                            
                            # Functional Utility: Updates the current device's sensor data with the script's result.
                            self.device.set_data(location, result)
                i = i + 1

            
            # Functional Utility: Clears the timepoint_done event to prepare for the next timepoint.
            self.device.timepoint_done.clear()
            # Functional Utility: Synchronizes all devices globally using the shared barrier, signifying completion of script processing for the current cycle.
            self.device.barrier.wait()
