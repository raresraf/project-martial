


"""
This module defines a distributed device simulation framework, including
classes for Device, DeviceThread, InnerThread, and a ReusableBarrierSem.
It manages device-specific data, script execution, inter-device communication,
and synchronization using threads and semaphores.
"""

from threading import Event, Thread, Semaphore, Lock
from Queue import Queue

class Device(object):
    """
    Represents a single device or processing unit within a distributed system.
    Each device manages its own sensor data, executes scripts, and interacts
    with other devices through a supervisor. It utilizes threads and
    synchronization primitives for concurrent operations and timepoint management.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): Initial sensor data for the device,
                                mapping locations to data values.
            supervisor (Supervisor): The supervisor managing this device
                                     and its interactions with others.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        # Event to signal that the current timepoint's tasks are completed.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()
        # Barrier for synchronizing all devices at the end of each timepoint.
        self.time_point_barrier = None
        # Dictionary to hold semaphores for each data location, ensuring
        # exclusive access during script execution.
        self.location_semaphore_dict = None

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the device with a list of other devices in the system
        and establishes shared synchronization primitives (barrier and semaphores).

        This method is designed to be called once by a "master" device (device_id 0)
        to initialize shared resources for all participating devices.

        Args:
            devices (list): A list of Device objects representing
                            all devices in the system.
        """
        # Block Logic: This block is executed only by the device with device_id 0,
        # acting as a coordinator to set up shared resources for all devices.
        if self.device_id == 0:
            # Initializes a reusable barrier for synchronizing all devices at timepoints.
            barrier = ReusableBarrierSem(len(devices))
            self.time_point_barrier = barrier
            # Invariant: All devices in the system will be assigned the same
            # time_point_barrier for synchronization.
            for dev in devices:
                if dev.time_point_barrier is None:
                    dev.time_point_barrier = barrier
            
            location_set = set()
            # Block Logic: Gathers all unique data locations across all devices.
            for dev in devices:
                for location in dev.sensor_data:
                    location_set.add(location)
            
            loc_dict = {}
            # Invariant: A semaphore is created for each unique data location
            # to control concurrent access to that location's data.
            for loc in location_set:
                loc_dict[loc] = Semaphore(1) # Initialize semaphore with a count of 1 for mutual exclusion.
            
            # Invariant: All devices in the system will be assigned the same
            # dictionary of location-specific semaphores.
            for dev in devices:
                dev.location_semaphore_dict = loc_dict

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by the device at a specific data location.

        Args:
            script (Script): The script object to be executed.
                             If None, it signifies the end of script assignments
                             for the current timepoint.
            location (int): The identifier for the data location
                            the script operates on.
        """
        # Block Logic: If a script is provided, it's added to the list of scripts
        # to be processed in the current timepoint.
        if script is not None:
            self.scripts.append((script, location))
        # Block Logic: If no script is provided, it signals that all scripts for
        # the current timepoint have been assigned.
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location.

        Args:
            location (int): The identifier of the data location.

        Returns:
            Any: The sensor data at the given location, or None if the
                 location does not exist in the device's sensor_data.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location.

        Args:
            location (int): The identifier of the data location to update.
            data (Any): The new data value to set for the location.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device,
        waiting for its background thread to complete.
        """
        self.thread.join()

class InnerThread(Thread):
    """
    A worker thread managed by DeviceThread, responsible for executing
    individual scripts from a shared queue. It handles data retrieval
    from the device and its neighbors, script execution, and result
    propagation, ensuring thread-safe access to data locations.
    """
    def __init__(self, device, barrier, queue):
        """
        Initializes a new InnerThread instance.

        Args:
            device (Device): The parent Device object.
            barrier (ReusableBarrierSem): A barrier for synchronizing with
                                          other InnerThread instances.
            queue (Queue): A queue from which to receive script tasks.
        """
        Thread.__init__(self)
        self.device = device
        self.barrier = barrier
        self.queue = queue
        self.neighbours = []

    def run(self):
        """
        The main execution loop for the InnerThread.

        Continuously retrieves tasks from the script queue, processes them
        (executes scripts, updates neighbors), and handles special commands
        like "exit" and "done" for synchronization and termination.
        """
        while True:
            # Block Logic: Retrieves a script task from the queue.
            # This call blocks until an item is available.
            script = self.queue.get()
            # Block Logic: Handles the "exit" command, signaling thread termination.
            if script[0] == "exit":
                self.barrier.wait() # Synchronizes with other threads before exiting.
                break
            # Block Logic: Handles the "done" command, signaling completion
            # of a set of scripts for a timepoint.
            if script[0] == "done":
                self.barrier.wait() # Synchronizes with other threads.
                continue
            # Block Logic: Handles the "neighbours" command, updating the
            # thread's knowledge of neighboring devices.
            if script[0] == "neighbours":
                self.neighbours = script[1]
                self.barrier.wait() # Synchronizes with other threads.
                continue
            
            script_improved = script[0]
            location = script[1]
            script_data = []
            
            # Block Logic: Acquires a semaphore for the specific data location
            # to ensure exclusive access during data retrieval and update.
            self.device.location_semaphore_dict[location].acquire()
            # Block Logic: Gathers sensor data from all known neighboring devices
            # for the specified location.
            for device in self.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Block Logic: Gathers sensor data from the local device for the specified location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Precondition: There is collected data for the script to process.
            # Invariant: If data is present, the script is run and its results are
            # propagated to the devices.
            if script_data != []:
                # Block Logic: Executes the script with the collected data.
                result = script_improved.run(script_data)
                
                # Block Logic: Updates the sensor data on all neighboring devices
                # with the result of the script execution.
                for device in self.neighbours:
                    device.set_data(location, result)
                
                # Block Logic: Updates the sensor data on the local device
                # with the result of the script execution.
                self.device.set_data(location, result)
            # Block Logic: Releases the semaphore for the data location,
            # allowing other threads to access it.
            self.device.location_semaphore_dict[location].release()

class DeviceThread(Thread):
    """
    The main thread for a Device object, responsible for orchestrating
    script execution by dispatching tasks to a pool of `InnerThread` workers.
    It manages timepoint synchronization, neighbor discovery, and gracefully
    handles device shutdown.
    """
    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device object that this thread will manage.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.cores_number = 8 # Number of worker threads to spawn for script execution.
        self.threads_barrier = ReusableBarrierSem(self.cores_number) # Barrier for synchronizing InnerThreads.
        self.scripts_queue = Queue() # Queue for dispatching scripts to InnerThreads.
        self.thread_list = [] # List to hold references to spawned InnerThread instances.

    def run(self):
        """
        The main execution loop for the DeviceThread.

        This method initializes and manages a pool of `InnerThread` workers.
        It continuously monitors for new timepoints, retrieves neighbor information,
        dispatches scripts to the worker threads, and handles synchronization
        across all devices and worker threads.
        """
        # Block Logic: Spawns a pool of InnerThread workers.
        for _ in range(self.cores_number):
            inner_t = InnerThread(self.device, self.threads_barrier,\
            self.scripts_queue)
            self.thread_list.append(inner_t)
        # Block Logic: Starts all spawned InnerThread workers.
        for thread in self.thread_list:
            thread.start()
        
        # Block Logic: Main loop for the device thread, continuously processing
        # timepoints until a shutdown signal (None neighbors) is received.
        while True:
            # Precondition: The supervisor provides the current neighbors for the device.
            # Invariant: If neighbors are None, it signals the worker threads to terminate.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                # Block Logic: Signals all InnerThreads to exit and then breaks the loop.
                for thread in self.thread_list:
                    self.scripts_queue.put(("exit", None))
                break
            
            # Block Logic: Waits for the current timepoint to be marked as done,
            # indicating all scripts have been assigned.
            self.device.timepoint_done.wait()
            
            # Block Logic: Dispatches the current neighbor list to all InnerThreads.
            for _ in range(self.cores_number):
                self.scripts_queue.put(("neighbours", neighbours))
            
            # Block Logic: Dispatches all assigned scripts for the current timepoint
            # to the InnerThread workers via the shared queue.
            for pair in self.device.scripts:
                self.scripts_queue.put(pair)
            
            # Block Logic: Signals all InnerThreads that all scripts for the timepoint
            # have been dispatched.
            for _ in range(self.cores_number):
                self.scripts_queue.put(("done", None))
            
            # Block Logic: Resets the timepoint_done event for the next timepoint.
            self.device.timepoint_done.clear()
            
            # Block Logic: Waits at the global barrier to synchronize with all other devices
            # before proceeding to the next timepoint.
            self.device.time_point_barrier.wait()

        # Block Logic: Joins all InnerThread workers to ensure they complete
        # their execution before the DeviceThread terminates.
        for thread in self.thread_list:
            thread.join()

class ReusableBarrierSem(object):
    """
    Implements a reusable barrier using semaphores and a lock.

    This barrier allows a fixed number of threads to wait for each other
    at a synchronization point and then proceed together, and can be
    reused multiple times.
    """
    def __init__(self, num_threads):
        """
        Initializes a new ReusableBarrierSem instance.

        Args:
            num_threads (int): The number of threads that must reach the
                                barrier before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for the first phase of the barrier.
        self.count_threads2 = self.num_threads # Counter for the second phase of the barrier.
        self.counter_lock = Lock() # Lock to protect access to the counters.
        self.threads_sem1 = Semaphore(0) # Semaphore for the first phase.
        self.threads_sem2 = Semaphore(0) # Semaphore for the second phase.

    def wait(self):
        """
        Main entry point for threads to wait at the barrier.

        A thread calling this method will block until `num_threads`
        threads have also called `wait()`, after which all waiting
        threads are released.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        The first synchronization phase of the barrier.

        Threads decrement a counter. The last thread to reach zero
        releases all waiting threads for this phase.
        """
        # Block Logic: Critical section protected by counter_lock to safely
        # decrement the thread counter for phase 1.
        with self.counter_lock:
            self.count_threads1 -= 1
            # Pre-condition: This is the last thread to reach the barrier in phase 1.
            # Invariant: All threads waiting on threads_sem1 will be released.
            if self.count_threads1 == 0:
                # Block Logic: Releases all semaphores to unblock threads waiting
                # in this phase.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                # Resets the counter for the next use of the barrier.
                self.count_threads1 = self.num_threads

        # Block Logic: Each thread waits on its semaphore, ensuring it
        # proceeds only after all threads have reached this point.
        self.threads_sem1.acquire()

    def phase2(self):
        """
        The second synchronization phase of the barrier.

        This phase is identical in logic to phase1 but uses separate counters
        and semaphores to allow for barrier reuse. Threads decrement a counter,
        and the last thread to reach zero releases all waiting threads for this phase.
        """
        # Block Logic: Critical section protected by counter_lock to safely
        # decrement the thread counter for phase 2.
        with self.counter_lock:
            self.count_threads2 -= 1
            # Pre-condition: This is the last thread to reach the barrier in phase 2.
            # Invariant: All threads waiting on threads_sem2 will be released.
            if self.count_threads2 == 0:
                # Block Logic: Releases all semaphores to unblock threads waiting
                # in this phase.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                # Resets the counter for the next use of the barrier.
                self.count_threads2 = self.num_threads

        # Block Logic: Each thread waits on its semaphore, ensuring it
        # proceeds only after all threads have reached this point.
        self.threads_sem2.acquire()
