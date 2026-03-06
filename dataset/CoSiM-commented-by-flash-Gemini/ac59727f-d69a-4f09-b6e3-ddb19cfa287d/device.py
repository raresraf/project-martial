

"""
This module implements a device simulation framework that utilizes multiple threads
and a reusable barrier for synchronization. It defines:
- ReusableBarrierSem: A re-usable barrier mechanism using semaphores.
- Device: Represents a simulated device with sensors, scripts, and multi-threaded processing.
- DeviceThread: The master thread for a Device, orchestrating job assignment and synchronization.
- Worker: A worker thread responsible for processing assigned scripts.
"""


from threading import Event, Thread, Lock, Semaphore

class ReusableBarrierSem(object):
    """
    Implements a reusable barrier synchronization mechanism using semaphores and a lock.
    This barrier allows a fixed number of threads to wait until all have reached a certain point
    before any are allowed to proceed, and can then be reset for subsequent synchronizations.
    """
    

    def __init__(self, num_threads):
        """
        Initializes the reusable barrier.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads


        self.counter_lock = Lock()
        self.threads_sem1 = Semaphore(0)
        self.threads_sem2 = Semaphore(0)

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have reached this barrier.
        Uses a two-phase approach to allow reusability.
        """
        
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        First phase of the barrier. Threads decrement a counter and the last thread
        releases all waiting threads for this phase.
        """
        
        with self.counter_lock: # Block Logic: Ensure atomic access to the counter.
            self.count_threads1 -= 1
            # Block Logic: If this is the last thread in phase 1, release all waiting threads.
            if self.count_threads1 == 0:
                for _ in range(self.num_threads): # Block Logic: Release each waiting thread.
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset counter for next use

        self.threads_sem1.acquire() # Block Logic: Wait for all threads to reach this point.

    def phase2(self):
        """
        Second phase of the barrier. Threads decrement a counter and the last thread
        releases all waiting threads for this phase.
        """
        
        with self.counter_lock: # Block Logic: Ensure atomic access to the counter.
            self.count_threads2 -= 1
            # Block Logic: If this is the last thread in phase 2, release all waiting threads.
            if self.count_threads2 == 0:
                for _ in range(self.num_threads): # Block Logic: Release each waiting thread.
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset counter for next use

        self.threads_sem2.acquire() # Block Logic: Wait for all threads to reach this point.

class Device(object):
    """
    Represents a simulated device in a distributed system. Each device
    manages its sensor data, executes scripts, and interacts with a supervisor.
    It utilizes a master thread and multiple worker threads for concurrent operations.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.lock = {} # Dictionary to store locks for different locations/devices

        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when a new script is assigned
        self.scripts = []             # List to store assigned scripts (script, location) tuples
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's processing
        self.setup_done = Event()      # Event to signal when initial device setup is complete
        self.terminate = Event()       # Event to signal termination of threads
        self.neighbours = []           # List to store neighboring devices

        self.barrier = None # Global barrier for device synchronization
        self.threads_barrier = ReusableBarrierSem(9) # Barrier for master and worker threads
        # Initialize and start the master DeviceThread
        self.master = DeviceThread(self, self.terminate, self.barrier, self.threads_barrier, \
                                    self.setup_done)
        self.master.start()

        self.threads = [] # List to hold Worker thread instances

        # Block Logic: Create and start Worker instances for this device.
        # Invariant: Each device will have 8 worker threads.
        for _ in range(8):
            thread = Worker(self.master, self.terminate, self.threads_barrier)

            self.threads.append(thread)
            thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs initial setup for all devices in the simulation.
        This includes initializing the global barrier and location locks,
        and propagating these synchronization objects to other devices.
        This method is typically called by a supervisor or a designated master device (device_id == 0).

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        # Block Logic: Device with ID 0 acts as the master to set up global synchronization.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices)) # Initialize global barrier for all devices
            # Block Logic: Initialize a lock for each device (used for data access).
            for dev in devices:
                self.lock[dev] = Lock()
            # Block Logic: Propagate the initialized global barrier and locks to all other devices.
            for dev in devices:
                if dev.device_id != 0:
                    dev.barrier = self.barrier
                    dev.lock = self.lock
                    dev.setup_done.set() # Signal that setup is complete for other devices

            self.setup_done.set() # Signal that setup is complete for the master device

    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific location.
        If a script is provided, it's added to the device's script list.
        If no script is provided (None), it signals that a timepoint is done.

        Args:
            script (Script or None): The script object to assign, or None to signal timepoint completion.
            location (str): The location associated with the script.
        """
        

        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list
            self.script_received.set()              # Signal that a new script is available
        else:
            self.timepoint_done.set()               # Signal that processing for the current timepoint is complete

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            Any or None: The sensor data if available for the location, otherwise None.
        """
        

        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        Args:
            location (str): The location for which to set data.
            data (Any): The new sensor data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Shuts down the device by signaling termination to all worker threads
        and the master thread, then waiting for them to complete.
        """
        

        self.terminate.set() # Signal all threads to terminate
        # Block Logic: Signal each worker thread to stop and wait for it to join.
        for i in range(8):
            self.threads[i].script_received.set() # Wake up worker if it's waiting for scripts
            self.threads[i].join() # Wait for worker thread to finish
        self.master.join() # Wait for the master thread to finish


class DeviceThread(Thread):
    """
    The master thread for a Device, responsible for orchestrating the overall
    simulation workflow, including synchronization with other devices, fetching
    neighbor information, and coordinating script execution by worker threads.
    """
    

    def __init__(self, device, terminate, barrier, threads_barrier, setup_done):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent Device instance this thread belongs to.
            terminate (Event): An event to signal the thread to terminate.
            barrier (ReusableBarrierSem): The global barrier for device-level synchronization.
            threads_barrier (ReusableBarrierSem): A local barrier for synchronizing master and worker threads.
            setup_done (Event): An event to signal when initial setup is complete.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.neighbours = []      # List to store neighboring devices
        self.terminate = terminate # Event to signal termination
        self.barrier = barrier    # Global device barrier
        self.threads_barrier = threads_barrier # Local master/worker barrier
        self.setup_done = setup_done # Setup completion event

    def run(self):
        """
        Executes the main logic of the DeviceThread.
        - Waits for initial device setup to complete.
        - Synchronizes with other devices using the global barrier.
        - Continuously fetches neighbor information from the supervisor.
        - If no neighbors are returned (signal for shutdown), the loop breaks.
        - Waits for timepoint processing to be signaled and clears it.
        - Distributes assigned scripts among worker threads in a round-robin fashion.
        - Synchronizes master and worker threads.
        """

        # Block Logic: Wait for the initial device setup to be marked as complete.
        self.setup_done.wait()
        # Block Logic: Participate in the global barrier, ensuring all devices are set up.
        self.device.barrier.wait()

        # Block Logic: Main loop for continuous processing of timepoints.
        while True:
            # Block Logic: Synchronize all devices at the global barrier.
            self.device.barrier.wait()

            # Block Logic: Fetch updated neighbor information from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: If no neighbors are returned, it's a shutdown signal.
            if self.neighbours is None:
                break # Exit the main loop

            # Block Logic: Wait for scripts to be assigned for the current timepoint, then clear the event.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()
            # Block Logic: Synchronize all devices at the global barrier before script distribution.
            self.device.barrier.wait()

            # Block Logic: Distribute assigned scripts to worker threads in a round-robin manner.
            scripts = []
            for i in range(8): # Assuming 8 worker threads
                scripts.append([])

            for i in range(len(self.device.scripts)):
                scripts[i%8].append(self.device.scripts[i])

            # Block Logic: Assign the subsets of scripts to each worker and signal them to process.
            for i in range(8):
                self.device.threads[i].scripts = scripts[i]
                self.device.threads[i].script_received.set() # Signal worker that scripts are available

            # Block Logic: Synchronize the master thread with its worker threads.
            # Pre-condition: Only proceed if the terminate event is not set.
            if not self.terminate.is_set():
                self.threads_barrier.wait() # Master waits for all workers to finish processing their scripts.


class Worker(Thread):
    """
    A worker thread for a Device, responsible for processing a subset of assigned scripts.
    It fetches scripts, retrieves data from relevant devices, executes the script,
    and updates data on neighboring devices and its own device.
    """
    

    def __init__(self, master, terminate, barrier):
        """
        Initializes a Worker thread.

        Args:
            master (DeviceThread): The master thread of the parent Device.
            terminate (Event): An event to signal the thread to terminate.
            barrier (ReusableBarrierSem): A barrier for synchronizing with the master thread.
        """

        Thread.__init__(self)
        self.master = master                # Reference to the master DeviceThread
        self.script_received = Event()      # Event to signal when new scripts are assigned to this worker
        self.terminate = terminate          # Event to signal thread termination
        self.scripts = []                   # List to store scripts assigned to this worker
        self.barrier = barrier              # Barrier for synchronization

    @staticmethod
    def append_data(device, location, script_data):
        """
        Appends sensor data from a device at a specific location to a list.
        Ensures thread-safe access to device data using a lock.

        Args:
            device (Device): The device from which to retrieve data.
            location (str): The location of the data.
            script_data (list): The list to which the retrieved data will be appended.
        """
        
        device.lock[device].acquire() # Block Logic: Acquire the device-specific lock.
        data = device.get_data(location)
        device.lock[device].release() # Block Logic: Release the device-specific lock.
        if data is not None:
            script_data.append(data)

    @staticmethod
    def set_data(device, location, result):
        """
        Sets sensor data on a device at a specific location.
        Ensures thread-safe access to device data using a lock.

        Args:
            device (Device): The device on which to set data.
            location (str): The location of the data.
            result (Any): The new data to set.
        """
        
        device.lock[device].acquire() # Block Logic: Acquire the device-specific lock.
        device.set_data(location, result)
        device.lock[device].release() # Block Logic: Release the device-specific lock.

    def run(self):
        """
        The main execution loop for the worker thread.
        It continuously waits for scripts to be assigned, processes them,
        and then synchronizes with the master thread.
        The worker terminates if the `terminate` event is set.
        """

        # Block Logic: Main loop for continuous processing of scripts.
        while True:
            # Block Logic: Wait for the master thread to signal that scripts are available.
            self.script_received.wait()
            self.script_received.clear() # Reset the event for the next cycle

            # Block Logic: Check if a termination signal has been received.
            if self.terminate.is_set():
                break # Exit the worker loop

            # Block Logic: Process the assigned scripts if any exist.
            if self.scripts is not None:
                # Invariant: Each (script, location) tuple represents a task to be executed.
                for (script, location) in self.scripts:

                    # Block Logic: Prepare a list to collect data for the current script.
                    script_data = []
                    # Block Logic: Collect data from neighboring devices, if available.
                    if self.master.neighbours is not None:
                        # Invariant: Each `device` is a neighbor from which data should be collected.
                        for device in self.master.neighbours:
                            self.append_data(device, location, script_data)


                    # Block Logic: Collect data from the current device itself.
                    self.append_data(self.master.device, location, script_data)

                    # Block Logic: If data was collected, execute the script and update devices.
                    if script_data != []:

                        # Inline: Execute the assigned script with collected data.
                        result = script.run(script_data)

                        # Block Logic: Propagate the script's result to neighboring devices.
                        if self.master.neighbours is not None:
                            # Invariant: Each `device` is a neighbor that needs to be updated.
                            for device in self.master.neighbours:
                                self.set_data(device, location, result)
                        
                        # Block Logic: Update the current device's sensor data with the script's result.
                        self.set_data(self.master.device, location, result)

            # Block Logic: Synchronize with the master thread after processing scripts for the current timepoint.
            self.barrier.wait()
