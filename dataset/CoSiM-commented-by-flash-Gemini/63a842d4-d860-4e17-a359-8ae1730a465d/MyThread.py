"""
@63a842d4-d860-4e17-a359-8ae1730a465d/MyThread.py
@brief Implements a device simulation for a sensor network, incorporating a
       custom thread pool for script execution and synchronization mechanisms.

This module defines the architecture for individual devices in a sensor
network. Each device manages its sensor data, receives and executes scripts
via a dedicated thread pool (`MyThread` class), and collaborates with a
supervisor and neighboring devices. It leverages threading primitives like
Queues, Events, Threads, and Locks for concurrent script processing and
a Barrier for global synchronization across devices.
"""


from Queue import Queue # Imported from Python 2 Queue module, in Python 3 it's queue.Queue.
from threading import Thread, Event, Lock # Import threading components.

class MyThread(object):
    """
    @brief Implements a custom thread pool for executing tasks.

    This class manages a fixed number of worker threads that pull tasks
    from a shared queue. It's designed to execute scripts for a Device
    in a concurrent manner.
    """
    

    def __init__(self, threads_count):
        """
        @brief Initializes the thread pool with a specified number of worker threads.

        @param threads_count: The number of worker threads to create in the pool.
        """

        self.queue = Queue(threads_count)

        self.threads = [] # List to store worker thread objects.
        self.device = None # Reference to the Device object, set later.

        # Block Logic: Create and append worker threads to the pool.
        for _ in xrange(threads_count): # xrange is Python 2, use range in Python 3.
            new_thread = Thread(target=self.execute)
            self.threads.append(new_thread)

        # Block Logic: Start all worker threads.
        for thread in self.threads:
            thread.start()

    def execute(self):
        """
        @brief The main loop for each worker thread in the pool.

        Each worker continuously pulls tasks from the queue, executes the script,
        and marks the task as done. It handles a special termination signal (None, None, None).
        """
        

        # Block Logic: Worker thread's infinite loop for processing tasks.
        while True:

            neighbours, script, location = self.queue.get()

            # Block Logic: Check for the termination signal.
            if neighbours is None: # Termination signal (None, None, None)
                if script is None:
                    self.queue.task_done() # Mark task as done for proper queue joining.
                    return # Exit the worker thread.


            self.run_script(neighbours, script, location) # Execute the script.
            self.queue.task_done() # Mark task as done.

    def run_script(self, neighbours, script, location):
        """
        @brief Executes a given script, collects data from neighbors and the local device,
               and updates sensor data based on the script's result.

        @param neighbours: A list of neighboring Device objects.
        @param script: The script object to execute.
        @param location: The data location pertinent to this script.
        """
        

        script_data = [] # List to hold data collected for the script.

        # Block Logic: Collect data from neighboring devices.
        # Invariant: Data is collected from all neighbors, excluding the local device itself.
        for device in neighbours:
            if device.device_id != self.device.device_id:
                data = device.get_data(location)
                if data is None:
                    continue

                script_data.append(data)

        # Block Logic: Collect data from the local device.
        data = self.device.get_data(location)
        if data is not None:
            script_data.append(data)

        # Block Logic: If any script data was collected, execute the script and update data.
        if script_data:
            
            result = script.run(script_data) # Execute the script with collected data.

            # Block Logic: Update sensor data for all neighboring devices.
            for device in neighbours:
                if device.device_id == self.device.device_id:
                    continue

                device.set_data(location, result)

            # Block Logic: Update sensor data for the local device.
            self.device.set_data(location, result)

    def end_threads(self):
        """
        @brief Signals all worker threads to terminate and waits for their completion.
        """
        

        self.queue.join() # Wait until all tasks in the queue are processed.

        # Block Logic: Send termination signals to all worker threads.
        for _ in xrange(len(self.threads)): # xrange is Python 2, use range in Python 3.
            self.queue.put((None, None, None))

        # Block Logic: Wait for all worker threads to actually terminate.
        for thread in self.threads:
            thread.join()




from threading import Event, Thread, Lock

from barrier import Barrier # Assuming barrier.py contains a Barrier class.
# from MyThread import MyThread # Already in the same file.


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

        self.new_adds() # Initialize additional device attributes.

        self.thread.start() # Start the main device processing thread.

    def new_adds(self):
        """
        @brief Initializes additional attributes for the device, including
               a barrier and location-specific locks.
        """
        self.barrier = None # Reference to the shared barrier for device synchronization.
        # Dictionary to store locks for specific data locations.
        self.locations = {}
        # Block Logic: Initialize a lock for each sensor data location.
        for location in self.sensor_data:
            self.locations[location] = Lock()
        # Flag to indicate if a script has arrived.
        self.script_arrived = False

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared synchronization barrier for a group of devices.

        This method ensures that all devices share the same `Barrier` instance.
        The barrier is initialized by the device with `device_id == 0`.

        @param devices: A list of all Device objects in the network.
        """
        # Block Logic: Only device with ID 0 initializes the shared Barrier.
        # Invariant: All devices in the 'devices' list will be assigned the same barrier.
        if self.device_id == 0:
            self.barrier = Barrier(len(devices))
            # Block Logic: Assign the initialized barrier to all other devices.
            for dev in devices:
                if dev.device_id == 0:
                    continue

                dev.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.

        If a script is provided, it's added to the device's script queue. An event
        is set to notify the device's processing thread that new scripts are available.
        If `script` is None, it signals the end of the current timepoint for script assignment.

        @param script: The script object to be executed, or None.
        @param location: The data location relevant to the script.
        """

        self.set_boolean(script) # Update the script_arrived flag.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal the DeviceThread that new scripts arrived.
        else:
            self.timepoint_done.set() # Signal that no more scripts are coming for this timepoint.

    def set_boolean(self, script):
        """
        @brief Sets a boolean flag indicating whether a script has arrived.

        @param script: The script object. If not None, `script_arrived` is set to True.
        """
        if script is not None:
            self.script_arrived = True


    def acquire_location(self, location):
        """
        @brief Acquires a lock for a specific data location.

        This ensures exclusive access to the data at `location` on this device.

        @param location: The data location for which to acquire the lock.
        """
        if location in self.sensor_data:
            self.locations[location].acquire()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location, acquiring a lock first.

        @param location: The location for which to retrieve data.
        @return: The sensor data at the specified location, or None if not found.
        """
        self.acquire_location(location) # Acquire lock before accessing data.
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location and releases its lock.

        @param location: The location at which to set the data.
        @param data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locations[location].release() # Release lock after updating data.

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its processing thread to complete.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief The main processing thread for a Device.

    This thread manages script execution for its associated device. It coordinates
    with other devices using a barrier, fetches neighbors' data, and dispatches
    scripts to a thread pool for processing.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread with a reference to its parent device.

        @param device: The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

        # Initialize a thread pool for script execution.
        self.thread_pool = MyThread(8) # Creates a pool with 8 worker threads.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        It continuously orchestrates script processing: it waits for the supervisor
        to provide neighbors, dispatches scripts to the thread pool, waits for
        all scripts to complete for a timepoint, and then synchronizes with
        other device threads using a global barrier. The loop terminates if the
        supervisor signals no more neighbors (end of simulation).
        """

        self.thread_pool.device = self.device # Assign the device to the thread pool workers.

        # Block Logic: Main loop for continuous operation of the DeviceThread.
        while True:

            # Get the current list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If supervisor returns None for neighbors, it indicates simulation end.
            if neighbours is None:
                break # Exit the main device thread loop if no more neighbors.

            # Block Logic: Waits for a timepoint to be done or a script to arrive.
            # Invariant: The loop ensures that either a new script is available for processing
            #            or the timepoint has been explicitly marked as done.
            while True:

                # Block Logic: Checks if a script has arrived or if the timepoint is done.
                # The 'wait()' on timepoint_done will block until it's set.
                if self.device.script_arrived or self.device.timepoint_done.wait():
                    if self.device.script_arrived:
                        self.device.script_arrived = False # Reset flag.

                        # Block Logic: Dispatch all current scripts to the thread pool.
                        for (script, location) in self.device.scripts:
                            self.thread_pool.queue.put((neighbours, script, location))
                    else:
                        self.device.timepoint_done.clear() # Clear timepoint_done for the next cycle.
                        self.device.script_arrived = True # Reset script_arrived as it was not actually processed.
                        break # Exit the inner loop to proceed with joining workers.


            # Wait for all tasks in the thread pool's queue to complete.
            self.thread_pool.queue.join()

            # Synchronization Point: Wait for all devices to finish their timepoint processing.
            self.device.barrier.wait() # This barrier is for synchronization across Devices.

        # Block Logic: Signal all thread pool workers to terminate and wait for their completion.
        self.thread_pool.end_threads()
