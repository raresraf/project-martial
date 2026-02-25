"""
@fdb6e52b-fa23-4104-a8a0-2530cb0c229b/device.py
@brief Implements a simulated device for a distributed sensor network,
       incorporating a custom thread pool for script execution and synchronization.

This module defines the architecture for individual devices in a sensor
network. Each device manages its sensor data, receives and executes scripts
via a dedicated thread pool, and collaborates with a supervisor and neighboring
devices. It leverages threading primitives like Queues, Events, Threads, and
Locks for concurrent script processing and a reusable barrier for global
synchronization across devices.
"""


from threading import Event, Thread, Lock

from barrier import ReusableBarrierCond
from threadpool import ThreadPool

class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.

    Each device has a unique ID, stores sensor data, executes assigned scripts
    via a thread pool, and collaborates with other devices for synchronized
    data processing. It manages its own script queue and communicates its
    state to a supervisor.
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

        # List to store (location, Lock) pairs, later converted to a dictionary.
        self.locations_locks = []

        # Block Logic: Initializes a Lock for each sensor data location.
        for location in sensor_data:
            self.locations_locks.append((location, Lock()))

        # Convert the list of (location, Lock) pairs into a dictionary for easy access.
        self.locations_locks = dict(self.locations_locks)

        # Reference to the shared barrier for device synchronization.
        self.barrier = None

        # The main processing thread for this device.
        self.thread = DeviceThread(self)
        self.thread.start() # Start the main device processing thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device %d".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the shared synchronization barrier for a group of devices.

        This method ensures that all devices share the same `ReusableBarrierCond`
        instance. The barrier is initialized by the device with `device_id == 0`.

        @param devices: A list of all Device objects in the network.
        """
        # Block Logic: Only device with ID 0 initializes the shared ReusableBarrierCond.
        # Invariant: All devices in the 'devices' list will be assigned the same barrier.
        if self.device_id == 0:
            self.barrier = ReusableBarrierCond(len(devices))
            # Block Logic: Assign the initialized barrier to all other devices.
            for device in devices:
                if device.device_id != 0:
                    device.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device for execution at a specific location.

        If a script is provided, it's added to the device's script queue. An event
        is set to notify the device's processing thread that new scripts are available.
        If `script` is None, it signals the end of the current timepoint for script assignment.

        @param script: The script object to be executed, or None.
        @param location: The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set() # Signal the DeviceThread that new scripts arrived.
        else:
            self.timepoint_done.set() # Signal that no more scripts are coming for this timepoint.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location, acquiring a lock first.

        This method acquires a lock for the specified location before accessing
        the sensor data, ensuring thread-safe read operations.

        @param location: The location for which to retrieve data.
        @return: The sensor data at the specified location, or None if not found.
        """
        if location in self.sensor_data:
            self.locations_locks[location].acquire() # Acquire lock for the specific location.
            return self.sensor_data[location]

        return None


    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location and releases its lock.

        This method updates the sensor data and then releases the lock for the
        specified location, completing the thread-safe write operation.

        @param location: The location at which to set the data.
        @param data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
            self.locations_locks[location].release() # Release lock after updating data.


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
        self.thread_pool = ThreadPool(8, device) # Creates a thread pool with 8 worker threads.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        It continuously orchestrates script processing: it waits for the supervisor
        to provide neighbors, dispatches scripts to the thread pool, waits for
        all scripts to complete for a timepoint, and then synchronizes with
        other device threads using a global barrier. The loop terminates if the
        supervisor signals no more neighbors (end of simulation).
        """
        # Block Logic: Main loop for continuous operation of the DeviceThread.
        while True:
            # Get the current list of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # Pre-condition: If supervisor returns None for neighbors, it indicates simulation end.
            if neighbours is None:
                break # Exit the main device thread loop if no more neighbors.

            # Block Logic: Inner loop to handle script assignment and dispatching for the current timepoint.
            while True:
                # Block Logic: Waits until the timepoint is done AND no more scripts are pending arrival.
                # Invariant: If timepoint_done is set AND script_received is NOT set, it means all scripts
                #            for this timepoint have arrived and been processed.
                if self.device.timepoint_done.wait() and not self.device.script_received.is_set():
                    self.device.timepoint_done.clear() # Clear timepoint_done for the next cycle.
                    self.device.script_received.set() # Set script_received to allow the outer if to clear it.
                    break # Exit the inner loop to proceed with joining threads and barrier.

                # Block Logic: Processes newly received scripts.
                # Invariant: 'script_received' is cleared after processing the scripts.
                if self.device.script_received.is_set():
                    self.device.script_received.clear() # Clear the event after processing.

                    # Block Logic: Iterate through assigned scripts and submit them to the thread pool.
                    for (script, location) in self.device.scripts:
                        self.thread_pool.submit_task(script, location, neighbours)


            # Wait for all tasks in the thread pool's queue to complete.
            self.thread_pool.tasks_queue.join()

            # Synchronization Point: Wait for all devices to finish their timepoint processing.
            self.device.barrier.wait()

        # Block Logic: Signal all thread pool workers to terminate and wait for their completion.
        self.thread_pool.join_threads()


from threading import Thread
from Queue import Queue

class ThreadPool(object):
    """
    @brief A custom thread pool implementation for executing tasks (scripts).

    This class manages a fixed number of worker threads that pull tasks
    from a shared queue. It's designed to execute scripts for a Device
    in a concurrent manner, collecting data from neighbors and updating
    sensor data.
    """
    

    def __init__(self, number_threads, device):
        """
        @brief Initializes the thread pool with a specified number of worker threads.

        @param number_threads: The number of worker threads to create in the pool.
        @param device: A reference to the Device object associated with this thread pool.
        """
        self.number_threads = number_threads
        self.device_threads = [] # List to store worker thread objects.
        self.device = device     # Reference to the Device object.
        self.tasks_queue = Queue(number_threads) # Queue to hold tasks (script, location, neighbours).


        # Block Logic: Create and append worker threads to the pool.
        for _ in xrange(0, number_threads): # xrange is Python 2, use range in Python 3.
            thread = Thread(target=self.apply_scripts)
            self.device_threads.append(thread)

        # Block Logic: Start all worker threads.
        for thread in self.device_threads:
            thread.start()

    def apply_scripts(self):
        """
        @brief The main loop for each worker thread in the thread pool.

        Each worker continuously pulls tasks from the tasks queue. A task
        consists of a script, a location, and a list of neighbors. The worker
        collects data, runs the script, updates relevant sensor data, and
        then marks the task as done. It handles a special termination signal.
        """
        while True:
            script, location, neighbours = self.tasks_queue.get()

            # Block Logic: Check for the termination signal.
            # Pre-condition: (None, None, None) is the termination signal.
            if neighbours is None and script is None:
                self.tasks_queue.task_done() # Mark task as done for proper queue joining.
                return # Exit the worker thread.

            script_data = [] # List to hold data collected for the script.
            
            # Block Logic: Collect data from neighboring devices, excluding the local device.
            for device in neighbours:
                if device.device_id != self.device.device_id:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
            
            # Block Logic: Collect data from the local device.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # Block Logic: If any script data was collected, execute the script and update data.
            if script_data != []:
                
                result = script.run(script_data) # Execute the script with collected data.

                # Block Logic: Update sensor data for all neighboring devices.
                for device in neighbours:
                    if device.device_id != self.device.device_id:
                        device.set_data(location, result)
                
                # Update sensor data for the local device.
                self.device.set_data(location, result)


            self.tasks_queue.task_done() # Mark task as done.


    def submit_task(self, script, location, neighbours):
        """
        @brief Submits a new task (script, location, neighbours) to the thread pool's queue.

        @param script: The script object to be executed.
        @param location: The data location relevant to the script.
        @param neighbours: A list of neighboring Device objects.
        """
        self.tasks_queue.put((script, location, neighbours))


    def join_threads(self):
        """
        @brief Waits for all pending tasks in the queue to be processed,
               sends termination signals to all worker threads, and waits
               for the worker threads to complete.
        """
        self.tasks_queue.join() # Wait until all tasks in the queue are processed.

        # Block Logic: Send termination signals to all worker threads.
        for _ in xrange(0, len(self.device_threads)): # xrange is Python 2, use range in Python 3.
            self.submit_task(None, None, None) # Use submit_task to add termination signal.

        # Block Logic: Wait for all worker threads to actually terminate.
        for thread in self.device_threads:
            thread.join()
