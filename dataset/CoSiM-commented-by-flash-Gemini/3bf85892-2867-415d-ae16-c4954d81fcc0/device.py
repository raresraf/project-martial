"""
This module implements a distributed simulation framework centered around a 'Device' concept.
It utilizes multi-threading for concurrent execution of tasks on devices,
coordinated by a 'Supervisor' (external) and a custom 'ReusableBarrier' for synchronization.
Each Device manages its sensor data, processes scripts, and interacts with neighboring devices.
"""


from threading import Event, Thread, Lock
from Queue import Queue # Note: In Python 3, Queue is renamed to queue.Queue
from reusable_barrier_condition import ReusableBarrier


class Device(object):
    """
    Represents a single device in the distributed simulation.
    Each device manages its own sensor data, receives and processes scripts,
    and coordinates its operations with other devices using a shared barrier.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        :param device_id: A unique identifier for the device.
        :param sensor_data: A dictionary containing initial sensor data for various locations on the device.
                            Format: {location_key: data_value}
        :param supervisor: A reference to the central supervisor managing all devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when a new script has been assigned.
        self.scripts = [] # List to store assigned scripts, each associated with a location.
        self.timepoint_done = Event() # Event to signal completion of processing for a timepoint.
        self.thread = DeviceThread(self) # The main thread responsible for device operation.
        self.location_locks = {} # Dictionary to store locks for specific data locations, ensuring thread-safe access.
        self.barrier = None # A shared barrier for synchronizing with other devices.
        self.num_threads = 8 # Number of worker threads this device will use for script processing.
        self.queue = Queue(self.num_threads) # Queue for distributing scripts to worker threads.
        self.thread.start() # Starts the device's main operational thread.

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Initializes shared resources (like the ReusableBarrier and shared location locks)
        across all devices in the simulation. This method is typically called once
        by the supervisor to prepare the environment for all devices.

        :param devices: A list of all Device instances participating in the simulation.
        """
        # Conditional Logic: Ensures the barrier is initialized only once.
        if self.barrier is None:
            # Functional Utility: Initializes a ReusableBarrier for synchronizing all devices.
            self.barrier = ReusableBarrier(len(devices))
            # Block Logic: Assigns the same shared barrier to all devices.
            for device in devices:
                device.barrier = self.barrier
                # Block Logic: Initializes a unique Lock for each distinct location across all devices.
                # This ensures that access to sensor data at any given location is thread-safe.
                for location in device.sensor_data:
                    if location not in self.location_locks:
                        self.location_locks[location] = Lock()
            # Block Logic: Assigns the globally consistent set of location locks to all devices.
            for device in devices:
                device.location_locks = self.location_locks


    def assign_script(self, script, location):
        """
        Assigns a script to be processed by this device at a specific sensor data location.
        If script is None, it signals that the current timepoint processing is complete.

        :param script: The script object to execute, or None to signal timepoint completion.
        :param location: The sensor data location relevant to this script.
        """
        # Conditional Logic: If a valid script is provided, add it to the device's script list.
        if script is not None:
            self.scripts.append((script, location))
        # Else Block: If script is None, it signifies the end of scripts for the current timepoint,
        # so set the 'timepoint_done' event.
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Retrieves sensor data for a specified location from this device.

        :param location: The key identifying the sensor data location.
        :return: The sensor data value, or None if the location does not exist.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Updates the sensor data for a specified location on this device.

        :param location: The key identifying the sensor data location.
        :param data: The new data value to set.
        """
        # Conditional Logic: Updates data only if the location already exists in sensor_data.
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device, primarily by joining its main thread.
        """
        self.thread.join()


class WorkerThread(Thread):
    """
    A worker thread responsible for executing scripts assigned to a Device.
    It fetches tasks from a queue, processes them, and updates sensor data,
    ensuring thread-safe access to shared data locations.
    """

    def __init__(self, queue, device):
        """
        Initializes a WorkerThread.

        :param queue: The Queue from which to get script execution tasks.
        :param device: The Device object this worker is associated with.
        """
        Thread.__init__(self)
        self.queue = queue
        self.device = device

    def run(self):
        """
        The main execution loop for the worker thread.
        It continuously fetches script execution tasks from the queue.
        Each task involves acquiring a lock for the relevant location,
        gathering data from the device and its neighbors, executing the script,
        updating the data, and releasing the lock.
        The loop terminates upon receiving a specific shutdown signal (None, None, None).
        """
        # Invariant: The worker thread continuously processes tasks from its queue.
        while True:
            # Functional Utility: Retrieves a task (script, location, neighbors) from the queue.
            # This call blocks until an item is available.
            data_tuple = self.queue.get()

            # Conditional Logic: Checks for a shutdown signal.
            # If received, the worker breaks out of its execution loop.
            if data_tuple == (None, None, None):
                break

            # Synchronization: Acquires a lock for the specific data location to ensure exclusive access
            # during script execution and data modification.
            self.device.location_locks[data_tuple[1]].acquire()
            script_data = [] # List to accumulate sensor data for the script.
            
            # Block Logic: Gathers sensor data from neighboring devices for the specified location.
            for device in data_tuple[2]: # data_tuple[2] contains a list of neighbor devices.
                data = device.get_data(data_tuple[1]) # Retrieves data from a neighbor.
                if data is not None:
                    script_data.append(data)
            
            # Functional Utility: Gathers sensor data from its own device for the specified location.
            data = self.device.get_data(data_tuple[1])
            if data is not None:
                script_data.append(data)

            # Conditional Logic: If there is any script data to process.
            if script_data != []:
                # Functional Utility: Executes the script with the gathered data.
                result = data_tuple[0].run(script_data)

                # Block Logic: Updates the sensor data on neighboring devices with the script's result.
                for device in data_tuple[2]:
                    device.set_data(data_tuple[1], result)
                
                # Functional Utility: Updates the sensor data on its own device with the script's result.
                self.device.set_data(data_tuple[1], result)
            self.device.location_locks[data_tuple[1]].release() # Synchronization: Releases the lock for the data location.



class DeviceThread(Thread):
    """
    The main thread for a Device, managing its lifecycle and coordinating worker threads.
    It handles fetching neighbor information, distributing scripts to worker threads,
    and synchronizing timepoints using a reusable barrier.
    """

    def __init__(self, device):
        """
        Initializes the DeviceThread.

        :param device: The Device object this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        The main execution loop for the DeviceThread.
        It initializes worker threads, then enters a loop to process timepoints.
        For each timepoint, it gets neighbor information, waits for scripts to be assigned,
        puts scripts into the worker queue, waits at a barrier for all devices to complete
        their processing for the current timepoint, and then clears the timepoint event.
        The loop terminates when the supervisor signals no more neighbors (end of simulation).
        """
        threads = [] # List to hold references to the worker threads.

        # Block Logic: Initializes and starts a pool of worker threads.
        for i in range(self.device.num_threads):
            thread = WorkerThread(self.device.queue, self.device)
            threads.append(thread)
            threads[i].start()

        # Invariant: The device thread continuously processes timepoints until shutdown.
        while True:
            # Functional Utility: Retrieves neighbor information from the supervisor.
            # This is critical for distributed calculations in the simulation.
            neighbours = self.device.supervisor.get_neighbours()
            # Conditional Logic: If no neighbors are returned (None), it signals the end of the simulation.
            if neighbours is None:
                break

            # Synchronization: Waits until all scripts for the current timepoint have been assigned
            # (signaled by the supervisor setting timepoint_done).
            self.device.timepoint_done.wait()

            # Block Logic: Distributes assigned scripts to the worker threads via the queue.
            # Each script is packaged with its location and neighbor information.
            for (script, location) in self.device.scripts:
                self.device.queue.put((script, location, neighbours))

            # Block Logic: Clears the scripts list after they have been queued for processing.
            self.device.scripts = [] # Clear the scripts for the next timepoint.


            # Synchronization: Waits at the barrier. This ensures all devices complete
            # their script processing for the current timepoint before proceeding.
            self.device.barrier.wait()
            self.device.timepoint_done.clear() # Resets the event for the next timepoint.

        # Block Logic: Sends shutdown signals to all worker threads.
        # This is a sentinel value that tells each worker to terminate its loop.
        for i in range(self.device.num_threads):
            self.device.queue.put((None, None, None))

        # Block Logic: Waits for all worker threads to complete their execution before the device thread exits.
        for i in range(self.device.num_threads):
            threads[i].join()


from threading import Condition

class ReusableBarrier(object):
    """
    A reusable synchronization barrier for coordinating multiple threads.
    Threads wait at the barrier until all expected threads have arrived,
    then all are released simultaneously. It can be used multiple times.
    """
    
    def __init__(self, num_threads):
        """
        Initializes a ReusableBarrier.

        :param num_threads: The total number of threads that must reach the barrier
                            before all waiting threads are released.
        """
        self.num_threads = num_threads # Total number of threads expected at the barrier.
        self.count_threads = self.num_threads # Current count of threads waiting at the barrier.
        self.cond = Condition() # The condition variable used for waiting and notifying.


    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all other
        expected threads have also reached it. Once all threads arrive,
        they are all released, and the barrier is reset for reuse.
        """
        self.cond.acquire() # Synchronization: Acquires the condition variable's lock.
        self.count_threads -= 1 # Decrements the count of threads yet to arrive.
        # Conditional Logic: Checks if this is the last thread to arrive at the barrier.
        if self.count_threads == 0:
            self.cond.notify_all() # Synchronization: Notifies all waiting threads to proceed.
            self.count_threads = self.num_threads # Resets the barrier for subsequent use.
        else:
            self.cond.wait() # Synchronization: Waits until notified by the last arriving thread.
        self.cond.release() # Synchronization: Releases the condition variable's lock.

