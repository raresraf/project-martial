

"""
@8e4e5154-5ebe-4d5b-8fb-19cbcfe2625b/device.py
@brief Implements a multi-threaded simulation for distributed sensor devices.

This module defines the core components for simulating a network of sensor devices,
each capable of executing scripts, managing local sensor data, and interacting
with a central supervisor. It leverages threading primitives for concurrent
operation, including events for synchronization, locks for resource protection,
queues for task distribution, semaphores for access control, and barriers for
coordinated execution across devices.

The simulation models device behavior over discrete timepoints, where devices
process scripts, update local data, and communicate with neighbors under the
guidance of a supervisor.

Classes:
- Device: Represents a single simulated sensor device.
- DeviceThread: Manages the lifecycle and operation of a Device instance in its own thread.
- WorkerThread: Executes assigned scripts for a Device, handling data access and updates.

Domain: Distributed Systems Simulation, Concurrent Programming, Sensor Networks.
"""

from threading import Event, Thread, Lock, Semaphore, Condition
from Queue import Queue
from barrier import ReusableBarrierSem

class Device(object):
    """
    @brief Represents a single simulated sensor device in a distributed network.

    Each device manages its own sensor data, interacts with a supervisor,
    and executes assigned scripts in a multi-threaded environment. It uses
    various synchronization primitives to coordinate its activities with
    other devices and its internal worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        Sets up the device's unique identifier, its initial sensor data,
        a reference to the central supervisor, and initializes various
        synchronization primitives required for multi-threaded operation.

        @param device_id: A unique integer identifier for the device.
        @param sensor_data: A dictionary containing the device's initial sensor readings.
                            Keys are location IDs, values are sensor data.
        @param supervisor: A reference to the Supervisor object managing the device network.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor    
        self.scripts = [] # Stores scripts assigned to this device for execution. Each script is (script_object, location_id).
        self.all_devices = [] # A list of all Device objects in the simulation, set during setup.

        # Synchronization primitive: Event to signal that processing for a specific timepoint is complete.
        self.event_timepoint_done = Event()

        # Synchronization primitive: Event to signal that the device setup process is complete.
        self.event_setup_done = Event()

        # Synchronization primitive: Event to signal that all threads should terminate.
        self.event_stop_threads = Event()

        # Synchronization primitive: Lock to protect access to the device's sensor data (self.sensor_data).
        self.lock_data = Lock()

        # Synchronization primitive: List of locks, one for each location, to protect concurrent access to specific sensor locations.
        self.lock_locations = []

        # Synchronization primitive: Queue to hold scripts assigned to worker threads for execution.
        self.queue_scripts = Queue()

        # Synchronization primitive: Semaphore to track the number of scripts available in the queue for worker threads.
        self.semaphore_queue = Semaphore(0)

        # Synchronization primitive: Barrier to synchronize all devices at specific points in the simulation.
        self.barrier_devices = None

        # Synchronization primitive: Condition variable for coordinating between the DeviceThread and WorkerThreads,
        # typically used to signal when the script queue is empty.
        self.condition_variable = Condition(Lock())

        self.thread = DeviceThread(self)
        self.thread.start()


    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the device's awareness of all other devices in the simulation
               and initializes shared synchronization primitives.

        This method is called once at the beginning of the simulation.
        If this is the master device (device_id == 0), it initializes a
        ReusableBarrierSem and a list of locks for locations, then propagates
        these to all other devices.

        @param devices: A list of all Device objects participating in the simulation.
        """
        self.all_devices = devices

        # Block Logic: Master device initializes and propagates shared synchronization objects.
        # This ensures that all devices share the same barrier and location-specific locks,
        # which are critical for coordinating access to shared resources and timepoint progression.
        if self.device_id == 0:
            # Initialize a reusable barrier for all devices.
            nr_of_devices = devices.__len__()
            barrier_devices = ReusableBarrierSem(nr_of_devices)

            # Initialize a set of locks for each possible location (24 in this case)
            # to prevent race conditions when multiple worker threads or devices
            # try to access or modify data at the same location simultaneously.
            for _ in range(24): # Pre-condition: Assumes a fixed number of locations (24).
                self.lock_locations.append(Lock())

            # Distribute the initialized barrier and location locks to all devices.
            # Post-condition: All devices will share the same barrier and location locks.
            for device in devices:
                device.barrier_devices = barrier_devices
                device.lock_locations = self.lock_locations
                device.event_setup_done.set() # Signal that this device's setup is complete.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device to be executed at a specific location
               or signals the completion of a timepoint if no script is provided.

        @param script: The script object to be executed, or None if the timepoint is done.
        @param location: The location ID associated with the script, or irrelevant if script is None.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # Block Logic: Signals that all scripts for the current timepoint have been assigned.
            # This event acts as a synchronization point, allowing the DeviceThread to proceed
            # after all assignments for a given timepoint are complete.
            self.event_timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The location ID for which to retrieve data.
        @return The sensor data for the specified location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.

        A lock is used to ensure thread-safe access to the sensor data dictionary.

        @param location: The location ID for which to set data.
        @param data: The new sensor data value.
        """
        # Block Logic: Ensures exclusive access to the shared sensor data dictionary.
        # Invariant: Only one thread can modify `self.sensor_data` at a time for any location.
        with self.lock_data:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown process for the device by joining its associated thread.

        This ensures that the Device's main thread completes its execution before
        the program exits.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the lifecycle and operation of a Device instance in its own thread.

    This thread is responsible for orchestrating the device's activities across
    timepoints, including retrieving neighbor information from the supervisor,
    assigning scripts to worker threads, and synchronizing with other devices
    via a shared barrier.
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
        @brief The main execution loop for the DeviceThread.

        It waits for the device to be set up, launches worker threads,
        and then enters a loop to process timepoints. In each timepoint,
        it retrieves neighbor information, assigns scripts to worker threads,
        and synchronizes with other devices using a barrier.
        """
        # Pre-condition: Wait until the device's shared resources (barrier, locks) are set up.
        self.device.event_setup_done.wait()

        # Block Logic: Initializes and starts worker threads for concurrent script execution.
        # These threads will pull scripts from the queue and process them in parallel.
        worker_threads = []
        for thread_id in range(8): # Invariant: 8 worker threads are launched per device.
            thread = WorkerThread(self.device, thread_id)
            worker_threads.append(thread)
            worker_threads[-1].start()

        # Main simulation loop for processing timepoints.
        while True:
            # Block Logic: Retrieves information about neighboring devices from the supervisor.
            # This information is crucial for scripts that need to interact with or gather
            # data from adjacent devices in the simulated environment.
            neighbours = self.device.supervisor.get_neighbours()

            # Pre-condition: If there are no more neighbors (end of simulation), break the loop.
            if neighbours is None:
                break

            # Pre-condition: Wait for the supervisor to signal that all scripts for the current
            # timepoint have been assigned to this device (via assign_script method).
            self.device.event_timepoint_done.wait()

            # Block Logic: Distributes assigned scripts to worker threads via a queue.
            # Each script, along with its location and neighbor information, is placed
            # into the queue for worker threads to pick up.
            for (script, location) in self.device.scripts:
                thread_script = (script, location, neighbours)
                self.device.queue_scripts.put(thread_script)
                # Inline: Releases a semaphore to indicate that a new script is available in the queue.
                self.device.semaphore_queue.release()

            # Post-condition: Resets the event for the next timepoint.
            self.device.event_timepoint_done.clear()

            # Block Logic: Waits until all scripts for the current timepoint have been processed
            # by the worker threads before proceeding to the device barrier synchronization.
            self.device.condition_variable.acquire()
            while self.device.queue_scripts.empty() is False: # Invariant: Loop until the script queue is empty.
                self.device.condition_variable.wait()
            self.device.condition_variable.release()

            # Synchronization point: All devices wait here until every other device
            # has completed its current timepoint processing. This ensures that
            # all devices are synchronized before moving to the next timepoint.
            self.device.barrier_devices.wait()

        # Post-condition: Signals worker threads to stop when the main simulation loop exits.
        self.device.event_stop_threads.set()

        # Block Logic: Releases the semaphore multiple times to unblock all worker threads
        # so they can check the `event_stop_threads` and terminate gracefully.
        for _ in range(8): # Invariant: There are 8 worker threads to unblock.
            self.device.semaphore_queue.release()

        # Block Logic: Waits for all worker threads to complete their execution and terminate.
        for thread in worker_threads:
            thread.join()



class WorkerThread(Thread):
    """
    @brief Executes assigned scripts for a Device, handling data access and updates.

    Worker threads operate in parallel, fetching scripts from a shared queue,
    acquiring necessary locks for locations, executing the scripts, and
    updating sensor data for both the device and its neighbors.
    """
    def __init__(self, device, my_id):
        """
        @brief Initializes a new WorkerThread instance.

        @param device: The Device object to which this worker thread belongs.
        @param my_id: A unique integer identifier for this worker thread.
        """
        Thread.__init__(self, name="Worker Thread %d" % my_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the WorkerThread.

        It continuously waits for scripts to become available in the queue,
        executes them, and updates shared data. The loop terminates when
        the `event_stop_threads` signal is received from the DeviceThread.
        """
        while True:
            # Pre-condition: Waits on a semaphore until a script is available in the queue.
            # This blocks the worker thread until there's work to do.
            self.device.semaphore_queue.acquire()

            # Pre-condition: Checks if the DeviceThread has signaled to stop all worker threads.
            # If so, the worker thread breaks its loop and terminates.
            if self.device.event_stop_threads.is_set():
                break

            # Block Logic: Retrieves a script task from the queue.
            # A script task consists of the script itself, its associated location, and neighbor information.
            thread_script = self.device.queue_scripts.get()

            script = thread_script[0]
            location = thread_script[1]
            neighbours = thread_script[2]

            # Block Logic: Acquires a location-specific lock to ensure exclusive access
            # to the sensor data at this particular location during script execution and data update.
            # Invariant: Only one thread can operate on data at a given 'location' at any time.
            with self.device.lock_locations[location]:
                script_data = []
                # Block Logic: Gathers sensor data from neighboring devices for the specified location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Gathers sensor data from the current device for the specified location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: If there is data to process, execute the script.
                if script_data != []:
                    # Functional Utility: Executes the assigned script with the collected sensor data.
                    # The script's `run` method presumably contains the core logic for data processing.
                    result = script.run(script_data)

                    # Block Logic: Updates the sensor data for all neighboring devices with the script's result.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Updates the sensor data for the current device with the script's result.
                    self.device.set_data(location, result)

            # Block Logic: Notifies the DeviceThread that a script has been processed.
            # This contributes to the condition variable's ability to signal when the queue is empty.
            self.device.condition_variable.acquire()
            self.device.condition_variable.notify() # Post-condition: Acknowledges completion of one script.
            self.device.condition_variable.release()
