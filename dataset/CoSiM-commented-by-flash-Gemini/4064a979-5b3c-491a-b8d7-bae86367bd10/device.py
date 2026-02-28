"""
@file device.py
@brief Implements core components for a distributed device simulation.

This module defines the `Device`, `DeviceThread`, and `WorkerThread` classes,
which together simulate a network of interconnected devices processing sensor data
and executing scripts. The architecture supports synchronization between devices
and their internal worker threads, as well as interaction with a central supervisor.

Classes:
    - Device: Represents an individual device in the simulation, managing its
              sensor data, script queue, and associated threads.
    - DeviceThread: The main execution thread for a Device, responsible for
                    synchronization, fetching neighbor information, and
                    orchestrating script assignment.
    - WorkerThread: Threads launched by a Device to execute assigned scripts
                    and process sensor data, including interaction with
                    neighboring devices.
"""

from threading import Event, Thread, Semaphore
from barrier import ReusableBarrierSem
from worker import WorkerThread


class Device(object):
    """
    @class Device
    @brief Represents a single device within the distributed simulation network.

    Manages its unique identifier, local sensor data, interaction with a supervisor,
    and various synchronization primitives for coordinating with other devices
    and its internal worker threads. It orchestrates script execution and data
    exchange within its local context and with neighbors.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id Unique identifier for this device.
        @param sensor_data Initial sensor data pertinent to this device.
        @param supervisor A reference to the central supervisor managing the network.
        """
        self.device_id = device_id  # Unique identifier for the device.
        self.sensor_data = sensor_data  # Dictionary holding sensor data specific to this device.
        self.supervisor = supervisor  # Reference to the supervisor object for network coordination.
        # Event flag to signal that the global barrier has been set up for this device.
        self.barrier_set = Event()
        # Event flag to signal that a script has been received (not currently used actively for coordination).
        self.script_received = Event()
        self.scripts = []  # List to store scripts assigned to this device for execution.
        # Event flag to signal that all scripts for a timepoint are processed by the worker threads.
        self.timepoint_done = Event()
        # The main thread responsible for this device's high-level operations (e.g., synchronization).
        self.thread = DeviceThread(self)
        
        self.barrier = None  # Global synchronization barrier shared by all devices.
        
        self.neighbours = []  # List of neighboring devices, populated by the supervisor.
        
        self.data_locks = []  # List of Semaphores to protect concurrent access to sensor data locations.
        
        self.thread_list = []  # List of WorkerThread instances associated with this device.
        
        self.worker_number = 8  # The number of worker threads to spawn for this device.
        # Barrier for internal synchronization among this device's worker threads.
        self.worker_barrier = ReusableBarrierSem(self.worker_number)
        
        self.script_queue = []  # Queue of scripts ready for worker threads to process.
        # Semaphore to protect access to the `script_queue`.
        self.script_lock = Semaphore(1)
        # Event flag to signal worker threads that the simulation should exit.
        self.exit_flag = Event()
        # Event flag set by worker threads when all tasks for a timepoint are finished.
        self.tasks_finished = Event()
        # Event flag set by the DeviceThread to signal worker threads to start tasks for a new timepoint.
        self.start_tasks = Event()

    def set_flag(self):
        """
        @brief Sets the `barrier_set` Event, signaling that this device is ready
               and its barrier has been set (typically by the master device).
        """
        self.barrier_set.set()

    def set_barrier(self, barrier):
        """
        @brief Sets the global synchronization barrier for this device.
        @param barrier The ReusableBarrierSem instance to be used for global synchronization.
        """
        self.barrier = barrier

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures the device and starts its main and worker threads.
        @param devices A list of all Device instances in the simulation.

        This method handles the initial setup. If this is the master device (device_id == 0),
        it initializes the global barrier and data locks, and propagates them to all other devices.
        Other devices wait for the master to complete this setup.
        Finally, it starts the DeviceThread and all WorkerThreads for this device.
        """
        # Master device (ID 0) performs initial setup for all devices.
        if self.device_id == 0:
            # Initialize a global reusable barrier for all devices.
            self.barrier = ReusableBarrierSem(len(devices))
            location_index = -1
            # Determine the maximum location index across all devices' sensor data.
            for dev in devices:
                for k in dev.sensor_data:
                    if k > location_index:
                        location_index = k

            # Propagate the initialized barrier and signal readiness to all devices.
            for dev in devices:
                dev.set_barrier(self.barrier)
                dev.set_flag()

            # Initialize a dictionary of Semaphores (data_locks) for each possible location,
            # to protect sensor data from race conditions during concurrent access.
            self.data_locks = {loc : Semaphore(1) for loc in range(location_index+1)}
            # Propagate the initialized data locks to all devices.
            for dev in devices:
                dev.data_locks = self.data_locks
        else:
            # Non-master devices wait for the master to set up the global barrier.
            self.barrier_set.wait()
        # Start the main DeviceThread for this device.
        self.thread.start()

        # Start the pool of worker threads for this device.
        for tid in range(self.worker_number):
            thread = WorkerThread(self, tid)
            self.thread_list.append(thread)
            thread.start()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device's script list.
        @param script The script object to be executed.
        @param location The location pertinent to the script execution.

        If `script` is None, it signals that no more scripts are coming for the current timepoint,
        and the `timepoint_done` event is set to unblock the `DeviceThread`.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        @param location The location identifier for which to retrieve data.
        @return The sensor data at the specified location, or None if not present.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.
        @param location The location identifier for which to set data.
        @param data The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its main and worker threads.
        Ensures all threads complete their execution before the program exits cleanly.
        """
        # Join all worker threads to ensure they finish.
        for thread in self.thread_list:
            thread.join()
        # Join the main DeviceThread.
        self.thread.join()


class DeviceThread(Thread):
    """
    @class DeviceThread
    @brief A dedicated thread for a `Device` instance.

    This thread handles the device's high-level synchronization points,
    interacts with the supervisor to obtain neighbor information, and
    orchestrates the flow of tasks for its associated worker threads.
    It manages timepoint progression within the simulation.
    """

    def __init__(self, device):
        """
        @brief Initializes the DeviceThread.
        @param device The Device instance this thread is associated with.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        This loop continuously synchronizes with other device threads,
        fetches updated neighbor information, handles simulation exit conditions,
        prepares the script queue for worker threads, and signals them to start
        processing tasks for the current timepoint.
        """
        while True:
            # Wait at the global barrier for all devices to synchronize at the start of a timepoint.
            self.device.barrier.wait()

            # Retrieve updated neighbor information from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()

            # Check if the simulation should terminate (supervisor returns None for neighbors).
            if self.device.neighbours is None:
                # Signal worker threads to exit and start tasks to unblock them for final exit.
                self.device.exit_flag.set()
                self.device.start_tasks.set()
                break

            # Wait for all scripts for the current timepoint to be assigned and marked as done.
            self.device.timepoint_done.wait()
            # Reset the event for the next timepoint.
            self.device.timepoint_done.clear()

            # Populate the script queue with the assigned scripts for worker threads to process.
            self.device.script_queue = list(self.device.scripts)

            # Signal worker threads to begin processing their assigned tasks for this timepoint.
            self.device.start_tasks.set()

            # Wait for all worker threads to finish their tasks for the current timepoint.
            self.device.tasks_finished.wait()
            # Reset the event for the next timepoint.
            self.device.tasks_finished.clear()

from threading import Thread
class WorkerThread(Thread):
    """
    @class WorkerThread
    @brief Dedicated thread for a Device to execute assigned scripts.

    Worker threads operate in parallel within a device, coordinating their work
    using a `ReusableBarrierSem`. They fetch scripts from a shared queue,
    acquire data locks for specific locations, gather data from neighbors,
    execute the script, and update sensor data.
    """
    
    def __init__(self, device, thread_id):
        """
        @brief Initializes a WorkerThread.
        @param device The Device instance this worker belongs to.
        @param thread_id A unique identifier for this worker thread within its device.
        """
        Thread.__init__(self)
        self.device = device
        self.thread_id = thread_id

    def run(self):
        """
        @brief The main execution loop for the WorkerThread.

        This loop continuously synchronizes with other worker threads,
        waits for tasks to be started, processes scripts from the device's
        shared script queue, and interacts with sensor data (including neighbors')
        while respecting data locks. It terminates when signaled by the `exit_flag`.
        """
        iteratii = 0 # Counter for timepoints/iterations (Romanian for iterations).
        while True:

            # Wait at the worker barrier to ensure all workers are ready for the current phase.
            self.device.worker_barrier.wait()

            # The first worker (thread_id == 0) signals the DeviceThread that tasks are finished
            # after the initial setup phase (iteratii != 0).
            if self.thread_id == 0 and iteratii != 0:
                self.device.tasks_finished.set()

            # Wait for the DeviceThread to signal that tasks for a new timepoint should start.
            self.device.start_tasks.wait()

            # Wait at the worker barrier again before starting actual task processing.
            self.device.worker_barrier.wait()
            # The first worker clears the start_tasks signal for the next timepoint.
            if self.thread_id == 0:
                self.device.start_tasks.clear()
            iteratii += 1
            # Check if the simulation exit flag has been set by the DeviceThread.
            if self.device.exit_flag.is_set():
                break

            # Acquire the script lock to safely access the shared script queue.
            self.device.script_lock.acquire()
            if len(self.device.script_queue) > 0:
                # Pop a script from the queue if available.
                (script, location) = self.device.script_queue.pop(0)
                self.device.script_lock.release()
            else:
                # If no scripts are available, release the lock and continue to next iteration.
                self.device.script_lock.release()
                continue

            # Acquire the data lock for the specific location relevant to this script.
            # This prevents race conditions when multiple workers/devices try to access/modify
            # data at the same location.
            self.device.data_locks[location].acquire()
            
            script_data = [] # List to accumulate data required by the script.
            
            # Gather data from neighboring devices for the specified location.
            for device in self.device.neighbours:
                data = device.get_data(location)
                if data is not None:
                    script_data.append(data)
            
            # Gather data from this device itself for the specified location.
            data = self.device.get_data(location)
            if data is not None:
                script_data.append(data)

            # If any data was collected, run the script and update relevant devices.
            if script_data != []:
                result = script.run(script_data) # Execute the script with the collected data.
                # Propagate the result to neighboring devices.
                for device in self.device.neighbours:
                    device.set_data(location, result)
                
                # Update this device's own data with the script result.
                self.device.set_data(location, result)
            # Release the data lock for the current location.
            self.device.data_locks[location].release()

