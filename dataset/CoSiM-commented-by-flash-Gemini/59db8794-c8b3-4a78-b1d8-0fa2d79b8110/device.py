


"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `Device`: Represents a single device, managing its sensor data and orchestrating operations.
- `WorkerThread`: A worker thread that fetches scripts from a `DeviceThread`'s queue,
  executes them, and updates device data.
- `DeviceThread`: The main thread for a `Device`, acting as a producer of scripts for `WorkerThread`s,
  managing neighbor information, and handling timepoint synchronization.

The system utilizes `collections.deque` as a thread-safe queue for script distribution,
`threading.Semaphore` for flow control and synchronization between producer and consumers,
`threading.Event` for signaling, and imports `ReusableBarrierSem` from a separate `barrier` module
for global device synchronization.
"""

from threading import Event, Thread, Lock, Semaphore
from barrier import ReusableBarrierSem
from collections import deque

class Device(object):
    """
    Represents a single device within the simulated distributed environment.
    Each device manages its own sensor data, communicates with a supervisor,
    and orchestrates multi-threaded script execution through its `DeviceThread`.
    It uses semaphores and a deque to manage script assignments and their processing.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for the device.
            supervisor (object): A reference to a supervisor object for inter-device communication.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to hold assigned scripts (tuples of (script, location)).
        self.setup_done = Event() # Event to signal when the device's setup is complete.
        self.script_semaphore = Semaphore(0) # Semaphore to control the flow of script assignments.
        self.location_locks = [] # List to hold location-specific locks, shared across devices.
        self.queue = deque() # A deque used to store script lengths as stop signals for timepoints.
        
        self.thread = None # Placeholder for the DeviceThread, initialized later.
        self.barrier = None # Placeholder for the global ReusableBarrierSem, initialized later.

    def __str__(self):
        """
        Returns a string representation of the device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def start_thread(self, barrier, locks):
        """
        Initializes and starts the `DeviceThread` for this device,
        assigning the global barrier and shared location locks.

        Args:
            barrier (ReusableBarrierSem): The global barrier for device synchronization.
            locks (list): A list of (location, Lock) tuples for shared data access control.
        """
        self.thread = DeviceThread(self) # Create the DeviceThread instance.
        self.barrier = barrier # Assign the global barrier.

        self.location_locks = locks # Assign the shared location locks.
        self.thread.start() # Start the DeviceThread.
        self.setup_done.set() # Signal that the device's setup is complete.

    def setup_devices(self, devices):
        """
        Initializes and distributes shared synchronization primitives (barrier and locks)
        across all devices in the simulation. This method is designed to be called only
        by the device with `device_id == 0`.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        # Block Logic: Only the device with device_id 0 performs this setup.
        if self.device_id == 0:
            # Inline: Creates a global `ReusableBarrierSem` for synchronization among all `DeviceThread`s.
            barrier = ReusableBarrierSem(len(devices))
            
            locks = [] # List to hold location-specific locks.
            # Block Logic: Initialize a unique `Lock` for each distinct data location present across all devices.
            for device in devices:
                for location in device.sensor_data:
                    if location not in locks: # Prevents adding duplicate locations.
                        locks.append((location, Lock())) # Store location with its new lock.
            
            # Inline: Calls `start_thread` for each device, distributing the global barrier and locks.
            for device in devices:
                device.start_thread(barrier, locks)

    def assign_script(self, script, location):
        """
        Assigns a script to the device or signals the end of scripts for a timepoint.
        If a script is provided, it's appended to the device's script list.
        If `script` is None, the length of the current scripts list is appended
        to an internal queue, serving as a signal for the `DeviceThread`.

        Args:
            script (object): The script object to be executed, or None to signal a timepoint boundary.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
        else:
            # Inline: If script is None, append the current number of scripts to the queue.
            # This value acts as a stop signal for the DeviceThread for the current timepoint.
            self.queue.append(len(self.scripts))
        
        # Inline: Release the semaphore, signaling to the DeviceThread that a new script
        # has been assigned or a timepoint boundary has been marked.
        self.script_semaphore.release()

    def get_data(self, location):
        """
        Retrieves sensor data for a given `location` from this device's `sensor_data` dictionary.

        Args:
            location (int): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        if location in self.sensor_data:
            return self.sensor_data[location]

    def set_data(self, location, data):
        """
        Sets sensor data for a given `location` in this device's `sensor_data` dictionary.
        The data is updated only if the location exists in the `sensor_data`.

        Args:
            location (int): The location identifier for which to set data.
            data (any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device.
        It waits for the device's setup to be complete and then joins its main `DeviceThread`,
        ensuring all associated tasks are finished before the program exits.
        """
        self.setup_done.wait() # Ensure setup is complete before attempting to join the thread.
        self.thread.join() # Wait for the DeviceThread to finish its execution.


class WorkerThread(Thread):
    """
    A worker thread that processes scripts distributed by a `DeviceThread`.
    It fetches script-location pairs, acquires necessary locks, collects data
    from the device and its neighbors, executes the script, and updates device data.
    """

    def __init__(self, device_thread):
        """
        Initializes a `WorkerThread` instance.

        Args:
            device_thread (DeviceThread): The `DeviceThread` that manages this worker.
                                          This provides access to the parent `Device` and shared resources.
        """
        Thread.__init__(self)
        self.device_thread = device_thread # Reference to the managing DeviceThread.

    def run(self):
        """
        The main execution method for the `WorkerThread`.
        It continuously waits for tasks (script and location pairs) from the `DeviceThread`'s
        queue. Upon receiving a task, it acquires necessary locks, collects data from relevant
        devices, executes the script, updates the device data, and then signals completion.
        If a `None` task is received, the thread terminates.
        """
        while True:
            # Block Logic: Acquire `threads_semaphore`, blocking until a script is available.
            # This semaphore is released by the `DeviceThread` when a script is added to `scripts_queue`.
            self.device_thread.threads_semaphore.acquire()
            
            script = None # Placeholder for the script object.
            location = None # Placeholder for the location identifier.
            # Inline: Attempt to retrieve a script and location from the queue.
            if len(self.device_thread.scripts_queue) > 0:
                (script, location) = self.device_thread.scripts_queue.popleft()
            
            # Inline: If `location` is None, it's a sentinel value indicating thread termination.
            if location is None:
                break # Exit the loop, terminating the worker thread.

            # Block Logic: Find the correct shared lock for the current `location`.
            # `next()` is used to get the first lock object that matches the location ID.
            lock = next(l for (x, l) in self.device_thread.device.location_locks
                if x == location)

            lock.acquire() # Acquire the location-specific lock to ensure exclusive data access.
            
            script_data = [] # List to collect input data for the script.
            
            # Block Logic: Collect data from all neighboring devices at the specified location.
            for device in self.device_thread.neighbours:
                data = device.get_data(location) # Get data from the neighbor.
                if data is not None:
                    script_data.append(data) # Add to script input if available.

            # Block Logic: Collect data from this worker's own parent device at the specified location.
            data = self.device_thread.device.get_data(location)
            if data is not None:
                script_data.append(data) # Add to script input if available.

            # Block Logic: If input data is available, execute the script and update device data.
            if script_data != []:
                # Inline: Execute the script's `run` method with the collected data.
                result = script.run(script_data)

                # Block Logic: Update sensor data for all involved devices (neighbors and self) with the result.
                for device in self.device_thread.neighbours:
                    device.set_data(location, result) # Update neighbor's data.
                
                self.device_thread.device.set_data(location, result) # Update this device's own data.
            
            lock.release() # Release the location-specific lock.
            
            # Inline: Release `worker_semaphore` to signal that this worker has completed a task.
            # This is acquired by the `DeviceThread` to track worker progress.
            self.device_thread.worker_semaphore.release()


class DeviceThread(Thread):
    """
    The main orchestrating thread for a `Device`.
    This thread acts as a producer, managing the distribution of scripts to its
    associated `WorkerThread` instances. It also handles fetching neighbor information,
    synchronization for timepoint progression, and overall worker management.
    """

    def __init__(self, device):
        """
        Initializes a `DeviceThread` instance.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device # Reference to the parent Device.
        self.threads_semaphore = Semaphore(0) # Semaphore to signal WorkerThreads that a new script is available.
        
        self.scripts_queue = deque() # Queue to hold (script, location) pairs for WorkerThreads.
        self.worker_threads = [] # List to hold the spawned WorkerThread instances.
        self.neighbours = [] # Local cache of neighbor devices.
        self.worker_semaphore = Semaphore(0) # Semaphore to track completion of tasks by WorkerThreads.
        
        self.nr_threads = 8 # The number of WorkerThread instances to spawn.

        # Block Logic: Create and start `nr_threads` WorkerThread instances.
        for _ in xrange(self.nr_threads):
            thread = WorkerThread(self)
            self.worker_threads.append(thread)
            thread.start()

    def run(self):
        """
        The main execution loop for the `DeviceThread`.
        This thread orchestrates the processing of scripts in timepoints.
        It manages the lifecycle of worker threads, fetches neighbor information,
        distributes scripts for execution, and synchronizes across devices.
        """
        
        index = 0 # Tracks the current script index being processed within a timepoint.
        while True:
            # Block Logic: Ensure all worker threads have completed their tasks from the previous iteration.
            # This is achieved by acquiring `worker_semaphore` for each task completed.
            for _ in xrange(index):
                self.worker_semaphore.acquire()
            
            # Block Logic: Synchronize with other devices at the global barrier.
            # This marks the end of a timepoint's processing for all devices.
            self.device.barrier.wait()
            
            index = 0 # Reset script index for the new timepoint.
            stop = None # Flag to determine when to stop assigning scripts for the current timepoint.
            
            # Block Logic: Fetch updated neighbor information from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()
            
            # Inline: If `neighbours` is None, it signals termination.
            if self.neighbours is None:
                break # Exit the main loop, initiating the shutdown sequence.

            # Block Logic: Loop to distribute scripts to worker threads for the current timepoint.
            while True:
                # Block Logic: Wait for a new script to be available or a timepoint stop signal.
                # If `device.scripts` has no more scripts beyond `index`, it blocks on `script_semaphore`
                # which is released by `device.assign_script`.
                if not len(self.device.scripts) > index:
                    self.device.script_semaphore.acquire()
                
                # Block Logic: Check if a timepoint stop signal has been received via `device.queue`.
                if stop is None:
                    if len(self.device.queue) > 0:
                        stop = self.device.queue.popleft() # Retrieve the stop signal (length of scripts for timepoint).
                
                # Inline: If a stop signal is received and matches the current script index, break the loop.
                if stop is not None and stop == index:
                    break
                
                # Inline: If no stop signal and no more scripts, continue waiting.
                if stop is None and not len(self.device.scripts) > index:
                    continue

                # Block Logic: Assign the current script to a worker thread.
                (script, location) = self.device.scripts[index]
                
                # Inline: Add the script to the internal queue for worker threads.
                self.scripts_queue.append((script, location))
                self.threads_semaphore.release() # Release semaphore to notify a worker thread.
                
                index += 1 # Move to the next script.
        
        # Block Logic: Shutdown sequence for worker threads.
        # After the main loop breaks, signal all worker threads to terminate
        # by releasing `threads_semaphore` for each worker one last time with a `None` in `scripts_queue`.
        for _ in xrange(len(self.worker_threads)):
            self.threads_semaphore.release()
        
        # Block Logic: Wait for all worker threads to finish their execution.
        for thread in self.worker_threads:
            thread.join()

