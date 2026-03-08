"""
@49a17207-eb15-4eae-8c2d-581c83340ce0/device.py
@brief This module defines a multi-threaded device simulation framework,
including `Device` for managing sensors and scripts, `DeviceThread` for
coordination, `SolveScript` for parallel script execution, and a
`ReusableBarrierSem` for synchronization.
"""

from threading import Event, Semaphore, Lock, Thread
from Queue import Queue

class Device(object):
    """
    Represents a single device in a simulated distributed system.

    Each device manages its own sensor data, processes scripts, and
    coordinates with a supervisor and other devices using threads and barriers.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for this device.
            sensor_data (dict): A dictionary representing the sensor data this device holds.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event() # Event to signal when scripts have been assigned for the current timepoint.
        self.scripts = [] # List to hold assigned scripts.
        self.timepoint_done = Event() # Event (currently unused, potential for future feature).

        # Main thread for device operations.
        self.thread = DeviceThread(self)
        self.thread.start()
        self.no_th = 8 # Number of SolveScript threads to be spawned for parallel processing.

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures shared synchronization primitives (barrier and location-specific locks)
        across all devices in the system. This method is typically called by the supervisor.

        Args:
            devices (list): A list of all Device instances in the simulated system.
        """
        # Block Logic: Only the device with device_id 0 (the 'master' device) initializes global resources.
        if self.device_id == 0:
            # Create a reusable barrier for all devices.
            barrier = ReusableBarrierSem(len(devices))

            # Initialize a dictionary to hold locks for specific data locations, shared across devices.
            lock_for_loct = {}
            for device in devices:
                # Assign the shared barrier to all devices.
                device.barrier = barrier
                for location in device.sensor_data:
                    # Initialize a lock for each unique data location if it doesn't already exist.
                    if location not in lock_for_loct:
                        lock_for_loct[location] = Lock()
                # Assign the shared location locks to all devices.
                device.lock_for_loct = lock_for_loct

    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific data location.

        Args:
            script (object): The script object to be executed.
            location (int): The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script to the device's list.
        else:
            self.scripts_received.set() # If no script, signal that script assignment for this timepoint is complete.

    def get_data(self, location):
        """
        Retrieves data from the device's sensor_data at the specified location.

        Args:
            location (int): The index or key of the data to retrieve.

        Returns:
            any: The data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data, source=None):
        """
        Sets or updates data in the device's sensor_data at the specified location.

        Args:
            location (int): The index or key of the data to set.
            data (any): The new data value.
            source (optional): Information about the source of the data update (currently unused).
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data # Update the data if the location exists.

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device, waiting for its main thread to finish.
        """
        self.thread.join() # Wait for the main DeviceThread to complete its execution.


class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for fetching neighbors,
    distributing scripts to worker threads, and synchronizing at timepoints.
    """
    

    def __init__(self, device):
        """
        Initializes a DeviceThread instance.

        Args:
            device (Device): The parent Device instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.queue = [] # This is actually used as a temporary storage for scripts, not a Queue object.
        self.neighbours = [] # Stores the list of neighboring devices.

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously fetches neighbors, waits for scripts to be assigned,
        distributes these scripts to a pool of `SolveScript` threads for parallel
        execution, and then synchronizes all devices using a global barrier.
        """
        while True:
            # Functional Utility: Fetch the list of neighboring devices from the supervisor.
            self.neighbours = self.device.supervisor.get_neighbours()
            if self.neighbours is None:
                break # If no neighbors are returned (e.g., supervisor signals shutdown), break the loop.

            # Block Logic: Wait until scripts for the current timepoint have been assigned.
            self.device.scripts_received.wait()
            self.device.scripts_received.clear() # Reset the event for the next timepoint.
            
            self.queue = Queue() # Create a new Queue for the current timepoint's scripts.
            # Block Logic: Populate the queue with all assigned scripts.
            for script in self.device.scripts:
                self.queue.put_nowait(script)

            # Block Logic: Start `no_th` (8) `SolveScript` threads to process scripts from the queue.
            for _ in range(self.device.no_th):
                SolveScript(self.device, self.neighbours, self.queue).start()
            
            self.queue.join() # Wait until all scripts in the queue have been processed.
            
            # Block Logic: Synchronize all devices at the global barrier after script execution.
            self.device.barrier.wait()

class SolveScript(Thread):
    """
    A worker thread designed to fetch a script from a queue, execute it,
    and update data on the current and neighboring devices.

    Multiple instances of this class run in parallel to process scripts.
    """
    

    def __init__(self, device, neighbours, queue):
        """
        Initializes a SolveScript worker thread.

        Args:
            device (Device): The parent Device instance this thread belongs to.
            neighbours (list): A list of neighboring Device instances.
            queue (Queue): The queue from which to retrieve scripts for execution.
        """
        Thread.__init__(self)
        self.device = device
        self.neighbours = neighbours
        self.queue = queue

    def run(self):
        """
        The main execution loop for the SolveScript worker thread.

        It continuously retrieves scripts from the queue, collects relevant data
        from the local and neighboring devices, executes the script, and updates
        the data based on the script's result, all while ensuring thread safety.
        """
        try:
            # Block Logic: Continuously get and process scripts from the queue.
            # The `for` loop is likely a stylistic choice as `queue.get()` handles blocking.
            for (script, location) in self.device.scripts: # This loop structure might be misleading; a `while True` with `queue.get()` is more typical for worker threads.
                # Functional Utility: Retrieve a script and its location from the queue. `False` for non-blocking initially.
                (script, location) = self.queue.get(False) # Assumes `get()` might be non-blocking with timeout for some reason, or should be `True` (blocking)

                # Pre-condition: Acquire a lock for the specific data location to ensure exclusive access during script execution.
                self.device.lock_for_loct[location].acquire()

                script_data = [] # List to accumulate data for the current script.
                
                # Block Logic: Collect data from neighboring devices.
                for device in self.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collect data from the current device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Pre-condition: If there is data collected, execute the script.
                if script_data != []:
                    # Action: Execute the script.
                    result = script.run(script_data)
                    
                    # Block Logic: Update data in neighboring devices.
                    for device in self.neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Update data in the current device.
                    self.device.set_data(location, result)

                # Post-condition: Release the lock for the data location.
                self.device.lock_for_loct[location].release()
                
                # Functional Utility: Signal that the task retrieved from the queue is done.
                self.queue.task_done()
        except:
            # Error Handling: Catches any exceptions during script processing, allowing the thread to terminate gracefully.
            pass

class ReusableBarrierSem():
    """
    A reusable barrier synchronization primitive implementing a two-phase wait
    mechanism using semaphores and a lock.

    Threads wait in two distinct phases, allowing for efficient resetting and reuse.
    """
    

    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrierSem.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                in each phase before proceeding.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads # Counter for threads in phase 1.
        self.count_threads2 = self.num_threads # Counter for threads in phase 2.
        self.counter_lock = Lock()               # Lock to protect access to thread counters.
        self.threads_sem1 = Semaphore(0)         # Semaphore for releasing threads from phase 1.
        self.threads_sem2 = Semaphore(0)         # Semaphore for releasing threads from phase 2.

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have passed both
        phase 1 and phase 2 of the barrier.
        """
        self.phase1() # Execute the first phase of the barrier.
        self.phase2() # Execute the second phase of the barrier.

    def phase1(self):
        """
        First phase of the barrier. All threads decrement a counter and wait
        on `threads_sem1` until the last thread releases them.
        """
        # Block Logic: Atomically decrement the counter and check if this is the last thread.
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # If this is the last thread, release all waiting threads from phase 1.
                for _ in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset counter for next use.

        self.threads_sem1.acquire() # Acquire (wait on) the semaphore until released by the last thread.

    def phase2(self):
        """
        Second phase of the barrier. Similar to phase 1, but uses `threads_sem2`
        and `count_threads2`.
        """
        # Block Logic: Atomically decrement the counter and check if this is the last thread.
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # If this is the last thread, release all waiting threads from phase 2.
                for _ in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset counter for next use.

        self.threads_sem2.acquire() # Acquire (wait on) the semaphore until released by the last thread.
