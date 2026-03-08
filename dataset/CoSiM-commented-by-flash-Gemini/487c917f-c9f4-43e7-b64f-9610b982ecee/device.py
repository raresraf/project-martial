"""
@487c917f-c9f4-43e7-b64f-9610b982ecee/device.py
@brief This module defines classes for simulating a distributed system, where
`Device` instances process sensor data and execute scripts. It includes
a custom `ReusableBarrier` for synchronization and `DeviceThread` for
multi-threaded operation.
"""

from threading import Lock, Event, Thread, Condition

class ReusableBarrier():
    """
    A reusable barrier synchronization primitive for coordinating multiple threads.

    Threads wait at the barrier until all `num_threads` have arrived. Once all
    threads have arrived, they are all released, and the barrier resets for
    subsequent use.
    """
    def __init__(self, num_threads):
        """
        Initializes the ReusableBarrier.

        Args:
            num_threads (int): The total number of threads that must reach the barrier
                                before any are released.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads     # Current count of threads waiting at the barrier.
        self.cond = Condition()                   # Condition variable for thread synchronization.

    def wait(self):
        """
        Blocks until all threads have reached the barrier.

        Upon the last thread's arrival, all waiting threads are released,
        and the barrier resets for subsequent synchronization rounds.
        """
        self.cond.acquire()                      # Acquire the condition variable lock.
        self.count_threads -= 1                  # Decrement the count of threads yet to arrive.
        if self.count_threads == 0:              # Check if this is the last thread to arrive.
            self.cond.notify_all()               # If all threads have arrived, release all waiting threads.
            self.count_threads = self.num_threads    # Reset the thread count for the next barrier cycle.
        else:
            self.cond.wait()                    # If not the last thread, wait until all others arrive.
        self.cond.release()                     # Release the condition variable lock.


class Device(object):
    """
    Represents a single device in a simulated distributed system.

    Each device manages its own sensor data, interacts with a supervisor,
    and dispatches script execution to a pool of `DeviceThread` workers.
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
        self.script_received = Event() # Event to signal when scripts are assigned.
        self.scripts = [] # List to hold assigned scripts.
        self.timepoint_done = Event() # Event to signal the completion of a timepoint's tasks.
        self.gotneighbours = Event() # Event to signal if neighbors have been fetched.
        self.zavor = Lock() # A general-purpose lock for critical sections within the Device.
        self.threads = [] # List to hold DeviceThread worker instances.
        self.neighbours = [] # List to store neighboring devices.
        self.nthreads = 8 # Number of worker threads for this device.
        self.barrier = ReusableBarrier(1) # Placeholder barrier, will be updated by setup_devices.
        self.lockforlocation = {} # Dictionary to store locks for specific data locations.
        # Functional Utility: Get the total number of locations from the supervisor's test case.
        self.num_locations = supervisor.supervisor.testcase.num_locations
        
        # Block Logic: Create and start DeviceThread worker instances.
        for i in xrange(self.nthreads):
            self.threads.append(DeviceThread(self, i))

        for i in xrange(self.nthreads):
            self.threads[i].start() # Start each worker thread.


    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures shared synchronization primitives (barrier and locks) among all devices.

        This method is typically called once by the supervisor.

        Args:
            devices (list): A list of all Device instances in the system.
        """
        # Block Logic: Create a shared ReusableBarrier for all threads across all devices.
        barrier = ReusableBarrier(devices[0].nthreads*len(devices))
        
        # Block Logic: Initialize shared locks for each data location.
        lockforlocation = {}
        for i in xrange(0, devices[0].num_locations):
            lock = Lock()
            lockforlocation[i] = lock # Assign a unique lock to each data location.
        
        # Block Logic: Distribute the shared barrier and location locks to all devices.
        for i in xrange(0, len(devices)):
            devices[i].barrier = barrier
            devices[i].lockforlocation = lockforlocation


    def assign_script(self, script, location):
        """
        Assigns a script and its associated data location to the device for future execution.

        Args:
            script (object): The script object to be executed.
            location (int): The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script to the device's list.
            self.script_received.set() # Signal that scripts have been received.
        else:
            self.timepoint_done.set() # If no script, signal timepoint completion anyway.

    def get_data(self, location):
        """
        Retrieves data from the device's sensor_data at the specified location.

        Args:
            location (int): The index or key of the data to retrieve.

        Returns:
            any: The data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets or updates data in the device's sensor_data at the specified location.

        Args:
            location (int): The index or key of the data to set.
            data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data # Update the data if the location exists.

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device, joining all its worker threads.
        """
        for i in xrange(self.nthreads):
            self.threads[i].join() # Wait for each worker thread to complete.


class DeviceThread(Thread):
    """
    Worker thread responsible for executing a subset of scripts assigned to a Device.

    Each Device can have multiple `DeviceThread` instances to process scripts concurrently.
    """
    
    def __init__(self, device, id_thread):
        """
        Initializes a DeviceThread instance.

        Args:
            device (Device): The parent Device instance this thread belongs to.
            id_thread (int): A unique identifier for this worker thread.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id_thread = id_thread

    def run(self):
        """
        The main execution loop for the worker thread.

        It continuously fetches neighbors (if not already fetched by another worker),
        waits for timepoint synchronization, processes its assigned scripts,
        and participates in barrier synchronization.
        """
        while True:
            # Block Logic: Acquire a lock to ensure only one thread fetches neighbors at a time.
            self.device.zavor.acquire()
            
            # Pre-condition: Check if neighbors have already been fetched for the current timepoint.
            if self.device.gotneighbours.is_set() == False:
                # If not, fetch neighbors from the supervisor.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.gotneighbours.set() # Signal that neighbors have been fetched.
            self.device.zavor.release() # Release the lock.
            
            # If no neighbors are returned (e.g., supervisor signals shutdown), break the loop.
            if self.device.neighbours is None:
                break

            # Block Logic: Wait for the current timepoint's scripts to be assigned.
            self.device.timepoint_done.wait()
            
            myscripts = [] # List to store scripts assigned to this specific worker thread.
            # Block Logic: Distribute scripts among worker threads in a round-robin fashion.
            # Invariant: Each thread processes scripts indexed by `id_thread`, `id_thread + nthreads + 1`, etc.
            for i in xrange(self.id_thread, len(self.device.scripts), self.device.nthreads + 1):
                myscripts.append(self.device.scripts[i])

            # Block Logic: Execute the scripts assigned to this worker thread.
            for (script, location) in myscripts:
                # Pre-condition: Acquire a lock for the specific data location to ensure exclusive access.
                self.device.lockforlocation[location].acquire()
                script_data = [] # Data collected for the current script.
                
                # Block Logic: Collect data from neighboring devices.
                for device in self.device.neighbours:
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
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    # Block Logic: Update data in the current device.
                    self.device.set_data(location, result)
                self.device.lockforlocation[location].release() # Release the lock for the data location.

            # Block Logic: Synchronize all worker threads and devices at the first barrier.
            self.device.barrier.wait()
            
            # Block Logic: Only thread 0 performs the cleanup for the next timepoint.
            if self.id_thread == 0:
                self.device.timepoint_done.clear() # Reset the timepoint completion signal.
                self.device.gotneighbours.clear() # Reset the neighbors fetched signal.
            
            # Block Logic: Synchronize all worker threads and devices at the second barrier.
            # This ensures that thread 0's cleanup operations are completed before any thread proceeds.
            self.device.barrier.wait()
            