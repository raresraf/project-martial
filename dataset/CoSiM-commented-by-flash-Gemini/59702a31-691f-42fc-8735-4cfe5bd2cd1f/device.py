


"""
This module implements a simulated multi-threaded distributed device system.

It defines classes for:
- `ReusableBarrier`: A barrier synchronization mechanism using `threading.Condition` that can be reused.
- `Device`: Represents a single device, managing its sensor data, communication with a supervisor,
  and orchestrating multi-threaded script execution.
- `DeviceThread`: Worker threads for a `Device`, each processing a subset of scripts,
  collecting data from neighbors, and participating in synchronization.

The system utilizes various `threading` primitives like `Lock`, `Event`, and `Condition`
for intricate synchronization, including a specific `zavor` lock and `gotneighbours` event
for managing neighbor information, and `lockforlocation` for fine-grained data access control.
"""

from threading import Lock, Event, Thread, Condition

class ReusableBarrier():
    """
    A reusable barrier synchronization mechanism for multiple threads using `threading.Condition`.
    Threads wait at this barrier until a specified number of threads (`num_threads`) have arrived.
    Once all threads arrive, they are all notified and released simultaneously.
    The barrier can then be reused for subsequent synchronization points.
    """
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier with a specified number of threads.

        Args:
            num_threads (int): The total number of threads that will participate
                               in the barrier synchronization.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads # Counter for threads currently waiting at the barrier.
        self.cond = Condition()                # The condition variable used for thread synchronization.
                                                 

    def wait(self):
        """
        Causes the calling thread to wait at the barrier until all `num_threads`
        have arrived. Once all threads are present, they are all released.
        """
        self.cond.acquire()                      # Acquire the condition variable's intrinsic lock.
        self.count_threads -= 1                  # Decrement the count of threads waiting.
        if self.count_threads == 0:              # If this is the last thread to arrive:
            self.cond.notify_all()               # Notify all waiting threads to resume.
            self.count_threads = self.num_threads    # Reset the counter for the next use of the barrier.
        else:
            self.cond.wait()                    # If not the last thread, wait to be notified.
        self.cond.release()                     # Release the condition variable's intrinsic lock.


class Device(object):
    """
    Represents a single device within a simulated distributed environment.
    Each device manages its own sensor data, communicates with a supervisor,
    and orchestrates multi-threaded script execution. It uses various
    synchronization primitives to coordinate its internal worker threads
    and interact with other devices.
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
        self.script_received = Event() # Event to signal when new scripts are assigned.
        self.scripts = [] # List to hold assigned scripts (tuples of (script, location)).
        self.timepoint_done = Event() # Event to signal that the current timepoint's processing is complete.

        self.gotneighbours = Event() # Event to signal that neighbor information has been fetched.
        self.zavor = Lock() # A lock to protect access to neighbor fetching logic.
        self.threads = [] # List to hold the DeviceThread worker instances.
        self.neighbours = [] # List to store references to neighboring devices.
        self.nthreads = 8 # The number of DeviceThread workers this Device will spawn.
        self.barrier = ReusableBarrier(1) # Placeholder for the global barrier, initialized to a dummy barrier.
                                          # This will be replaced by a shared barrier in `setup_devices`.
        self.lockforlocation = {} # Dictionary to hold locks for specific data locations, shared across devices.
        # Retrieves the total number of locations from the supervisor's testcase.
        self.num_locations = supervisor.supervisor.testcase.num_locations 
        # Inline: Create and start `nthreads` DeviceThread workers for this device.
        for i in xrange(self.nthreads):
            self.threads.append(DeviceThread(self, i))
        for i in xrange(self.nthreads):
            self.threads[i].start()


    def __str__(self):
        """
        Returns a string representation of the device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the global barrier and shared location-specific locks across all devices.
        This method is intended to be called by a single orchestrating entity (e.g., the supervisor)
        after all devices have been initialized.

        Args:
            devices (list): A list of all `Device` objects in the simulation.
        """
        # Block Logic: This setup typically assumes all devices have the same `nthreads` and `num_locations`.
        # Creates a single `ReusableBarrier` for all `DeviceThread` instances across all `Device`s.
        barrier = ReusableBarrier(devices[0].nthreads*len(devices))
        lockforlocation = {} # Dictionary to store a Lock for each distinct location.
        # Inline: Initialize a unique `Lock` for each possible data location.
        for i in xrange(0, devices[0].num_locations):
            lock = Lock()
            lockforlocation[i] = lock
        # Inline: Distribute the created global barrier and location-specific locks to all devices.
        for i in xrange(0, len(devices)):
            devices[i].barrier = barrier
            devices[i].lockforlocation = lockforlocation


    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific data location.

        Args:
            script (object): The script object to be executed. If None, it signals timepoint completion.
            location (int): The location identifier in the sensor data to which the script applies.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list.
            self.script_received.set() # Signal that a new script has been received.
        else:
            self.timepoint_done.set() # If no script, signal that the timepoint's script assignment is done.

    def get_data(self, location):
        """
        Retrieves sensor data for a given location from this device's `sensor_data` dictionary.

        Args:
            location (int): The location identifier for which to retrieve data.

        Returns:
            any: The sensor data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets sensor data for a given location in this device's `sensor_data` dictionary.
        The data is updated only if the location exists in the `sensor_data`.

        Args:
            location (int): The location identifier for which to set data.
            data (any): The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Initiates the shutdown process for the device by waiting for all its `DeviceThread` instances to complete.
        """
        for i in xrange(self.nthreads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    A worker thread for a `Device` instance.
    Each `DeviceThread` processes a subset of the assigned scripts,
    collects data from neighboring devices (including its own `Device`),
    executes the scripts, and updates data. It also participates in
    complex synchronization for fetching neighbor data and timepoint progression.
    """
    
    def __init__(self, device, id_thread):
        """
        Initializes a `DeviceThread` worker.

        Args:
            device (Device): The parent `Device` object this thread belongs to.
            id_thread (int): A unique identifier (index) for this thread among the device's workers.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id_thread = id_thread

    def run(self):
        """
        The main execution loop for a `DeviceThread` worker.
        It continuously processes scripts assigned to it, handles neighbor data fetching,
        executes scripts, updates data, and participates in various synchronization steps.
        """
        while True:
            # Block Logic: Coordinate neighbor information fetching.
            # Only one thread (protected by `self.device.zavor`) fetches neighbors for the device.
            self.device.zavor.acquire() # Acquire the lock to protect neighbor fetching.
            
            # Inline: If neighbor information hasn't been fetched for the current timepoint:
            if self.device.gotneighbours.is_set() == False:
                self.device.neighbours = self.device.supervisor.get_neighbours() # Fetch neighbors from supervisor.
                self.device.gotneighbours.set() # Signal that neighbors have been fetched.
            self.device.zavor.release() # Release the lock.
            
            # Inline: If `neighbours` is None, it's a termination signal.
            if self.device.neighbours is None:
                break # Exit the loop, signaling thread shutdown.

            # Block Logic: Wait for the current timepoint's script assignment to be complete.
            self.device.timepoint_done.wait()
            
            myscripts = [] # List to store scripts assigned specifically to this worker thread.
            # Block Logic: Distribute scripts among `nthreads` workers in a round-robin fashion.
            # Each worker processes scripts with an index matching its `id_thread` modulo `nthreads`.
            # The original code uses `self.device.nthreads + 1` in the step, which is unusual for a direct round-robin.
            # Assuming it should be `self.device.nthreads` for standard round-robin distribution.
            for i in xrange(self.id_thread, len(self.device.scripts), self.device.nthreads + 1):
                myscripts.append(self.device.scripts[i])

            # Block Logic: Process scripts assigned to this worker thread.
            for (script, location) in myscripts:
                # Inline: Acquire the global lock for the specific location to ensure exclusive data access.
                self.device.lockforlocation[location].acquire()
                script_data = [] # List to collect input data for the current script.
                
                # Block Logic: Collect data from all neighboring devices for the current script's location.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collect data from this device's own sensor data.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: If data was collected, execute the script and update device data.
                if script_data != []:
                    # Inline: Execute the script's `run` method with the collected data.
                    result = script.run(script_data)
                    # Block Logic: Distribute the result to all involved devices (neighbors and self).
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    self.device.set_data(location, result) # Update this device's own data.

                self.device.lockforlocation[location].release() # Release the global lock for the current location.

            # Block Logic: First synchronization point for all DeviceThreads.
            # Ensures all workers have finished processing their scripts for the timepoint.
            self.device.barrier.wait()
            
            # Block Logic: Clear events for the next timepoint.
            # Only the thread with `id_thread == 0` is responsible for clearing these global events.
            if self.id_thread == 0:
                self.device.timepoint_done.clear() # Clear timepoint done event.
                self.device.gotneighbours.clear() # Clear neighbor fetched event.
            
            # Block Logic: Second synchronization point for all DeviceThreads.
            # Ensures all threads are ready before starting the next timepoint cycle.
            self.device.barrier.wait()            