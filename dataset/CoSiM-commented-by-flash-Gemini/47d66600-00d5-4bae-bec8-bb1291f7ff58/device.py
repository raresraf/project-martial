"""
@47d66600-00d5-4bae-bec8-bb1291f7ff58/device.py
@brief Defines classes for simulating a distributed system, where individual
`Device` instances process sensor data and execute scripts in a multi-threaded
environment. It includes a custom reusable barrier for synchronization and
dedicated threads for script execution and worker management.
"""


from threading import Event, Thread

from threading import Condition, RLock    




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
        self.count_threads = num_threads
        self.cond = Condition(RLock())

    def wait(self):
        """
        Blocks until all threads have reached the barrier.

        Upon the last thread's arrival, all waiting threads are released,
        and the barrier resets for subsequent synchronization rounds.
        """
        self.cond.acquire()                      # Acquire the condition variable lock to protect shared state.
        self.count_threads -= 1;                 # Decrement the count of threads yet to arrive.
        if self.count_threads == 0:              # Check if this is the last thread to arrive.
            self.cond.notify_all()               # If all threads have arrived, release all waiting threads.
            self.count_threads = self.num_threads # Reset the thread count for the next barrier cycle.
        else:
            self.cond.wait();                    # If not the last thread, wait until all others arrive.
        self.cond.release();                   # Release the condition variable lock.




class DeviceThread_Worker(Thread):
    """
    Manages the execution of assigned scripts on a subset of data for a specific device.

    Each worker thread processes a portion of the scripts, collecting data from
    neighboring devices and its own device before running the script and updating
    relevant data.
    """
    def __init__(self, device, neighbours, tid, scripts):
        """
        Initializes a DeviceThread_Worker instance.

        Args:
            device (Device): The Device instance this worker belongs to.
            neighbours (list): A list of neighboring Device instances.
            tid (int): The thread ID of this worker.
            scripts (list): A list of (script, location) tuples for this worker to execute.
        """
        Thread.__init__(self)
        self.neighbours = neighbours
        self.device = device
        self.scripts = scripts
        self.tid = tid 

    def run(self):
        """
        Executes the assigned scripts, collecting sensor data from neighbors and
        the current device, running the script, and updating the data.

        Block Logic: Iterates through each script, gathering data from specified
        locations in neighboring devices and the current device, then executes
        the script with the collected data. If the script produces a result,
        it updates the local device's data and potentially neighboring devices'
        data based on a comparison.
        """
        for (script, location) in self.scripts:

            # Initialize a list to hold data for the current script execution.
            script_data = []
            index = location # The location/index for data retrieval and updates.

            # Block Logic: Iterate through neighboring devices to collect sensor data.
            for device in self.neighbours:
                # Acquire locks to ensure exclusive access to the device's data at the specified location.
                self.device.locks[index].acquire()
                self.device.lock.acquire()

                data = device.get_data(location) # Retrieve data from the neighbor at the given location.

                self.device.lock.release() # Release the main device lock.
                self.device.locks[index].release() # Release the specific location lock.

                if data is not None:
                    script_data.append(data) # If data is valid, add it to the script's input data.

            # Block Logic: Collect sensor data from the current device.
            self.device.locks[index].acquire() # Acquire the lock for the current device's data at the location.
            self.device.lock.acquire() # Acquire the main device lock.

            data = self.device.get_data(location) # Retrieve data from the current device at the given location.

            self.device.lock.release() # Release the main device lock.
            self.device.locks[index].release() # Release the specific location lock.

            if data is not None:
                script_data.append(data) # If data is valid, add it to the script's input data.

            # Block Logic: Execute the script if there is collected data and update device data based on the result.
            if script_data != []:
                result = script.run(script_data) # Execute the script with the collected data.

                self.device.locks[index].acquire() # Acquire lock before updating shared data.

                # Update neighboring devices' data if the script result is greater.
                for dev in self.neighbours:
                    if result > dev.get_data(location):
                        dev.set_data(location, result)
                    
                # Update the current device's data with the script's result.
                self.device.set_data(location, result)

                self.device.locks[index].release() # Release lock after data update.


class Device(object):
    """
    Represents a single device in the distributed system, responsible for managing
    its own sensor data, interacting with a supervisor, and executing scripts
    via worker threads.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): Initial sensor data for the device.
            supervisor (Supervisor): The supervisor object responsible for managing devices.
        """
        self.lock = RLock()
        self.barrier = None
        self.devices = []
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.scripts = []
        self.locations = []
        self.locks = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Sets up the device's awareness of other devices in the system and initializes
        shared locks and a synchronization barrier if this is the first device.

        Args:
            devices (list): A list of all Device instances in the system.
        """
        
        self.devices = devices
        if self.device_id == 0:
            # Block Logic: Initialize shared RLock objects for each possible data location.
            # These locks ensure that only one device can modify data at a specific location at a time.
            for num in range(0, 1000):
                lock = RLock()
                for i in range (0, len(devices)):
                    devices[i].locks.append(lock)
            
            # Initialize a reusable barrier for synchronizing all devices.
            barrier = ReusableBarrier(len(devices)) 
            # Assign the same barrier instance to all devices.
            for i in range(0,len(devices)):
                if devices[i].barrier == None:
                    devices[i].barrier = barrier


    def assign_script(self, script, location):
        """
        Assigns a script to the device for execution at a specific data location.

        Args:
            script (object): The script object to be executed.
            location (int): The data location relevant to the script.
        """
        if script is not None:
            self.scripts.append((script, location)) 
            self.timepoint_done.set() # Signal that a script has been assigned and the timepoint can proceed.
            
        else:
            self.timepoint_done.set() # Signal even if no script, to unblock the timepoint.
            self.script_received.set() # Signal that script reception is complete for this timepoint.

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
        self.lock.acquire() # Acquire the main device lock before modifying shared sensor data.
        if location in self.sensor_data:
            self.sensor_data[location] = data # Update the data if the location exists.
        self.lock.release() # Release the main device lock.
    

    def shutdown(self):
        """
        Initiates the shutdown sequence for the device, joining its main thread.
        """
        self.thread.join()


 


class DeviceThread(Thread):
    """
    Manages the overall execution flow for a Device, including fetching neighbors,
    distributing scripts to worker threads, and synchronizing at timepoints.
    """

    def __init__(self, device):
        """
        Initializes a DeviceThread instance.

        Args:
            device (Device): The Device instance this thread manages.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def divide_in_threads(self, neighbours):
        """
        Divides the device's assigned scripts among multiple worker threads
        for parallel execution.

        Args:
            neighbours (list): A list of neighboring Device instances.

        Block Logic: This method creates `DeviceThread_Worker` instances,
        assigns a subset of scripts to each, starts them, and waits for
        their completion. The number of worker threads is capped at 8
        or the total number of scripts if fewer than 8.
        """
        # List to hold the worker threads.
        threads = []

        # Determine the number of scripts and calculate how many worker threads to use.
        nr = len(self.device.scripts)
        numar = 1 
        if nr > 8:
            numar = nr / 8 # Distribute scripts among 8 worker threads.
            nr = 8 # Cap the number of worker threads at 8.

        # Block Logic: Create and initialize DeviceThread_Worker instances.
        for i in range(0,nr):
            if i == nr - 1:
                # Assign remaining scripts to the last worker thread.
                t = DeviceThread_Worker(self.device, neighbours, i, self.device.scripts[i * numar : len(self.device.scripts)])
            else:
                # Assign a chunk of scripts to each worker thread.
                t = DeviceThread_Worker(self.device, neighbours, i, self.device.scripts[i * numar : i*numar + numar])
            threads.append(t)

        # Block Logic: Start all worker threads.
        for i in range(0, nr):
            threads[i].start()

        # Block Logic: Wait for all worker threads to complete their execution.
        for i in range(0,nr):
            threads[i].join()

    def run(self):
        """
        The main execution loop for the DeviceThread.

        It continuously fetches neighbors, waits for scripts to be assigned,
        divides and executes scripts in worker threads, and then synchronizes
        with other devices using a barrier.
        """
        while True:
            # Pre-condition: Fetches the current set of neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Exit the loop if no neighbors are returned (e.g., shutdown signal).

            # Block Logic: Wait for a script to be assigned or a timepoint to be done.
            self.device.script_received.wait()

            # Block Logic: Divides the assigned scripts into worker threads and executes them.
            self.divide_in_threads(neighbours)

            # Post-condition: Clears the script_received event, resetting it for the next timepoint.
            self.device.script_received.clear()

            # Block Logic: Synchronizes with all other devices at the barrier, marking the end of a timepoint.
            self.device.barrier.wait()

 
