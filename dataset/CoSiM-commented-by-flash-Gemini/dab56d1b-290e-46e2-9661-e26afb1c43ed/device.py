

"""
@brief This module defines the ReusableBarrier class for thread synchronization and the Device and DeviceThread classes,
which represent simulated devices in a distributed system.
@details The devices can execute scripts, interact with their neighbors, and share sensor data, simulating
a distributed sensor network or an IoT environment.
"""

from threading import Thread,Event,Condition,Lock

class ReusableBarrier():
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.
    @details This barrier allows a set number of threads to wait until all participants arrive at a synchronization point,
    after which all threads are released simultaneously. The barrier can then be reused for subsequent synchronization points.
    @algorithm Condition Variable based synchronization.
    @time_complexity O(1) for `wait` operation, assuming constant time for underlying threading primitives.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes a new instance of the ReusableBarrier.
        @param num_threads (int): The total number of threads that must reach the barrier before it can be passed.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads    # Current count of threads waiting at the barrier.
        self.cond = Condition()                   # Condition variable used for thread synchronization.
 
    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all `num_threads` threads have arrived.
        @details Once all threads have arrived, they are all released, and the barrier is reset for future use.
        @block_logic Thread synchronization mechanism.
        @pre_condition `self.cond` is an initialized Condition object, `self.count_threads` accurately reflects
                       the number of threads currently waiting or yet to arrive.
        @invariant All threads attempting to pass the barrier will eventually be released together.
        """
        self.cond.acquire()                      # Acquire the lock associated with the condition variable.
        self.count_threads -= 1;                 # Decrement the count of threads yet to arrive.
        if self.count_threads == 0:              # Check if this is the last thread to arrive at the barrier.
            self.cond.notify_all()               # Release all waiting threads.
            self.count_threads = self.num_threads    # Reset the thread count, making the barrier reusable.
        else:
            self.cond.wait();                    # Wait for other threads to arrive (releases the lock implicitly).
        self.cond.release();                     # Release the lock after being notified or decrementing the count.


class Device(object):
    """
    @brief Represents a simulated device within a distributed sensor network.
    @details This class models an individual device that can collect sensor data,
    interact with a central supervisor, and execute scripts. It can also communicate
    with neighboring devices to share and process data.
    @architectural_intent Acts as an autonomous agent in a distributed system,
                           capable of local data processing and communication with peers.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id (int): A unique identifier for the device.
        @param sensor_data (dict): A dictionary containing initial sensor data,
                                   where keys are locations and values are data readings.
        @param supervisor (object): A reference to the supervisor object that manages
                                    the overall distributed system and provides access
                                    to network information (e.g., neighbors).
        """
        self.lock = None             # Lock for protecting shared resources (e.g., scripts list)
        self.barrier = None          # Reference to the ReusableBarrier for thread synchronization
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event() # Event to signal when a new script has been assigned
        self.scripts = []            # List to store assigned scripts and their locations
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's processing

    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return str: A string in the format "Device %d" % device_id.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the device's threading environment and barrier.
        @details This method initializes the ReusableBarrier (if this is the first device),
        creates a DeviceThread for this device, and starts the thread.
        @param devices (list): A list of all Device objects in the simulation.
        @block_logic Initializes the global barrier and creates the worker thread for this device.
        @pre_condition `devices` is a list of `Device` instances.
        @invariant `Device.barrier` is initialized exactly once for the entire system, and `self.thread` is a running `DeviceThread`.
        """
        # Block Logic: Ensure the ReusableBarrier is initialized only once by the first device (device_id == 0).
        for i in devices:
	        if self.device_id == 0:
	            Device.barrier = ReusableBarrier(len(devices)) # Initialize a shared barrier for all devices.


        self.lock = Lock() # Initialize a new lock for this device's critical sections.
        self.thread = DeviceThread(self, Device.barrier , self.lock) # Create a new thread for this device.
        self.thread.start() # Start the device's execution thread.


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific location.
        @details If a script is provided, it's added to the device's script queue, and the `script_received` event is set.
        If no script is provided (i.e., `script` is None), it signifies that the current timepoint's script assignment is complete,
        and the `timepoint_done` event is set.
        @param script (object): The script object to be executed, or None to signal end of assignments for a timepoint.
        @param location (str): The location associated with the script or data.
        @block_logic Handles the assignment of new scripts or signals the completion of script assignment for a timepoint.
        @pre_condition `self.scripts` is a list, `self.script_received` and `self.timepoint_done` are Event objects.
        @invariant Either a script is added and `script_received` is set, or `timepoint_done` is set.
        """
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the queue.
            self.script_received.set() # Signal that a new script has been received.
        else:
            self.timepoint_done.set() # Signal that script assignments for the current timepoint are complete.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.
        @param location (str): The location for which to retrieve data.
        @return object: The sensor data at the specified location, or None if the location is not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.
        @param location (str): The location whose data is to be updated.
        @param data (object): The new data value for the specified location.
        @block_logic Updates the internal sensor data if the location exists.
        @pre_condition `self.sensor_data` is a dictionary.
        @invariant If `location` is a key in `self.sensor_data`, its value is updated.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by joining its associated thread.
        @details This ensures that the device's worker thread completes its execution before the program exits.
        """
        self.thread.join()




class DeviceThread(Thread):
    """
    @brief Encapsulates the execution logic for a Device instance in a separate thread.
    @details This thread is responsible for synchronizing with other device threads,
    fetching neighbor information, executing assigned scripts, and updating sensor data.
    @architectural_intent Enables concurrent operation of multiple Device instances,
                           allowing them to perform their tasks in parallel.
    """

    def __init__(self, device , barrier , lock):
        """
        @brief Initializes a new DeviceThread instance.
        @param device (Device): The Device object that this thread will manage.
        @param barrier (ReusableBarrier): The shared barrier for synchronizing with other device threads.
        @param lock (Lock): The lock associated with the device for protecting shared resources.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.barrier = barrier
        self.lock = lock
        
    def run(self):
        """
        @brief The main execution loop for the device thread.
        @details This method continuously synchronizes with other threads, retrieves neighbor data,
        executes scripts, and updates its own and neighbors' sensor data. It breaks
        the loop if the supervisor indicates that there are no more neighbors (signifying simulation end).
        @block_logic Orchestrates the device's main operational cycle within the simulation.
        @pre_condition `self.device`, `self.barrier`, and `self.lock` are initialized.
        @invariant The thread waits at the barrier, processes scripts if any, and updates data for each timepoint.
        """
        while True:
            # Block Logic: Synchronize all device threads before proceeding to the next timepoint.
            # Invariant: All active device threads will reach this point before any proceed.
            self.barrier.wait()
            
            # Functional Utility: Get information about neighboring devices from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            # Block Logic: Check if the simulation should terminate.
            # Pre-condition: `neighbours` list indicates the current state of the network.
            # Invariant: The loop terminates if no neighbors are returned by the supervisor.
            if neighbours is None:
                break
            
            # Block Logic: Wait until script assignments for the current timepoint are complete.
            # Pre-condition: `self.device.timepoint_done` is an Event object.
            # Invariant: The thread proceeds only after the supervisor signals completion of script assignment.
            self.device.timepoint_done.wait()
            
            # Functional Utility: Reset the event for the next timepoint.
            self.device.timepoint_done.clear()

            # Block Logic: Acquire a lock to safely process scripts and modify shared data.
            # Pre-condition: `self.lock` is an initialized Lock object.
            # Invariant: Only one thread can execute the script processing logic for this device at a time.
            self.lock.acquire()
            
            # Block Logic: Iterate through all assigned scripts for the current timepoint and execute them.
            # Pre-condition: `self.device.scripts` contains tuples of (script, location).
            # Invariant: Each script is run with collected data and results are propagated to neighbors and itself.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Block Logic: Collect data from neighboring devices for the current location.
                # Invariant: `script_data` will contain data from all available neighbors for the given location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collect data from the current device itself for the current location.
                # Invariant: If available, the device's own data for the location is added to `script_data`.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Execute the script if there is any data to process.
                # Pre-condition: `script` is an object with a `run` method, and `script_data` is a list of data.
                # Invariant: `result` holds the output of the script's execution.
                if script_data != []:
                    result = script.run(script_data)

                    # Block Logic: Propagate the script's result to neighboring devices.
                    # Invariant: All neighbors receive the updated data for the given location.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Functional Utility: Update the current device's own data with the script's result.
                    self.device.set_data(location, result)
            self.lock.release() # Release the lock after processing all scripts.