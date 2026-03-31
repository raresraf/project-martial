

"""
This module implements a simulated device within a distributed sensor network,
handling data management, script execution, and synchronization using a reusable barrier.

Domain: Distributed Systems, Concurrency, Sensor Networks.
"""

from threading import enumerate, Event, Thread, Lock, Semaphore

class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier for thread synchronization using semaphores.

    This barrier allows a fixed number of threads to wait until all have reached a certain point
    before any are allowed to proceed. It uses a two-phase approach to ensure reusability
    without deadlocks.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrierSem with a specified number of threads.

        @param num_threads: The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # @brief Counter for the first phase of the barrier.
        self.count_threads1 = self.num_threads

        # @brief Counter for the second phase of the barrier.
        self.count_threads2 = self.num_threads
        
        # @brief Lock to protect access to the thread counters.
        self.counter_lock = Lock()       
        # @brief Semaphore for the first phase of waiting threads.
        self.threads_sem1 = Semaphore(0) 
        # @brief Semaphore for the second phase of waiting threads.
        self.threads_sem2 = Semaphore(0) 

    def wait(self):
        """
        @brief Blocks until all participating threads have reached this point.

        This method orchestrates the two phases of the barrier to ensure all threads
        synchronize before proceeding, allowing for reusability.
        """
        self.phase1()
        self.phase2()

    def phase1(self):
        """
        @brief First phase of the barrier synchronization.

        Threads decrement a counter. The last thread to reach zero releases all
        waiting threads in this phase and resets the counter for the *second* phase.
        Invariant: All threads must pass through this phase before any can proceed to phase 2.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Block Logic: Release all threads waiting in phase 1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
            # Inline: Reset the counter for the second phase, preparing for reuse.
            self.count_threads2 = self.num_threads
         
        self.threads_sem1.acquire()

    def phase2(self):
        """
        @brief Second phase of the barrier synchronization.

        Similar to phase 1, threads decrement a counter. The last thread releases
        all waiting threads for this phase and resets the counter for the *first* phase,
        effectively resetting the barrier for reuse.
        Invariant: All threads must pass through phase 1 and this phase for full synchronization.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Block Logic: Release all threads waiting in phase 2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
            # Inline: Reset the counter for the first phase, preparing for reuse.
            self.count_threads1 = self.num_threads

        self.threads_sem2.acquire()

class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.

    This class manages sensor data, executes scripts, and interacts with a supervisor
    within a network. It also participates in a synchronization barrier.
    """


    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's sensor readings.
                            Keys are locations, values are data.
        @param supervisor: A reference to the supervisor object that manages the network.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # @brief Event to signal that a script has been assigned to the device.
        self.script_received = Event()
        # @brief List to store assigned scripts and their locations.
        self.scripts = []
        # @brief The thread responsible for running the device's main logic.
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
        @brief Sets up the synchronization barrier for the device network.

        Device with ID 0 creates the barrier, and other devices reference it.

        @param devices: A list of all Device objects in the network.
        """
        if self.device_id == 0:
            # Block Logic: Device 0 initializes the shared barrier.
            self.barrier = ReusableBarrierSem(len(devices))
        else:
            # Block Logic: Other devices reference the barrier created by Device 0.
            for device in devices:
                if device.device_id == 0:
                    self.barrier = device.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If a script is provided, it's added to the list of scripts.
        If no script is provided, it signals that script assignment is complete for the timepoint.

        @param script: The script object to be executed.
        @param location: The data location (index) the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            self.script_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location: The data location (index) to retrieve data from.
        @return The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location.

        @param location: The data location (index) to set data for.
        @param data: The new data value to be set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data
    def shutdown(self):
        """
        @brief Shuts down the device's main thread, ensuring proper termination.
        """
        self.thread.join()

class Node(Thread):
    """
    @brief A worker thread class responsible for executing a single script.

    This class encapsulates the execution of a script with provided data and
    stores its result, allowing for parallel script execution.
    """

    def __init__(self, script, script_data):
        """
        @brief Initializes a new Node (worker thread) instance.

        @param script: The script object to be executed.
        @param script_data: The input data for the script.
        """
        Thread.__init__(self)
        self.script = script
        self.script_data = script_data
        # @brief Stores the result of the script execution.
        self.result = None
         
    def run(self):
        """
        @brief Executes the assigned script with its data and stores the result.
        """
        self.result = self.script.run(self.script_data)

    def join(self):
        """
        @brief Waits for the thread to terminate and returns the script and its result.

        @return A tuple containing the script object and its execution result.
        """
        Thread.join(self)
        return (self.script, self.result)


class DeviceThread(Thread):
    """
    @brief The main thread for a Device, managing its operational lifecycle.

    This thread continuously fetches neighbor information from the supervisor,
    executes assigned scripts using worker threads (Nodes), and synchronizes with
    other devices via a barrier after each timepoint.
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
        @brief The main execution loop for the device thread.

        This loop continuously performs the following actions:
        1. Fetches current neighbor information from the supervisor.
        2. Waits for scripts to be assigned for the current timepoint.
        3. Collects data from neighbors and the device itself for each script.
        4. Executes scripts concurrently using `Node` worker threads.
        5. Gathers results from the worker threads.
        6. Disseminates the results to the device and its neighbors.
        7. Synchronizes with other devices using the barrier.
        Invariant: The device processes data in discrete timepoints, synchronizing with the network
                   after each timepoint.
        """
        while True:
            # Block Logic: Fetch updated neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            thread_list=[] # @brief List to hold Node worker threads.
            scripts_result = {} # @brief Dictionary to store results of script execution.
            scripts_data = {} # @brief Dictionary to store data collected for each script.
            
            # Block Logic: Terminate the thread if no more neighbors are returned (simulation end).
            if neighbours is None:
                break

            string = "" # @brief Debugging or logging variable for neighbors.
            for neighbour in neighbours:
                string = string + " " + str(neighbour)
            
            # Block Logic: Wait until a new timepoint's scripts are ready for execution.
            self.device.script_received.wait()
            # Inline: Clear the event for the next timepoint.
            self.device.script_received.clear()
            
            # Block Logic: Prepare and launch Node worker threads for each assigned script.
            for (script, location) in self.device.scripts:
                script_data = []
                
                # Block Logic: Collect data from neighboring devices for the current script's location.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Collect data from the current device for the current script's location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)
                
                scripts_data[script] = script_data # Store collected data for the script.
                if script_data != []:
                    # Inline: Create and add a new Node thread for script execution.
                    nod = Node(script,script_data)
                    thread_list.append(nod)
            
            # Block Logic: Start all Node worker threads concurrently.
            for nod in thread_list:
                
                nod.start()
            
            # Block Logic: Wait for all Node worker threads to complete and collect their results.
            for nod in thread_list:
                key ,value = nod.join()
                scripts_result[key] = value
            
            # Block Logic: Disseminate the script execution results to the device and its neighbors.
            for (script, location) in self.device.scripts:
                
                if scripts_data[script] != []:
                    # Inline: Update data in neighboring devices.
                    for device in neighbours:
                        device.set_data(location, scripts_result[script])
                        
                    # Inline: Update data in the current device.
                    self.device.set_data(location, scripts_result[script])
            
            # Block Logic: Synchronize with all other devices in the network via the barrier.
            self.device.barrier.wait()
