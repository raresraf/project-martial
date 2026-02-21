


"""
@97ead1a1-dff2-4c17-80f1-be290a7a3293/device.py
@brief Implements a multi-threaded device simulation with a custom semaphore-based reusable barrier.

This module defines a `Device` class that manages sensor data and orchestrates script execution
through a main `DeviceThread`. For each script, a dedicated `ScriptThread` is spawned
to process data. Synchronization across devices for timepoint progression is handled
by a custom `ReusableBarrier` implementation using semaphores, ensuring coordinated
advancement of the simulation.
"""

from threading import Event, Thread, Lock, Semaphore

class ReusableBarrier:
    """
    @brief A reusable N-thread barrier implemented using semaphores.

    This barrier allows a specified number of threads to synchronize in two phases.
    Threads wait at the barrier until all have arrived, then are released. The barrier
    can then be reused for subsequent synchronization points.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes a ReusableBarrier instance.

        Sets the total number of threads, initializes two counters for the two phases
        of the barrier, a lock for protecting these counters, and two semaphores
        for releasing threads in each phase.

        @param num_threads: The total number of threads expected to synchronize at this barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 
        self.threads_sem1 = Semaphore(0)         
        self.threads_sem2 = Semaphore(0)         

    def wait(self):
        """
        @brief Blocks the calling thread until all threads have completed both phases of the barrier.

        This method coordinates two phases of synchronization. A thread first completes
        the first phase, and then the second. Only when all threads have completed
        a phase are they all released from that phase.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)

    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single phase of the barrier synchronization.

        A thread acquires the count lock, decrements the thread count for the current phase.
        If it's the last thread to arrive, it releases all waiting threads from the semaphore
        and resets the count. Otherwise, it waits on the semaphore until signaled.

        @param count_threads: The list containing the current count of threads for this phase.
        @param threads_sem: The semaphore used to release threads for this phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for i in range(self.num_threads):
                    threads_sem.release()        
                count_threads[0] = self.num_threads  
        threads_sem.acquire()

class Device(object):
    """
    @brief Represents a simulated device managing sensor data and script execution.

    Each Device instance is responsible for its unique ID, sensor readings,
    and a reference to the supervisor. It processes scripts sequentially
    within its `DeviceThread` and participates in global synchronization
    through a `ReusableBarrier`.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        Sets up device-specific attributes such as ID, sensor data, and supervisor reference.
        It also initializes internal state for script management, event signaling,
        and thread management. The global barrier (`barrier`) is initialized to None
        and is expected to be set up by the `setup_devices` method of a coordinating device.

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's initial sensor readings.
        @param supervisor: A reference to the supervisor object managing the device network.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
        self.barrier = None
        self.scripts = []
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures global synchronization resources.

        If this device is the one with `device_id == 0`, it initializes a global
        `ReusableBarrier` (sized for all devices) and assigns it to its own `barrier`
        attribute. It then propagates this barrier instance to all other devices
        in the simulation that do not yet have one, ensuring consistent synchronization
        across the network.

        @param devices: A list of all Device instances participating in the simulation.
        """
        
        
        
        
        if self.device_id == 0:
            bariera = ReusableBarrier(len(devices))
            self.barrier = bariera
            for dev in devices:
                if dev.barrier is None:
                    dev.barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be processed or signals completion of a timepoint.

        If a `script` is provided, it is appended to the device's internal `scripts` list,
        and the `script_received` event is set.
        If `script` is None, it signals that all scripts for the current timepoint
        have been received by setting the `timepoint_done` event.

        @param script: The script object to be executed, or None to signal timepoint completion.
        @param location: The data location (e.g., sensor ID) the script operates on.
        """
        
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specific location.

        @param location: The key identifying the sensor data to retrieve.
        @return: The sensor data at the specified location, or None if not found.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specific location.

        Updates the sensor data if the location already exists in the device's
        sensor data dictionary.

        @param location: The key identifying the sensor data to update.
        @param data: The new data value to set for the specified location.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's main processing thread.

        This method waits for the device's main `DeviceThread` to complete
        its execution, ensuring a clean and orderly shutdown.
        """
        
        self.thread.join()


class ScriptThread(Thread):
    """
    @brief Represents a worker thread that executes a single script for a Device.

    Each `ScriptThread` is spawned by the `DeviceThread` to process a specific script.
    Its responsibility is to gather relevant data from neighbors and the device itself,
    execute the script, and then disseminate the results.
    """
    
    def __init__(self, device, script, location, neighbours):
        """
        @brief Initializes a new ScriptThread instance.

        Associates the thread with its parent `Device` instance, the specific
        `script` to execute, the `location` it pertains to, and a snapshot
        of the current `neighbours` list.

        @param device: The parent Device instance this script thread belongs to.
        @param script: The script object to be executed.
        @param location: The data location (e.g., sensor ID) the script operates on.
        @param neighbours: A list of neighboring Device instances relevant for this script's execution.
        """
        Thread.__init__(self)
        self.device = device


        self.location = location
        self.script = script
        self.neighbours = neighbours
        self.script_data = []

    def run(self):
        """
        @brief Executes the assigned script, gathering data and disseminating results.

        This method performs the core logic of a script worker:
        1.  Collects relevant sensor data from all `neighbours` and the worker's
            `device` itself for the specified `location`.
        2.  If data is available, it executes the provided `script` with the
            collected data.
        3.  Disseminates the `result` of the script execution by updating the
            sensor data of all `neighbours` and the worker's own `device` at
            the specified `location`.
        """
        # Block Logic: Gathers relevant sensor data from all specified neighbors for the current script's location.
        # Functional Utility: Collects necessary input for the script based on the current network state.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                self.script_data.append(data)
        
        # Block Logic: Includes the device's own sensor data for the current script's location.
        # Functional Utility: Ensures the script considers the device's local state.
        data = self.device.get_data(self.location)
        if data is not None:
            self.script_data.append(data)

        if self.script_data != []:
            # Block Logic: Executes the assigned script with the aggregated data.
            # Architectural Intent: Decouples computational logic from data management,
            #                      allowing dynamic script execution based on current data.
            result = self.script.run(self.script_data)
            
            # Block Logic: Disseminates the computed result to neighboring devices.
            # Functional Utility: Propagates state changes across the network as a result of script execution.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Block Logic: Updates the device's own sensor data with the computed result.
            # Functional Utility: Reflects local state changes due to script processing.
            self.device.set_data(self.location, result)
        self.script_data = []

class DeviceThread(Thread):
    """
    @brief Orchestrates the execution of `ScriptThread`s for a Device at each timepoint.

    This thread acts as the main control unit for a `Device`, responsible for
    fetching neighborhood information and spawning a `ScriptThread` for each
    assigned script. It ensures all script threads complete their tasks and
    then participates in a global barrier synchronization before
    proceeding to the next timepoint.
    """
    

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        Sets up the thread with a descriptive name and associates it with
        the Device instance it will manage.

        @param device: The Device instance this thread is responsible for.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        This loop continuously performs the following steps for each simulation timepoint:
        1.  Fetches the current set of neighboring devices from the supervisor.
        2.  Terminates if no neighbors are found, signifying the end of the simulation for this device.
        3.  Waits for the supervisor to signal that a new timepoint has begun.
        4.  Spawns a `ScriptThread` for each assigned script to process them concurrently.
        5.  Waits for all `ScriptThread`s to complete their execution for the current timepoint.
        6.  Clears the timepoint completion event, preparing for the next timepoint.
        7.  Participates in a global `ReusableBarrier` synchronization, ensuring all devices
            are synchronized before advancing to the next timepoint.
        """
        while True:
            # Block Logic: Fetches the current set of active neighbors for data exchange.
            # Functional Utility: Dynamically updates the device's awareness of its network topology.
            neighbours = self.device.supervisor.get_neighbours()
            # Invariant: If no neighbors are returned, the simulation for this device is complete.
            if neighbours is None:
                break
            
            # Block Logic: Waits for the supervisor to signal the start of a new timepoint.
            # Functional Utility: Orchestrates the progression of simulation timepoints.
            self.device.timepoint_done.wait()
            threads = []
            
            # Block Logic: Spawns a ScriptThread for each assigned script to process them concurrently.
            # Architectural Intent: Leverages multi-threading to parallelize the execution of scripts
            #                      for the current timepoint.
            for (script, location) in self.device.scripts:
                thrScript = ScriptThread(self.device, script, location, neighbours)
                threads.append(thrScript)

            # Block Logic: Starts all created ScriptThread instances.
            for thread in threads:
                thread.start()
            # Block Logic: Waits for all spawned ScriptThread instances to complete their assigned tasks.
            # Functional Utility: Ensures all scripts for the current timepoint are fully processed
            #                      before signaling completion and moving to the next step.
            for thread in threads:
                thread.join()
            
            # Block Logic: Clears the timepoint completion event, preparing for the next timepoint.
            # Functional Utility: Resets the event for a new cycle of timepoint synchronization.
            self.device.timepoint_done.clear()
            
            # Block Logic: Global synchronization point for all devices across the simulation.
            # Functional Utility: Ensures all devices have completed their processing for the current
            #                      timepoint before advancing to the next.
            self.device.barrier.wait()
