



"""
@9baee79c-98df-4c5c-add4-7b5cffa12c4c/device.py
@brief Implements a single-threaded device simulation with a custom semaphore-based reusable barrier.

This module defines a `Device` class that manages sensor data and processes scripts
within a dedicated `DeviceThread`. Synchronization across devices for timepoint progression
and script execution is handled by a custom `ReusableBarrier` implementation using semaphores,
ensuring coordinated advancement of the simulation, alongside a global lock for safe data modification.
"""

from threading import Lock, Semaphore, Event, Thread

class ReusableBarrier(object):
    """
    @brief A reusable N-thread barrier implemented using a Lock and two Semaphores.

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
                i = 0
                while i < self.num_threads:
                    threads_sem.release() 
                    i += 1                
                count_threads[0] = self.num_threads  
        threads_sem.acquire() 
                              
                              

class Device(object):
    """
    @brief Represents a simulated device managing sensor data and sequential script execution.

    Each Device instance is responsible for its unique ID, sensor readings,
    and a reference to the supervisor. It processes scripts sequentially
    within its `DeviceThread` and participates in global synchronization
    through a `ReusableBarrier`. A global `Lock` is also managed for data modification.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        Sets up device-specific attributes such as ID, sensor data, and supervisor reference.
        It also initializes internal state for script management, event signaling,
        and thread management. The global barrier (`barrier`) and lock (`lock`) are
        initialized to None and expected to be set up by the `setup_devices` method
        of a coordinating device (typically `device_id == 0`).

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's initial sensor readings.
        @param supervisor: A reference to the supervisor object managing the device network.
        """
        
        self.barrier = None
        self.lock = None
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()
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
        @brief Configures the global synchronization resources (barrier and lock).

        This method identifies a coordinating device (typically `device_id == 0`).
        If this device is the coordinating device and the barrier hasn't been set up yet,
        it initializes a global `ReusableBarrier` (sized for all devices) and a global `Lock`.
        These instances are then assigned to all devices in the simulation to ensure
        consistent synchronization and data access control across the network.

        @param devices: A list of all Device instances participating in the simulation.
        """



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
        @brief Sets or updates sensor data for a specific location, ensuring thread safety.

        Access to modify the sensor data is protected by `self.lock` (a global lock)
        to prevent race conditions during concurrent write operations from multiple devices.

        @param location: The key identifying the sensor data to update.
        @param data: The new data value to set for the specified location.
        """
        with self.lock:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's main processing thread.

        This method waits for the device's main `DeviceThread` to complete
        its execution, ensuring a clean and orderly shutdown.
        """
        
        self.thread.join()


class DeviceThread(Thread):
    """
    @brief Manages the execution lifecycle of a Device.

    This thread is responsible for continuously fetching neighbor information,
    processing assigned scripts sequentially, and participating in global
    synchronization through a `ReusableBarrier`. It operates as the single
    processing unit for its associated `Device` instance.
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
        4.  Iterates through all assigned scripts for the current timepoint:
            a.  Acquires a global lock to ensure exclusive access during script processing for this device.
            b.  Collects relevant data from neighbors and the device's own sensors.
            c.  Executes the script with the collected data.
            d.  Disseminates the processed result by updating the sensor data of neighbors and the device itself.
            e.  Releases the global lock.
        5.  Clears the timepoint completion event, preparing for the next timepoint.
        6.  Participates in a global `ReusableBarrier` synchronization, ensuring all devices
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

            # Block Logic: Processes each assigned script sequentially.
            # Invariant: Each script operates on a specific 'location' and its collected data.
            for (script, location) in self.device.scripts:
                # Block Logic: Acquires a global lock for exclusive access during script processing.
                # Functional Utility: Prevents race conditions and ensures atomic updates to shared resources
                #                      when processing scripts.
                self.device.lock.acquire()
                script_data = []
                
                # Block Logic: Gathers relevant sensor data from neighboring devices for the current script's location.
                # Functional Utility: Collects necessary input for the script based on the current network state.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Includes the device's own sensor data for the current script's location.
                # Functional Utility: Ensures the script considers the device's local state.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Block Logic: Executes the assigned script with the aggregated data.
                    # Architectural Intent: Decouples computational logic from data management,
                    #                      allowing dynamic script execution based on current data.
                    result = script.run(script_data)

                    # Block Logic: Disseminates the computed result to neighboring devices.
                    # Functional Utility: Propagates state changes across the network as a result of script execution.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    # Block Logic: Updates the device's own sensor data with the computed result.
                    # Functional Utility: Reflects local state changes due to script processing.
                    self.device.set_data(location, result)
                # Block Logic: Releases the global lock after script processing is complete.
                # Functional Utility: Allows other devices to acquire the lock and process their scripts.
                self.device.lock.release()
            
            # Block Logic: Clears the timepoint completion event, preparing for the next timepoint.
            # Functional Utility: Resets the event for a new cycle of timepoint synchronization.
            self.device.timepoint_done.clear()
            # Block Logic: Global synchronization point for all devices across the simulation.
            # Functional Utility: Ensures all devices have completed their processing for the current
            #                      timepoint before advancing to the next.
            self.device.barrier.wait()
