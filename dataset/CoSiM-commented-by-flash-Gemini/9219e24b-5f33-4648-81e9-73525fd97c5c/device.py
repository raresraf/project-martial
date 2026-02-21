


"""
@9219e24b-5f33-4648-81e9-73525fd97c5c/device.py
@brief Implements a single-threaded device simulation with condition variable-based barrier synchronization.

This module defines a `Device` that manages its own sensor data and processes scripts
within a dedicated `DeviceThread`. Synchronization across devices for timepoint progression
and script execution is managed via a `CondBarrier`, a condition variable-based mechanism,
along with a global lock to ensure data consistency during concurrent updates across devices.
"""

from threading import Event, Thread, Condition, Lock


class Device(object):
    """
    @brief Represents a simulated device managing sensor data and script execution.

    Each Device instance is responsible for its unique ID, sensor readings,
    and a reference to the supervisor. It processes scripts sequentially
    within its `DeviceThread` and participates in global synchronization
    through a `CondBarrier` and a global `Lock`.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        Sets up device-specific attributes such as ID, sensor data, and supervisor reference.
        It also initializes internal state for script management, event signaling,
        and thread management. Synchronization primitives (`barr`, `lock`) are
        initialized to None and expected to be set up by the `setup_devices` method
        of a coordinating device (typically `device_id == 0`).

        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing the device's initial sensor readings.
        @param supervisor: A reference to the supervisor object managing the device network.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script has been assigned to the device, or timepoint is done.
        self.script_received = Event()
        # List to store (script, location) tuples for execution.
        self.scripts = []
        # Event to signal that the device has completed processing for the current timepoint.
        self.timepoint_done = Event()
        # The dedicated thread for this device's execution logic.
        self.thread = DeviceThread(self)
        # Global barrier for synchronizing all devices. Set by setup_devices.
        self.barr = None
        # Global lock for critical sections across devices. Set by setup_devices.
        self.lock = None
        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures global synchronization resources and starts the device's main thread.

        This method handles the initialization and propagation of shared synchronization
        primitives. If this device is the first in the `devices` list and has not
        yet had its barrier initialized, it creates a new `CondBarrier` (sized for
        all devices) and a global `Lock`. These instances are then assigned to all
        devices in the simulation to ensure consistent synchronization across the network.
        Finally, it starts its own `DeviceThread` to begin processing.

        @param devices: A list of all Device instances participating in the simulation.
        """
        
        
        
		
        if devices[0].barr is None and self.device_id == devices[0].device_id:
                # Create a new CondBarrier for all participating devices.
                barr = CondBarrier(len(devices))
                # Assign the created barrier to all devices.
                for i in devices:
                        i.barr = barr
        # Create a single global lock to be shared among all devices.
        lock = Lock()
        # Assign the global lock to all devices.
        for d in devices:
                d.lock = lock 

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be processed or signals completion of a timepoint.

        If a `script` is provided, it is appended to the device's list of `scripts`.
        If `script` is None, it signals the completion of script assignments for the
        current timepoint by setting both the `timepoint_done` and `script_received` events.

        @param script: The script object to be executed, or None to signal timepoint completion.
        @param location: The data location (e.g., sensor ID) the script operates on.
        """
        
        if script is not None:
            self.scripts.append((script, location))
        else:
            # When script is None, it means the assignment phase is complete for this timepoint.
            # Both events are set to release any waiting threads/processes.
            self.timepoint_done.set()
            self.script_received.set()

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

class CondBarrier():
    """
    @brief A reusable synchronization barrier implemented using a Condition variable.

    This barrier allows a fixed number of threads to wait until all have reached
    a specific point of execution before any are allowed to proceed. It is designed
    to be reusable for multiple synchronization points within a simulation.
    """
	    def __init__(self, num_threads):
	        """
	        @brief Initializes a new CondBarrier instance.
	
	        Sets the total number of threads that must reach the barrier and initializes
	        the internal counter and condition variable.
	
	        @param num_threads: The total number of threads expected to synchronize at this barrier.
	        """		self.num_threads = num_threads
		self.count_threads = self.num_threads
		self.cond = Condition()

	    def wait(self):
	        """
	        @brief Blocks the calling thread until all registered threads have reached the barrier.
	
	        Acquires a condition lock, decrements the internal count of threads yet to reach the barrier.
	        If this thread is the last to arrive, it notifies all waiting threads and
	        resets the barrier for reuse. Otherwise, it waits until signaled by the last thread.
	        """		self.cond.acquire()
		self.count_threads -= 1
		if self.count_threads == 0:
			self.cond.notify_all()
			self.count_threads = self.num_threads
		else:
			self.cond.wait()
		self.cond.release()

    def wait(self):
        """
        Causes the calling thread to wait until all `num_threads` have reached this barrier.
        The barrier automatically resets itself for subsequent uses.
        """
        self.cond.acquire() # Acquire the lock associated with the condition variable.
        self.count_threads -= 1 # Decrement the count of waiting threads.
        if self.count_threads == 0:
            # If this is the last thread, wake up all waiting threads.
            self.cond.notify_all()
            # Reset the counter for the next cycle of the barrier.
            self.count_threads = self.num_threads
        else:
            # If not the last thread, wait until notified.
            self.cond.wait()
        self.cond.release() # Release the lock.


class DeviceThread(Thread):
    """
    @brief Manages the execution lifecycle of a Device.

    This thread is responsible for continuously fetching neighbor information,
    processing assigned scripts, and participating in global synchronization
    through a `CondBarrier`. It operates as a single processing unit for
    its associated `Device` instance.
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
        3.  Waits for the supervisor to signal that a new timepoint has begun and scripts are ready.
        4.  Iterates through all assigned scripts for the current timepoint:
            a.  Acquires a global lock to ensure exclusive access during script processing for this device.
            b.  Collects relevant data from neighbors and the device's own sensors.
            c.  Executes the script with the collected data.
            d.  Disseminates the processed result by updating the sensor data of neighbors and the device itself.
            e.  Releases the global lock.
        5.  Clears the timepoint completion event, preparing for the next timepoint.
        6.  Participates in a global `CondBarrier` synchronization, ensuring all devices
            are synchronized before advancing to the next timepoint.
        """
        while True:
            # Block Logic: Fetches the current set of active neighbors for data exchange.
            # Functional Utility: Dynamically updates the device's awareness of its network topology.
            neighbours = self.device.supervisor.get_neighbours()
            # Invariant: If no neighbors are returned, the simulation for this device is complete.
            if neighbours is None:
                break

            # Block Logic: Waits for the supervisor to signal the start of a new timepoint and availability of scripts.
            # Functional Utility: Orchestrates the progression of simulation timepoints, ensuring data consistency.
            self.device.timepoint_done.wait()

            # Block Logic: Processes each assigned script using data from local sensors and neighbors.
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

                # Execute the script if there is data to process.
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
            self.device.barr.wait()
