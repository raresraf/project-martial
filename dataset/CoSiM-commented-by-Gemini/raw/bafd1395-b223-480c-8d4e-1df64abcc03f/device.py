


"""
@file device.py
@brief Implements simulated device functionality, including synchronization primitives and worker threads.
This module provides a `Device` class to represent an individual simulated device, a `ReusableBarrierSem`
for coordinating multiple threads, and `DeviceThread` for concurrent script execution and data processing
within a simulated environment, potentially for sensor networks or distributed systems.
"""


from threading import Event, Thread, Lock, Semaphore


class ReusableBarrierSem():
    """
    @brief Implements a reusable barrier synchronization mechanism using semaphores and locks.
    This barrier allows a specified number of threads to wait until all threads have reached
    a certain point, then releases them, and can be reused for subsequent synchronization points.
    """
    
    
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrierSem.
        @param num_threads: The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        self.count_threads1 = self.num_threads
        self.count_threads2 = self.num_threads
        self.counter_lock = Lock()               # Protects access to thread counters
        self.threads_sem1 = Semaphore(0)         # Semaphore for first phase of barrier
        self.threads_sem2 = Semaphore(0)         # Semaphore for second phase of barrier
    
    def wait(self):
        """
        @brief Blocks the calling thread until all `num_threads` have reached this point.
        This method orchestrates the two phases of the reusable barrier.
        """
        self.phase1()
        self.phase2()
    
    def phase1(self):
        """
        @brief First phase of the barrier.
        Threads acquire `counter_lock` to decrement `count_threads1`. The last thread
        reaching this phase releases all threads waiting on `threads_sem1`.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            # Invariant: If this is the last thread in phase 1, release all waiting threads.
            if self.count_threads1 == 0:
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads # Reset for reuse
        
        self.threads_sem1.acquire() # Block until released by the last thread of phase 1
    
    def phase2(self):
        """
        @brief Second phase of the barrier.
        Threads acquire `counter_lock` to decrement `count_threads2`. The last thread
        reaching this phase releases all threads waiting on `threads_sem2`.
        This design prevents a "lost wakeup" problem and allows for barrier reuse.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            # Invariant: If this is the last thread in phase 2, release all waiting threads.
            if self.count_threads2 == 0:
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads # Reset for reuse
        
        self.threads_sem2.acquire() # Block until released by the last thread of phase 2



class Device(object):
    """
    @brief Represents a simulated device in a distributed system, handling sensor data,
    script execution, and synchronization with other devices.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary containing sensor data for various locations.
        @param supervisor: A reference to the supervisor entity managing the devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = []
        self.timepoint_done = Event() # Event to signal completion of a timepoint's tasks

        self.barrier = None # ReusableBarrierSem for inter-device synchronization
        self.lock_location = None # List of Locks for protecting sensor data access by location
        self.lock_script = Lock() # Lock for protecting access to the scripts list and available status
        self.lock_neighbours = Lock() # Lock for protecting access to the neighbours list

        self.available = [] # List to track availability of script slots
        self.neighbours = None # List of neighboring devices
        self.init_done = Event() # Event to signal completion of device initialization
        self.update_neighbours = True # Flag to indicate if neighbours list needs updating
        
        self.threads = []
        # Functional Utility: Spawns 8 DeviceThread instances for concurrent script execution.
        for i in range(8):
            self.threads.append(DeviceThread(self))
            self.threads[i].start()

    def __str__(self):
        """
        @brief Provides a string representation of the Device.
        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the device's synchronization mechanisms and shared resources.
        Only device_id 0 initializes the barrier and location locks, then signals
        other devices to use these shared instances.
        @param devices: A list of all Device instances in the system.
        Pre-condition: This method is expected to be called by all devices during setup.
        Post-condition: The device's `barrier` and `lock_location` are initialized
                        or linked to the master device's instances.
        """
        # Block Logic: Master device (device_id == 0) initializes shared synchronization primitives.
        if self.device_id == 0:
            self.barrier = ReusableBarrierSem(len(devices))
            self.lock_location = []
            for _ in range(200):
                self.lock_location.append(Lock())
            self.init_done.set() # Signal that initialization is complete
        # Block Logic: Other devices wait for the master device's initialization and then link to its shared instances.
        else:
            for device in devices:
                if device.device_id == 0:
                    device.init_done.wait() # Wait for master device to finish initialization
                    self.barrier = device.barrier
                    self.lock_location = device.lock_location
                    return

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific location on the device.
        If a script is provided, it's added to the device's script list. If `script` is None,
        it signals the end of a timepoint's assignments, triggering a barrier wait and device reset.
        @param script: The script object to assign, or None to signal a timepoint completion.
        @param location: The location associated with the script.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.available.append(True) # Mark script slot as available
        else:
            # Block Logic: If no script is assigned, it indicates a timepoint boundary.
            # The device waits on the barrier, resets its state, and signals timepoint completion.
            self.barrier.wait() # Wait for all threads/devices to reach this timepoint
            self.reset() # Reset device state for the next timepoint
            self.timepoint_done.set() # Signal completion of this timepoint

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The location for which to retrieve sensor data.
        @return The sensor data if `location` exists, otherwise `None`.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets sensor data for a given location if it exists.
        @param location: The location for which to set sensor data.
        @param data: The new sensor data to set.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down all worker threads associated with this device.
        Pre-condition: Assumes threads have a mechanism to gracefully exit their run loop.
        """
        for thread in self.threads:
            if thread.isAlive(): # Check if thread is still running
                thread.join() # Wait for thread to finish

    def reset(self):
        """
        @brief Resets the device's state for a new timepoint.
        This involves re-enabling neighbor updates and making all script slots available again.
        """
        with self.lock_neighbours: # Protects access to update_neighbours flag
            self.update_neighbours = True # Force update of neighbours in the next cycle
        for i in range(len(self.available)):
            self.available[i] = True # Mark all script slots as available


class DeviceThread(Thread):
    """
    @brief A worker thread responsible for executing scripts and processing data
    on behalf of a `Device` instance. Each `Device` can have multiple `DeviceThread`s.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread.
        @param device: The `Device` instance this thread belongs to.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        This loop continuously processes scripts assigned to the device,
        manages neighbor data, acquires and releases locks for data consistency,
        and waits for timepoint synchronization.
        """
        while True:
            # Block Logic: Acquire lock to check and update neighbors if necessary.
            # This ensures that the thread has the latest view of the network topology.
            self.device.lock_neighbours.acquire()

            if self.device.update_neighbours:
                self.device.neighbours = self.device.supervisor.get_neighbours()
                self.device.update_neighbours = False

            self.device.lock_neighbours.release()

            neighbours = self.device.neighbours

            # Invariant: If `neighbours` is None, it signals that the device is shutting down,
            # and the thread should exit.
            if neighbours is None:
                break

            # Block Logic: Iterate through assigned scripts, ensuring only available slots are processed.
            for (script, location) in self.device.scripts:

                self.device.lock_script.acquire()

                index = self.device.scripts.index((script, location))
                if self.device.available[index]:
                    self.device.available[index] = False # Mark script slot as busy
                else:
                    self.device.lock_script.release()
                    continue

                self.device.lock_script.release()

                # Block Logic: Acquire a location-specific lock to ensure exclusive access to sensor data
                # for the current location during script execution and data updates.
                self.device.lock_location[location].acquire()

                script_data = []
                
                # Functional Utility: Collect data from neighboring devices and the current device
                # for the specified location to provide as input to the script.
                for device in neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    # Functional Utility: Execute the assigned script with the collected data.
                    result = script.run(script_data)

                    # Functional Utility: Update data on neighboring devices and the current device
                    # with the result of the script execution.
                    for device in neighbours:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                self.device.lock_location[location].release()

            # Block Logic: Wait for the device to signal that all timepoint tasks are done
            # before starting the next cycle. This ensures synchronization across all threads
            # within the device.
            self.device.timepoint_done.wait()
