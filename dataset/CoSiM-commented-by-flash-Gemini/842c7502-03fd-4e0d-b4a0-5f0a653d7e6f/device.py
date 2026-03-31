

"""
This module implements a distributed device simulation framework, including thread synchronization mechanisms,
device management, and script execution across a network of simulated devices.

Domain: Distributed Systems, Concurrency, Sensor Networks.
"""

from threading import Event, Thread, Lock, Semaphore

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

        Threads decrement a counter and the last thread to reach zero releases all
        waiting threads in this phase.
        Invariant: All threads must pass through this phase before any can proceed to phase 2.
        """
        with self.counter_lock:
            self.count_threads1 -= 1
            if self.count_threads1 == 0:
                # Block Logic: Release all threads waiting in phase 1.
                for i in range(self.num_threads):
                    self.threads_sem1.release()
                self.count_threads1 = self.num_threads       
        self.threads_sem1.acquire()
    
    def phase2(self):
        """
        @brief Second phase of the barrier synchronization.

        Similar to phase 1, threads decrement a counter, and the last thread releases
        all waiting threads for this phase, effectively resetting the barrier for reuse.
        Invariant: All threads must pass through phase 1 and this phase for full synchronization.
        """
        with self.counter_lock:
            self.count_threads2 -= 1
            if self.count_threads2 == 0:
                # Block Logic: Release all threads waiting in phase 2.
                for i in range(self.num_threads):
                    self.threads_sem2.release()
                self.count_threads2 = self.num_threads       
        self.threads_sem2.acquire()


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.

    This class manages sensor data, executes scripts, and synchronizes with other
    devices through a supervisor and a barrier mechanism.
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
        # @brief Event to signal that processing for a timepoint is complete.
        self.timepoint_done = Event()
        # @brief The thread responsible for running the device's main logic.
        self.thread = DeviceThread(self)

        
        # @brief List of neighboring devices for local data exchange.
        self.neighbours = []
        # @brief List of all devices in the network, used for barrier setup.
        self.alldevices = []
        # @brief Synchronization barrier for coordinating timepoints across devices.
        self.barrier = None
        # @brief List to hold worker threads for script execution.
        self.threads = []
        # @brief The maximum number of threads allowed for concurrent script execution.
        self.threads_number = 8
        # @brief List of locks, indexed by location, to protect sensor data access.
        self.locks = [None] * 100

        self.thread.start()

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up the synchronization barrier and registers all devices.

        If a barrier doesn't exist, it creates a new ReusableBarrierSem and assigns
        it to all devices in the network. It also populates the alldevices list.

        @param devices: A list of all Device objects in the network.
        """
        if self.barrier is None:
            barrier = ReusableBarrierSem(len(devices))
            self.barrier = barrier
            for d in devices:
                if d.barrier is None:
                    d.barrier = barrier
        
        for device in devices:
            if device is not None:
                self.alldevices.append(device)


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        If the location does not have a lock associated with it, a new lock is created.
        Signals that a script has been received. If no script is provided, it signals
        that the timepoint processing is done.

        @param script: The script object to be executed.
        @param location: The data location (index) the script operates on.
        """
        no_lock_for_location = 0;
        if script is not None:
            self.scripts.append((script, location))
            # Block Logic: Check if a lock already exists for the given location among other devices.
            for device in self.alldevices:
                if device.locks[location] is not None:
                    self.locks[location] = device.locks[location]
                    no_lock_for_location = 1;
                    break;
            # Block Logic: If no existing lock is found for the location, create a new one.
            if no_lock_for_location == 0:
                self.locks[location] = Lock()
            self.script_received.set()
        else:
            self.timepoint_done.set()

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


class MyThread(Thread):
    """
    @brief A worker thread that executes a script on a specific data location.

    This thread collects data from its device and neighbors, runs an assigned script,
    and then disseminates the results. It ensures data consistency using locks.
    """

    def __init__(self, device, location, script, neighbours):
        """
        @brief Initializes a new MyThread instance.

        @param device: The Device object this thread is associated with.
        @param location: The data location (index) this thread will operate on.
        @param script: The script object to be executed.
        @param neighbours: A list of neighboring Device objects.
        """
        Thread.__init__(self)
        self.device = device
        self.location = location
        self.script = script
        self.neighbours = neighbours

    def run(self):
        """
        @brief Executes the script for the assigned location.

        This method acquires a lock for the location, collects data from the device
        and its neighbors, runs the script with the collected data, and then updates
        the device and neighbor data with the results. Ensures thread safety using locks.
        """
        self.device.locks[self.location].acquire()
        script_data = []
        
        # Block Logic: Collect data from neighboring devices for the current location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Collect data from the current device for the current location.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            # Block Logic: Execute the assigned script with the collected data.
            result = self.script.run(script_data)

            # Block Logic: Disseminate the script's result to neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)
            
            # Block Logic: Update the current device's data with the script's result.
            self.device.set_data(self.location, result)
        self.device.locks[self.location].release()

class DeviceThread(Thread):
    """
    @brief The main thread for a Device, managing its operational lifecycle.

    This thread continuously fetches neighbor information from the supervisor,
    executes assigned scripts using worker threads, and synchronizes with
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
        2. Waits for a timepoint to be ready for processing.
        3. Assigns and executes scripts to worker threads for each location.
        4. Waits for all worker threads to complete.
        5. Clears the timepoint done event and synchronizes with other devices using the barrier.
        Invariant: The device processes data in discrete timepoints, synchronizing with the network
                   after each timepoint.
        """
        while True:
            # Block Logic: Fetch updated neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()

            # Block Logic: Terminate the thread if no more neighbors are returned (simulation end).
            if neighbours is None:
                break

            # Block Logic: Wait until a new timepoint's scripts are ready for execution.
            self.device.timepoint_done.wait()

            # Block Logic: Update the device's current list of neighbors.
            self.device.neighbours = neighbours

            count = 0
            # Block Logic: Iterate through assigned scripts and launch MyThread workers.
            # Only processes up to `self.device.threads_number` scripts concurrently.
            for (script, location) in self.device.scripts:
                
                if count >= self.device.threads_number:
                    break
                count = count + 1
                thread = MyThread(self.device, location, script, neighbours)
                self.device.threads.append(thread)

            # Block Logic: Start all worker threads for the current timepoint.
            for thread in self.device.threads:
                thread.start()
            # Block Logic: Wait for all worker threads to complete their execution.
            for thread in self.device.threads:
                thread.join()
            # Block Logic: Clear the list of worker threads for the next timepoint.
            self.device.threads = []

            # Block Logic: Reset the timepoint_done event and wait for all devices to synchronize.
            self.device.timepoint_done.clear()
            self.device.barrier.wait()
