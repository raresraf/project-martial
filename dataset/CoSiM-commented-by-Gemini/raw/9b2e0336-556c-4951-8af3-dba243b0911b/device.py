"""
@file device.py
@brief This file defines a simulated device and its associated threading model for a distributed sensing and computation environment.
@details The script models a network of devices that can execute scripts on sensor data. 
         Each device operates concurrently, with multiple internal threads for parallel processing. 
         Synchronization is managed through a combination of barriers, locks, and events to coordinate actions both within a single device and across the entire network.
"""


import cond_barrier
from threading import Event, Thread, Lock


class Device(object):
    """
    @brief Represents a single device in the simulated network.
    @details A Device manages its own sensor data, executes assigned scripts, and communicates with neighboring devices.
             It uses multiple threads to process scripts in parallel and synchronizes with other devices at each timepoint.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for the device.
        @param sensor_data: A dictionary representing the device's local sensor readings, keyed by location.
        @param supervisor: An object responsible for global coordination, such as providing neighbor information.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()


        self.scripts = []
        self.timepoint_done = Event()
        self.threads = []

        self.neighbourhood = None
        # Locks to ensure exclusive access to data at specific locations.
        self.map_locks = {}
        # Barrier for synchronizing threads within this device.
        self.threads_barrier = None
        # Global barrier for synchronizing all devices in the network.
        self.barrier = None
        self.counter = 0
        # Lock for thread-safe access to the script counter.
        self.threads_lock = Lock()

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs collective setup for a group of devices.
        @details The device with ID 0 acts as a coordinator to initialize a shared global barrier for all devices.
                 It also initializes and starts the internal worker threads for this device.
        @param devices: A list of all Device objects in the simulation.
        """
        
        if self.device_id == 0:
            num_threads = len(devices)
            
            # The coordinator device (ID 0) creates a global barrier for all threads in the simulation.
            # The barrier size is the number of devices multiplied by the number of threads per device (8).
            self.barrier = cond_barrier.ReusableBarrier(num_threads * 8)

            for device in devices:
                device.barrier = self.barrier
                device.map_locks = self.map_locks

        
        # Each device gets its own barrier to synchronize its 8 worker threads.
        self.threads_barrier = cond_barrier.ReusableBarrier(8)
        for i in range(8):
            self.threads.append(DeviceThread(self, i, self.threads_barrier))

        
        for thread in self.threads:
            thread.start()


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by this device.
        @param script: The script object to be executed.
        @param location: The location context for the script execution.
        """
        

        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script signals the end of script assignments for the current timepoint.
            self.timepoint_done.set()

        
        # Create a lock for a location if it doesn't exist, ensuring data integrity for that location.
        if location not in self.map_locks:
            self.map_locks[location] = Lock()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.
        @param location: The location for which to retrieve data.
        @return: The sensor data at the specified location, or None if not available.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data at a given location.
        @param location: The location at which to update data.
        @param data: The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for all its worker threads to complete.
        """
        

        
        for thread in self.threads:
            thread.join()


class DeviceThread(Thread):
    """
    @brief A worker thread for a Device.
    @details Each Device runs multiple DeviceThreads to process scripts concurrently. 
             These threads synchronize at different stages of the simulation cycle.
    """
    

    def __init__(self, device, id, barrier):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id = id
        self.thread_barrier = barrier

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
        @details This loop coordinates fetching neighborhood data, waiting for simulation signals,
                 processing assigned scripts, and synchronizing with other threads and devices.
        """
        while True:
            
            # Pre-condition: At the beginning of each timepoint, one thread (id 0) is responsible for updating the device's neighborhood.
            if self.id == 0:
                
                self.device.neighbourhood = self.device.supervisor.get_neighbours()

            
            # Invariant: All threads of a device wait here until the neighborhood information is fetched.
            self.thread_barrier.wait()

            if self.device.neighbourhood is None:
                # A None neighborhood is a signal to shut down.
                break 

            
            # Invariant: All threads wait until the main process signals that all scripts for the current timepoint have been assigned.
            self.device.timepoint_done.wait()

            
            # This loop processes the scripts assigned for the current timepoint.
            while True:
                
                # Block Logic: This section distributes scripts among the worker threads in a thread-safe manner
                # using a shared counter.
                with self.device.threads_lock:
                    if self.device.counter == len(self.device.scripts):
                        # No more scripts to process for this timepoint.
                        break
                    (script, location) = self.device.scripts[self.device.counter]
                    self.device.counter = self.device.counter + 1
                


                # Acquire a lock for the specific location to ensure exclusive access to the data.
                self.device.map_locks[location].acquire()
                script_data = []

                # Block Logic: Gathers data from all neighboring devices for the script's location.
                for device in self.device.neighbourhood:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)

                # Also gather data from the current device.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                if script_data != []:
                    


                    # Execute the script with the aggregated data.
                    result = script.run(script_data)

                    # Block Logic: The result of the script is broadcast back to all neighboring devices and the current device.
                    for device in self.device.neighbourhood:
                        device.set_data(location, result)
                    
                    self.device.set_data(location, result)

                self.device.map_locks[location].release()

            
            # Invariant: All threads from all devices synchronize here, marking the end of the computation for the current timepoint.
            self.device.barrier.wait()
            # Post-condition: After global synchronization, one thread (id 0) resets the state for the next timepoint.
            if self.id == 0:
                
                
                self.device.counter = 0
                self.device.timepoint_done.clear()