"""
@file device.py
@brief This file defines a simulated device using a master-worker threading model with semaphore-based concurrency control
       and centralized, location-based locking.
@details The script models a network of devices that execute scripts on sensor data. Each device has a master thread
         that spawns multiple worker threads to handle script execution in parallel. The number of concurrent workers
         per device is limited by a semaphore. Data integrity across the entire system is maintained by a set of
         location-specific locks held by a designated root device, ensuring that only one script can operate on a
         given data location at a time.
"""


from threading import Event, Thread, RLock, Semaphore
from barrier import ReusableBarrierSem

class Device(object):
    """
    @brief Represents a single device in the network, managed by a master thread.
    @details This class encapsulates the device's state, including its sensor data and assigned scripts.
             It relies on a root device (ID 0) to manage shared synchronization primitives like
             data locks and a global step barrier.
    """
    

    def __init__(self, device_id, sensor_data, supervisor, max_workers=8):
        """
        @brief Initializes a Device instance.
        @param device_id A unique identifier for the device.
        @param sensor_data A dictionary holding the local sensor data.
        @param supervisor An object for querying global state, like neighbors.
        @param max_workers The maximum number of concurrent worker threads this device can spawn.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts_received = Event()
        self.scripts = []
        self.master = Thread(target=self.master_func)


        self.master.start()
        # A semaphore to limit the number of concurrently active worker threads for this device.
        self.active_workers = Semaphore(max_workers)

        
        # A reference to the root device (ID 0), which holds global synchronization objects.
        self.root_device = None

        
        self.step_barrier = None
        self.data_locks = {}

    def __str__(self):
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Performs collective setup for all devices.
        @details The device with ID 0 is designated as the root and is responsible for creating a
                 global step barrier and a lock for each unique data location across all devices.
        @param devices A list of all Device objects in the simulation.
        """
        for dev in devices:
            if dev.device_id == 0:
                self.root_device = dev

        # Pre-condition: The root device (ID 0) initializes shared resources.
        if self.device_id == 0:
            
            
            
            # A global barrier to synchronize all devices at the end of a time step.
            self.step_barrier = ReusableBarrierSem(len(devices))

            
            # Block Logic: Create a reentrant lock for each unique data location found across all devices.
            # This ensures that operations on the same location are serialized, preventing race conditions.
            for device in devices:
                for (location, _) in device.sensor_data.iteritems():


                    if location not in self.data_locks:
                        self.data_locks[location] = RLock()

    def assign_script(self, script, location):
        """
        @brief Assigns a script to the device.
        @param script The script to be executed.
        @param location The location context for the script.
        """
        if script is not None:
            self.scripts.append((script, location))
        else:
            # A `None` script signals that all scripts for the time step have been assigned.
            self.scripts_received.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location in a thread-safe manner.
        @details It acquires a location-specific lock from the root device before reading.
        @param location The location for which to retrieve data.
        @return The sensor data or None if not found.
        """
        with self.root_device.data_locks[location]:
            if location in self.sensor_data:
                return self.sensor_data[location]
            else:
                return None

    def set_data(self, location, data):
        """
        @brief Updates sensor data for a given location in a thread-safe manner.
        @details It acquires a location-specific lock from the root device before writing.
        @param location The location to update.
        @param data The new data value.
        """
        with self.root_device.data_locks[location]:
            if location in self.sensor_data:
                self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device by waiting for its master thread to complete.
        """
        self.master.join()

    def master_func(self):
        """
        @brief The main control loop for the device, running in the master thread.
        @details This function waits for scripts, spawns worker threads to execute them (respecting
                 the concurrency limit), and synchronizes at a global barrier after each time step.
        """
        while True:
            
            neighbours = self.supervisor.get_neighbours()
            if neighbours is None:
                # A `None` value for neighbors is the signal to terminate.
                break



            # Invariant: Wait until the main process signals that all scripts are assigned.
            self.scripts_received.wait()

            workers = []
            
            # Block Logic: For each script, spawn a new worker thread to execute it.
            for (script, location) in self.scripts:
                
                # Acquire the semaphore, blocking if the max number of workers are already active.
                self.active_workers.acquire()

                
                # This design creates a new thread for each task, which can be inefficient.
                worker = Thread(target=self.worker_func, \
                    args=(script, location, neighbours))
                workers.append(worker)
                worker.start()

            
            # Wait for all spawned worker threads for this time step to finish.
            for worker in workers:
                worker.join()

            
            self.scripts_received.clear()
            
            # Invariant: After all work is done, wait at the global barrier to synchronize with other devices.
            self.root_device.step_barrier.wait()


    def worker_func(self, script, location, neighbours):
        """
        @brief The function executed by each worker thread.
        @details This function performs the core logic: it acquires a location-specific lock, 
                 gathers data, runs the script, distributes the result, and finally releases the lock
                 and the semaphore.
        @param script The script to execute.
        @param location The data location context.
        @param neighbours A list of neighboring devices.
        """
        # Acquire the global, location-specific lock to ensure data integrity.
        with self.root_device.data_locks[location]:
            
            script_data = []
            
            # Block Logic: Aggregate data from neighbors and the local device.
            for dev in neighbours:
                data = dev.get_data(location)
                if data is not None:
                    script_data.append(data)

            
            data = self.get_data(location)
            if data is not None:
                script_data.append(data)

            if script_data != []:
                
                result = script.run(script_data)

                
                # Disseminate the result to all involved devices.
                for dev in neighbours:
                    dev.set_data(location, result)

                
                self.set_data(location, result)

        
        # Release the semaphore to allow another worker thread to become active.
        self.active_workers.release()