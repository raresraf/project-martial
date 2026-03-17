"""
@6984f612-e3ae-43cf-8930-6c97fcee175d/device.py
@brief This module implements a distributed simulation or data processing system.

It defines three core classes:
- `ReusableBarrier`: A custom barrier synchronization mechanism for threads.
- `Device`: Represents a computational node that manages its sensor data,
  worker threads, and coordinates with a supervisor and other devices.
- `DeviceThread`: Worker threads spawned by a `Device` to execute assigned scripts
  and manage data access.

The system relies on `threading` primitives (Lock, Event, Thread, Condition)
for concurrency and synchronization, allowing parallel processing of scripts
across multiple devices and within a single device.

Algorithm:
- Decentralized processing: Each `Device` operates semi-autonomously.
- Timepoint synchronization: Devices (via their threads) synchronize at discrete timepoints using a custom barrier.
- Concurrent script execution: `DeviceThread`s execute scripts in parallel.
- Distributed locking: Location-specific locks ensure data consistency across devices.
- Load balancing: Scripts are spread among worker threads based on their `id_thread`.

Time Complexity:
- `Device.__init__`: O(N_threads) where N_threads is number of threads.
- `Device.setup_devices`: O(D * L + D * N_threads) where D is number of devices, L is number of locations, N_threads is number of threads per device.
- `DeviceThread.run`: O(T * S * N_neighbors * L_locations) where T is timepoints, S is scripts per thread, N_neighbors is number of neighbors, L_locations is number of locations.
Space Complexity:
- `Device`: O(L) for locks per location, O(N_threads) for threads.
- `ReusableBarrier`: O(1).
- `DeviceThread`: O(1) beyond script and data storage.
"""

from threading import Lock, Event, Thread, Condition

class ReusableBarrier():
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.
           Threads wait at the barrier until all participating threads have arrived,
           after which all are released simultaneously.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.
        @param num_threads: The total number of threads that must reach the barrier
                            before any can proceed.
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads    # Pre-condition: Initial count set to total threads.
        self.cond = Condition()                  # Functional Utility: Manages waiting and notification for threads.
                                                 

    def wait(self):
        """
        @brief Causes the calling thread to wait at the barrier until all other
               participating threads have also called `wait()`.
        Pre-condition: All threads have not yet reached the barrier.
        Invariant: `self.count_threads` accurately reflects the number of threads
                   yet to reach the barrier in the current cycle.
        Post-condition: All threads are released, and the barrier resets for reuse.
        """
        # Block Logic: Atomically decrements the count of threads and manages waiting.
        self.cond.acquire()                      
        self.count_threads -= 1
        # Block Logic: Checks if this thread is the last to arrive at the barrier.
        # Pre-condition: `self.count_threads` is 0.
        # Post-condition: All waiting threads are notified, and the barrier count is reset.
        if self.count_threads == 0:
            self.cond.notify_all()               # Functional Utility: Releases all threads waiting on this condition.
            self.count_threads = self.num_threads    # Functional Utility: Resets the barrier for subsequent reuse.
        else:
            self.cond.wait()                    # Functional Utility: Pauses the current thread until notified.
        self.cond.release()                     # Functional Utility: Releases the lock, allowing other threads to acquire it.


class Device(object):
    """
    @brief Represents a computational device within the distributed simulation system.
           Manages sensor data, worker threads (`DeviceThread`), and coordinates
           with a supervisor and other devices for distributed processing.
    """
    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a Device instance.
        @param device_id: A unique identifier for this device.
        @param sensor_data: A dictionary containing initial sensor data for various locations.
        @param supervisor: The central supervisor object responsible for orchestrating devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Functional Utility: Signals when new scripts are assigned.
        self.scripts = []               # Functional Utility: Stores scripts awaiting execution.
        self.timepoint_done = Event()   # Functional Utility: Signals when a timepoint's processing is complete.
        self.gotneighbours = Event()    # Functional Utility: Signals when neighbor information has been retrieved.
        self.zavor = Lock()             # Functional Utility: Protects shared resources during neighbor updates.
        self.threads = []               # Functional Utility: List of worker threads for this device.
        self.neighbours = []            # Functional Utility: Stores references to neighboring devices.
        self.nthreads = 8               # Configuration: Number of worker threads per device.
        self.barrier = ReusableBarrier(1) # Functional Utility: Placeholder barrier, reconfigured in `setup_devices`.
        self.lockforlocation = {}       # Functional Utility: Maps location IDs to locks for data consistency.
        self.num_locations = supervisor.supervisor.testcase.num_locations
        # Block Logic: Initializes and starts worker threads for this device.
        # Invariant: Each DeviceThread is associated with this Device instance.
        for i in xrange(self.nthreads):
            self.threads.append(DeviceThread(self, i))
        # Block Logic: Starts each worker thread.
        for i in xrange(self.nthreads):
            self.threads[i].start()


    def __str__(self):
        """
        @brief Returns a string representation of the Device.
        @return: A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Configures shared synchronization primitives across all devices.
               Initializes a global barrier and location-specific locks.
        @param devices: A list of all Device instances in the simulation.
        Pre-condition: All Device instances have been initialized.
        Post-condition: All devices share the same barrier and location locks.
        """
        # Functional Utility: Creates a global reusable barrier for all worker threads across all devices.
        barrier = ReusableBarrier(devices[0].nthreads*len(devices))
        lockforlocation = {}
        # Block Logic: Initializes a unique lock for each location to protect shared sensor data.
        for i in xrange(0, devices[0].num_locations):
            lock = Lock()
            lockforlocation[i] = lock
        # Block Logic: Assigns the global barrier and shared location locks to each device.
        for i in xrange(0, len(devices)):
            devices[i].barrier = barrier
            devices[i].lockforlocation = lockforlocation


    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by a worker thread at a specific location.
               If `script` is None, it signals the end of timepoint processing.
        @param script: The script object to execute.
        @param location: The location ID where the script should be executed.
        Pre-condition: `self.scripts` contains previously assigned scripts or is empty.
        Post-condition: `script` is added to `self.scripts`, or `timepoint_done` is set.
        """
        # Block Logic: Appends the script and its location to the list of pending scripts
        # and signals that a new script has been received.
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # Functional Utility: Signals to worker threads that the current timepoint
            # has no more scripts to be assigned, allowing them to proceed to processing.
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location.
        @param location: The location ID for which to retrieve data.
        @return: The sensor data for the given location, or None if not present.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Updates the sensor data for a specified location.
        @param location: The location ID for which to update data.
        @param data: The new data value to set for the location.
        Pre-condition: `location` exists in `self.sensor_data`.
        Post-condition: `self.sensor_data[location]` is updated with `data`.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down all worker threads associated with this device.
        Pre-condition: All worker threads are running.
        Post-condition: All worker threads have completed their execution and joined.
        """
        # Block Logic: Iterates through and joins each worker thread, ensuring they complete.
        for i in xrange(self.nthreads):
            self.threads[i].join()


class DeviceThread(Thread):
    """
    @brief Represents a worker thread for a Device, responsible for executing
           assigned scripts and managing data access.
    """
    def __init__(self, device, id_thread):
        """
        @brief Initializes a DeviceThread instance.
        @param device: The parent Device instance that spawned this thread.
        @param id_thread: A unique identifier for this thread within its parent device.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        self.id_thread = id_thread

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.
               Continuously processes scripts assigned by the parent Device,
               synchronizes with other threads, and handles data access.
        Pre-condition: The Device is operational and may assign scripts.
        Invariant: The thread continues to run until explicitly broken out of the loop
                   (e.g., when `self.device.neighbours` becomes None).
        Post-condition: The thread terminates after processing all assigned tasks and
                        receiving a shutdown signal (implicitly by `self.device.neighbours` being None).
        """
        # Block Logic: Main loop for continuous script processing and synchronization.
        while True:
            # Block Logic: Acquires a lock to safely check and update neighbor information.
            self.device.zavor.acquire()
            
            # Block Logic: Checks if neighbor information has already been retrieved for the current timepoint.
            # Pre-condition: `self.device.gotneighbours` is not set.
            # Post-condition: `self.device.neighbours` is populated, and `self.device.gotneighbours` is set.
            if self.device.gotneighbours.is_set() == False:
                # Functional Utility: Retrieves neighbor devices from the supervisor.
                self.device.neighbours = self.device.supervisor.get_neighbours()
                # Functional Utility: Sets the event to indicate neighbors have been retrieved.
                self.device.gotneighbours.set()
            self.device.zavor.release() # Functional Utility: Releases the lock.
            

            # Block Logic: Checks if the device has been signaled to shut down.
            # Pre-condition: `self.device.neighbours` is None, indicating shutdown.
            # Post-condition: The thread exits its main loop and terminates.
            if self.device.neighbours is None:
                break

            
            # Block Logic: Waits until all scripts for the current timepoint have been assigned.
            # Invariant: `self.device.timepoint_done` will be set by the Device when all scripts are ready.
            self.device.timepoint_done.wait()
            
            myscripts = []
            # Block Logic: Selects a subset of scripts to execute based on the thread's ID (load balancing).
            # Pre-condition: `self.device.scripts` contains all scripts for the current timepoint.
            # Post-condition: `myscripts` contains scripts assigned to this specific thread.
            for i in xrange(self.id_thread, len(self.device.scripts), self.device.nthreads + 1):
                myscripts.append(self.device.scripts[i])

            
            # Block Logic: Iterates through the assigned scripts and executes them.
            # Invariant: Each script is executed within the protection of a location-specific lock.
            for (script, location) in myscripts:
                # Functional Utility: Acquires a lock for the specific location to ensure data consistency.
                self.device.lockforlocation[location].acquire()
                script_data = []
                # Block Logic: Gathers sensor data from neighboring devices for the current location.
                for device in self.device.neighbours:
                    data = device.get_data(location)
                    if data is not None:
                        script_data.append(data)
                
                # Block Logic: Gathers own sensor data for the current location.
                data = self.device.get_data(location)
                if data is not None:
                    script_data.append(data)

                # Block Logic: Executes the script if relevant data is available and updates data.
                # Pre-condition: `script_data` is not empty.
                # Post-condition: Sensor data in `self.device` and `self.device.neighbours` is updated.
                if script_data != []:
                    
                    # Functional Utility: Executes the script with the gathered data.
                    result = script.run(script_data)
                    # Block Logic: Propagates the script's result to neighboring devices.
                    for device in self.device.neighbours:
                        device.set_data(location, result)
                    # Functional Utility: Updates the device's own sensor data with the script's result.
                    self.device.set_data(location, result)
                self.device.lockforlocation[location].release() # Functional Utility: Releases the lock for the specific location.

            
            # Functional Utility: Synchronizes all worker threads at the end of script execution.
            # Pre-condition: All scripts assigned to this thread have been processed.
            # Post-condition: All worker threads across all devices have completed script processing.
            self.device.barrier.wait()


            # Block Logic: Only the first thread (id_thread 0) is responsible for resetting timepoint-specific events.
            # Pre-condition: All threads have passed the first barrier.
            # Post-condition: `timepoint_done` and `gotneighbours` events are reset for the next timepoint.
            if self.id_thread == 0:
                self.device.timepoint_done.clear()  # Functional Utility: Resets the timepoint completion signal.
                self.device.gotneighbours.clear()   # Functional Utility: Resets the neighbor retrieval signal.
            
            # Functional Utility: Synchronizes all worker threads again after event resetting to ensure consistency
            # before starting the next timepoint's processing.
            self.device.barrier.wait()
