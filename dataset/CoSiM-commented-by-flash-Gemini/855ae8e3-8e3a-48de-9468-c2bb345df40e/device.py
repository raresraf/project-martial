

"""
This module implements a simulated device within a distributed sensor network,
featuring a specialized reusable barrier for synchronization and a multi-threaded
approach to script execution. It uses shared locks and barriers for coordination
among devices.

Domain: Distributed Systems, Concurrency, Sensor Networks.
"""

from threading import Event, Thread, Lock, Semaphore
from multiprocessing import cpu_count


class ReusableBarrier():
    """
    @brief Implements a reusable barrier for thread synchronization.

    This barrier allows a fixed number of threads to wait until all have reached a certain point
    before any are allowed to proceed. It uses a two-phase approach (implemented by the generic
    `phase` method) to ensure reusability without deadlocks. The counters are wrapped in lists
    to allow modification within nested scopes (e.g., `with` statements).
    """
    def __init__(self, num_threads):
        """
        @brief Initializes the ReusableBarrier with a specified number of threads.

        @param num_threads: The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # @brief Counter for the first phase of the barrier, wrapped in a list.
        self.count_threads1 = [self.num_threads]
        # @brief Counter for the second phase of the barrier, wrapped in a list.
        self.count_threads2 = [self.num_threads]
        # @brief Lock to protect access to the thread counters.
        self.count_lock = Lock()                 
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
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        @brief Implements a single phase of the barrier synchronization.

        Threads decrement a shared counter. The last thread to reach zero releases all
        waiting threads in this phase and resets the counter for the next cycle.
        
        @param count_threads: A list containing the counter for the current phase.
        @param threads_sem: The semaphore associated with the current phase.
        Invariant: All threads must pass through this phase before any can proceed
                   if it's the first phase, or before the barrier can be reused
                   if it's the second phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                # Block Logic: Release all threads waiting in this phase.
                for i in range(self.num_threads):
                    threads_sem.release()        
                # Inline: Reset the counter for the next cycle.
                count_threads[0] = self.num_threads  
        threads_sem.acquire()


class Device(object):
    """
    @brief Represents a simulated device in a distributed sensor network.

    This class manages sensor data, stores assigned scripts, and coordinates
    synchronization and data access using shared locks and barriers (received
    via `set_lock` from a coordinating device).
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
        # @brief List to store assigned scripts and their locations.
        self.scripts = []

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device <device_id>".
        """
        return "Device %d" % self.device_id

    def set_lock(self, lock1, lock2, barrier1, barrier2):
        """
        @brief Sets shared synchronization primitives for this device.

        These primitives are typically initialized by a coordinating device (e.g., Device 0).

        @param lock1: A shared Lock for coordinating access to supervisor interactions (e.g., getting neighbors).
        @param lock2: A shared Lock for coordinating data modifications (e.g., set_data by MyThread).
        @param barrier1: The first shared ReusableBarrier instance, used as `script_received`.
        @param barrier2: The second shared ReusableBarrier instance, used as `timepoint_done`.
        """
        self.lock1=lock1 # @brief Shared lock for supervisor interaction.
        self.lock2=lock2 # @brief Shared lock for data modification.
        self.script_received = barrier1 # @brief Barrier for signaling script reception.
        self.timepoint_done = barrier2 # @brief Barrier for signaling timepoint completion.


    def setup_devices(self, devices):
        """
        @brief Sets up and distributes shared synchronization primitives across devices.

        The first device in the `devices` list initializes two Locks and two ReusableBarriers,
        then distributes these instances to all other devices in the network.
        Each device then starts its own processing thread.

        @param devices: A list of all Device objects in the network.
        """
        # Block Logic: Only the first device (based on `device_id` comparison with `devices[0].device_id`)
        # initializes the shared locks and barriers.
        if self.device_id==devices[0].device_id:
            lock1=Lock() # @brief Initialize shared lock for supervisor access.
            lock2=Lock() # @brief Initialize shared lock for data modification.
            barrier1=ReusableBarrier(len(devices)) # @brief Initialize barrier for script reception.
            barrier2=ReusableBarrier(len(devices)) # @brief Initialize barrier for timepoint completion.
            
            # Block Logic: Distribute the initialized locks and barriers to all devices.
            for dev in devices:
                dev.set_lock(lock1, lock2, barrier1, barrier2)

        # Inline: Each device starts its own processing thread after setup.
        self.thread = DeviceThread(self)
        self.thread.start()        

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        @param script: The script object to be executed.
        @param location: The data location (index) the script operates on.
        """
        if script is not None:
            self.scripts.append((script, location))

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
    @brief A worker thread that executes a script and updates shared data.

    This thread is responsible for running a given script with its data,
    and then applying the result to the device's and its neighbors' sensor data,
    using shared locks for data consistency.
    """
    def __init__(self, script, script_date, device, neighbours, location):
        """
        @brief Initializes a new MyThread instance.

        @param script: The script object to be executed.
        @param script_date: The input data for the script.
        @param device: The Device object this thread is associated with.
        @param neighbours: A list of neighboring Device objects.
        @param location: The data location (index) this thread will operate on.
        """
        Thread.__init__(self)
        self.script=script
        self.script_data=script_date
        self.result=None # @brief Stores the result of the script execution.
        self.device=device
        self.neighbours=neighbours
        self.location=location

    def run(self):
        """
        @brief Executes the script, applies the result, and manages data access.

        This method acquires a lock for the specific data modification (`self.device.lock2`).
        It collects data from its associated device and neighbors, executes the script,
        and then updates their sensor data with the script's result. Finally, it releases
        the data modification lock. Concurrency control for `MyThread` instances is
        managed by the `DeviceThread`'s batch processing and semaphore usage at its level.
        """
        # Block Logic: Execute the assigned script with its data.
        result = self.script.run(self.script_data)
        
        # Block Logic: Acquire a global lock (lock2) to ensure exclusive write access to sensor data.
        self.device.lock2.acquire()
        
        # Block Logic: Disseminate the script's result to neighboring devices.
        for device in self.neighbours:
            device.set_data(self.location, result)
        
        # Block Logic: Update the current device's data with the script's result.
        self.device.set_data(self.location, result)
        
        # Block Logic: Release the global data modification lock (lock2).
        self.device.lock2.release()

class DeviceThread(Thread):
    

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        """
        @brief The main execution loop for the device thread.

        This loop continuously performs the following actions:
        1. Acquires `device.lock1` to safely interact with the supervisor for neighbor information.
        2. Releases `device.lock1`.
        3. Terminates if no neighbors are returned (simulation end).
        4. Waits on `device.script_received` barrier, indicating scripts are assigned for the timepoint.
        5. Collects data and prepares worker threads (`MyThread` instances) for each assigned script.
        6. Executes `MyThread` instances in batches, using `cpu_count()` to optimize parallelism.
        7. Waits for all `MyThread` instances in a batch to complete before processing the next batch.
        8. Waits on `device.timepoint_done` barrier, synchronizing with other devices after all scripts for the timepoint are processed.
        Invariant: The device processes data in discrete timepoints, synchronizing with the network
                   after each timepoint.
        """
        while True:
            
            # Block Logic: Acquire a lock to safely get neighbors from the supervisor.
            self.device.lock1.acquire()
            neighbours = self.device.supervisor.get_neighbours()
            self.device.lock1.release() # Inline: Release the lock after interacting with the supervisor.
            
            # Block Logic: Terminate the thread if no more neighbors are returned (simulation end).
            if neighbours is None:
                break
    
            # Block Logic: Wait on the barrier until scripts for the current timepoint have been assigned to all devices.
            self.device.script_received.wait()
            
            threads=[] # @brief List to hold MyThread worker instances for the current timepoint.
            
            # Block Logic: Prepare data for each assigned script and create MyThread instances.
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

                if script_data != []:
                        
                    threads.append(MyThread(script,script_data,self.device,neighbours,location))

            step=cpu_count()*2 # @brief Determine batch size for concurrent thread execution.
            # Block Logic: Execute MyThread instances in batches to manage system resources.
            for i in range(0,len(threads),step):
                for j in range(step):
                    if i+j<len(threads):
                        threads[i+j].start() # Inline: Start a MyThread instance.
                for j in range(step):
                    if i+j<len(threads):
                        threads[i+j].join() # Inline: Wait for a MyThread instance to complete.

            # Block Logic: Wait on the barrier until all devices have completed script processing for the timepoint.
            self.device.timepoint_done.wait()
