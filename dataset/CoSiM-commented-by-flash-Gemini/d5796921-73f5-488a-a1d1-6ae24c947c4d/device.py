

"""
This module implements a device simulation framework that utilizes a worker pool
and a reusable barrier for synchronization. It defines:
- ReusableBarrier: A re-usable barrier mechanism for threads.
- Worker: A thread that processes assigned jobs (scripts).
- WorkerPool: Manages a collection of worker threads.
- Job: A data class representing a script execution task.
- DeviceSync: Manages synchronization primitives and shared states for a Device.
- Device: Represents a simulated device with sensor data, scripts, and multi-threaded processing.
- DeviceThread: The main thread for a Device, orchestrating job assignment and synchronization.
"""


from threading import Event, Thread, Lock, Condition
import Queue

class ReusableBarrier():
    """
    Implements a reusable barrier synchronization mechanism using a condition variable.
    This barrier allows a fixed number of threads to wait until all have reached a certain point
    before any are allowed to proceed, and can then be reset for subsequent synchronizations.
    """
    
    def __init__(self, num_threads):
        """
        Initializes the reusable barrier.

        Args:
            num_threads (int): The number of threads that must reach the barrier
                                before any can proceed.
        """
        
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        self.cond = Condition()

    def wait(self):
        """
        Blocks the calling thread until all `num_threads` have reached this barrier.
        When the last thread arrives, all waiting threads are notified.
        """
        
        self.acquire() # Acquire the condition variable's intrinsic lock
        self.count_threads -= 1
        # Block Logic: If this is the last thread to reach the barrier,
        # reset the counter and notify all waiting threads.
        if self.count_threads == 0:
            self.cond.notify_all()
            self.count_threads = self.num_threads # Reset counter for next use
        else:
            self.cond.wait() # Wait until all threads have arrived
        self.cond.release() # Release the condition variable's intrinsic lock
        
    def acquire(self):
        """
        Acquires the intrinsic lock associated with the condition variable.
        This is a necessary precursor to calling `wait()` or `notify()`.
        """
        
        self.cond.acquire()
              
class Worker(Thread):
    """
    A worker thread responsible for processing jobs (scripts) assigned to a device.
    It fetches jobs from a buffer, executes scripts, and updates data on neighboring devices.
    """
    
    def __init__(self, scripts_buffer, device):
        """
        Initializes a Worker thread.

        Args:
            scripts_buffer (Queue.Queue): The queue from which the worker fetches jobs.
            device (Device): The parent Device instance this worker belongs to.
        """
        
        Thread.__init__(self)
        self.device = device
        self.script_buffer = scripts_buffer
        
    def get_script_data(self, job):
        """
        Retrieves relevant sensor data for a script from neighboring devices and the current device.

        Args:
            job (Job): The job object containing information about the script and its location.

        Returns:
            list: A list of sensor data points collected for the script.
        """
        
        script_data = []
        # Block Logic: Collect data from neighboring devices specified in the job.
        for device in job.neighbours:
            data = device.get_data(job.location)
            if data is not None:
                script_data.append(data)
        
        # Block Logic: Collect data from the current device for the job's location.
        data = self.device.get_data(job.location)
    
        if data is not None:
            script_data.append(data)
        return script_data
        
    def update_data_on_neighbours(self, job, result):
        """
        Updates the sensor data on neighboring devices and the current device
        with the result of a script execution.

        Args:
            job (Job): The job object containing information about the script and its location.
            result (Any): The result returned by the executed script.
        """
        
        for device in job.neighbours:
            device.set_data(job.location, result)
        self.device.set_data(job.location, result)
    
    def run(self):
        """
        The main execution loop for the worker thread.
        It continuously fetches jobs from the `scripts_buffer`, executes the script
        if one is provided, and updates data on relevant devices.
        The worker terminates if it receives a job with a None script.
        """

        while True:
        
            job = self.script_buffer.get() # Block Logic: Fetch a job from the queue.

            if job.script is None: # Block Logic: Check for a shutdown signal (None script).
                self.script_buffer.task_done()
                break
            
            # Block Logic: Acquire a lock for the job's location to ensure exclusive access
            # during script execution and data updates.
            with self.device.sync.get_location_lock(job.location):
                script_data = self.get_script_data(job)
                
                if script_data != []: # Block Logic: Execute script only if relevant data is found.
                    # Inline: Execute the assigned script with collected data.
                    result = job.script.run(script_data)    
                    self.update_data_on_neighbours(job, result) # Update data on neighbours and self.
            
            self.script_buffer.task_done() # Mark the current job as done in the queue.      

class WorkerPool(object):
    """
    Manages a pool of worker threads. It initializes a specified number of Worker
    threads, provides a method to add jobs to their shared queue, and handles
    starting and stopping these workers.
    """
    

    def __init__(self, workers, device):
        """
        Initializes the WorkerPool.

        Args:
            workers (int): The number of worker threads to create in the pool.
            device (Device): The Device instance that this worker pool serves.
        """
        
        self.workers = workers
        self.workers_scripts = [] # List to hold Worker thread instances
        self.scripts_buffer = Queue.Queue() # Queue for jobs to be processed by workers
        self.device = device
        self.start_workers()
        
    def start_workers(self):
        """
        Starts the specified number of worker threads and adds them to the pool.
        """
        
        for i in range(0, self.workers):
            self.workers_scripts.append(Worker(self.scripts_buffer, 
                                               self.device))
            self.workers_scripts[i].start()
            
    def add_job(self, job):
        """
        Adds a job to the scripts buffer for workers to process.

        Args:
            job (Job): The job to be added to the queue.
        """
        
        self.scripts_buffer.put(job)

    def delete_workers(self):
        """
        Deletes worker threads from the pool.
        """
        
        for _ in (0, self.workers-1):
            del self.workers_scripts[-1]
            
    def join_workers(self):
        """
        Waits for all jobs in the queue to be processed and for worker threads to terminate.
        """
        # Block Logic: Ensure all tasks put into the queue have been processed.
        # Inline: Iterate through worker indices.
        for i in (0, self.workers-1):
            self.scripts_buffer.join() # Block until all items in the queue have been gotten and processed.
            self.workers_scripts[i].join() # Wait for the worker thread to complete its execution.
            
    def make_workers_stop(self):
        """
        Sends stop signals to all worker threads and waits for them to terminate gracefully.
        """
        # Block Logic: Add 'None' jobs to the queue as stop signals for each worker.
        for _ in range(0, 8): # Assuming 8 workers as per the Device class's nr_t
            self.add_job(Job(None, None, None)) # Job with None script acts as a shutdown signal
        self.join_workers()
        
class Job():
    """
    Represents a single job or task for a worker thread, encapsulating
    the script to be executed, its location, and relevant neighbor information.
    """
    
    def __init__(self, neighbours, script, location):
        """
        Initializes a Job instance.

        Args:
            neighbours (list): A list of neighboring Device instances relevant to this job.
            script (Script): The script object to be executed.
            location (str): The location associated with this script execution.
        """
        
        self.neighbours = neighbours
        self.script = script
        self.location = location

    def get_neighbours(self):
        """
        Retrieves the list of neighboring devices associated with this job.

        Returns:
            list: A list of neighboring Device instances.
        """
        
        return self.neighbours
    
    def get_script(self):
        """
        Retrieves the script associated with this job.

        Returns:
            Script: The script object to be executed.
        """
        
        return self.script
     
class DeviceSync(object):
    """
    Manages synchronization primitives and shared state relevant to a Device's
    multi-threaded operations. This includes events for setup and script reception,
    locks for locations, and a reusable barrier.
    """
    
    def __init__(self):
        """
        Initializes DeviceSync with events, an empty list for location-specific locks,
        and a placeholder for the barrier.
        """
        
        self.setup = Event()            # Event to signal when initial device setup is complete
        self.scripts_received = Event() # Event to signal when scripts have been assigned for a timepoint
        self.location_locks = []        # List to hold locks for different locations
        self.barrier = None             # Reusable barrier for thread synchronization
        
    def init_location_locks(self, locations):
        """
        Initializes a specified number of location-specific locks.

        Args:
            locations (int): The number of location locks to initialize.
        """
        
        for _ in range(0, locations):
            self.location_locks.append(Lock())
            
    def init_barrier(self, threads):
        """
        Initializes the reusable barrier with a given number of threads.

        Args:
            threads (int): The total number of threads that will participate in the barrier.
        """
        
        self.barrier = ReusableBarrier(threads)
        
    def set_setup_event(self):
        """
        Sets the setup event, signaling that initial device setup is complete.
        """
        
        self.setup.set()
        
    def wait_setup_event(self):
        """
        Blocks until the setup event is set.
        """
        
        self.setup.wait()
        
    def set_scripts_received(self):
        """
        Sets the scripts_received event, signaling that scripts have been assigned for the current timepoint.
        """
        
        self.scripts_received.set()
        
    def wait_scripts_received(self):
        """
        Blocks until the scripts_received event is set.
        """
        
        self.scripts_received.wait()
        


    def clear_scripts_received(self):
        """
        Clears the scripts_received event, resetting it for the next timepoint.
        """
        
        self.scripts_received.clear()
        
    def wait_threads(self):
        """
        Causes the calling thread to wait at the barrier until all participating threads arrive.
        """
        
        self.barrier.wait()
        
    def get_location_lock(self, location):
        """
        Retrieves the lock associated with a specific location.

        Args:
            location (int): The index of the location for which to retrieve the lock.

        Returns:
            threading.Lock: The lock object for the specified location.
        """
        
        return self.location_locks[location]
        
class Device(object):
    """
    Represents a simulated device in a distributed system. Each device
    manages its sensor data, executes scripts using a worker pool,
    and interacts with a supervisor. Synchronization is handled by DeviceSync.
    """
    
    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary containing initial sensor data for various locations.
            supervisor (Supervisor): The supervisor object responsible for managing devices
                                     and providing global information (e.g., neighbours).
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.scripts = [] # List to store assigned scripts
        
        self.sync = DeviceSync() # Synchronization object for this device
        self.worker_pool = WorkerPool(8, self) # Initialize a worker pool with 8 threads
        
        self.thread = DeviceThread(self) # Create the main device thread
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device <device_id>".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Performs initial setup for all devices in the simulation.
        This includes initializing location-specific locks and the global barrier,
        and propagating synchronization objects to all other devices.
        This method is typically called by a supervisor or a designated master device.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        # Block Logic: The last device in the list initializes shared synchronization primitives.
        if self.device_id == len(devices)-1:
            self.sync.init_location_locks(25) # Initialize 25 location locks
            self.sync.init_barrier(len(devices)) # Initialize barrier for all devices

            # Block Logic: Propagate the initialized synchronization objects to all devices.
            for device in devices:
                device.sync.barrier = self.sync.barrier
                device.sync.location_locks = self.sync.location_locks
                device.sync.set_setup_event() # Signal that setup is complete
            
    def add_job(self, job):
        """
        Adds a job to the device's worker pool for processing.

        Args:
            job (Job): The job to be added.
        """
        
        self.worker_pool.add_job(job)
        
    def assign_script(self, script, location):
        """
        Assigns a script to the device to be executed at a specific location.
        If a script is provided, it's added to the device's script list.
        If no script is provided (None), it signals that scripts have been received for the timepoint.

        Args:
            script (Script or None): The script object to assign, or None to signal script reception.
            location (str): The location associated with the script.
        """
        
        if script is not None:
            self.scripts.append((script, location)) # Add the script and its location to the list
        else:
            self.sync.set_scripts_received() # Signal that all scripts for this timepoint have been assigned

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location.

        Args:
            location (str): The location for which to retrieve data.

        Returns:
            Any or None: The sensor data if available for the location, otherwise None.
        """
        
        if location in self.sensor_data:
            return self.sensor_data[location]
        else:
            return None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location.

        Args:
            location (str): The location for which to set data.
            data (Any): The new sensor data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

        
    def shutdown(self):
        """
        Shuts down the device's main thread and worker pool.
        """
        
        self.thread.join()

class DeviceThread(Thread):
    """
    The main thread for a Device, responsible for orchestrating the overall
    simulation workflow, including synchronization with other devices and
    assigning jobs to its worker pool.
    """
    
    def __init__(self, device):
        """
        Initializes a DeviceThread.

        Args:
            device (Device): The parent Device instance this thread belongs to.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device  

    def run(self):
        """
        Executes the main logic of the DeviceThread.
        - Waits for initial device setup to complete.
        - Continuously fetches neighbor information from the supervisor.
        - If no neighbors are returned (signal for shutdown), it stops the worker pool and exits.
        - Waits for all threads to reach a barrier.
        - Assigns scripts to the worker pool.
        - Waits for all worker jobs to complete and then clears the scripts_received event.
        """
        # Block Logic: Wait for the global setup to be completed by a master device.
        self.device.sync.wait_setup_event()
        # Block Logic: Main loop for processing timepoints.
        while True: 
            # Block Logic: Get updated neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            
            # Block Logic: If no neighbors are returned, it's a shutdown signal.
            if neighbours is None:
                self.device.worker_pool.make_workers_stop() # Stop all worker threads
                break # Exit the main loop
            
            # Block Logic: Synchronize all DeviceThreads at the global barrier before processing scripts.
            self.device.sync.wait_threads()
            # Block Logic: Wait for scripts to be assigned by the supervisor for the current timepoint.
            self.device.sync.wait_scripts_received()
            
            # Block Logic: Assign each script to the worker pool as a new job.
            for (script, location) in self.device.scripts:
                self.device.add_job(Job(neighbours, script, location))
            
            # Block Logic: Synchronize all DeviceThreads after scripts have been added to the worker pool.
            self.device.sync.wait_threads()
            # Block Logic: Clear the scripts_received event for the next timepoint.
            self.device.sync.clear_scripts_received()