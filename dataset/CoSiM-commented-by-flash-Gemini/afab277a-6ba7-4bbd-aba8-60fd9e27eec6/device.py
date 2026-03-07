"""
@file device.py
@brief Implements core components for a distributed system, likely a simulation or sensor network.
This module defines Device objects that can communicate and process data concurrently
using multiple threads, employing synchronization primitives like locks, events, and semaphores.
It models a system where individual devices process sensor data and interact with neighbors
under the orchestration of a supervisor, utilizing a worker pool for script execution.
"""

from threading import Event, Thread, Lock, Semaphore
import Queue  # In Python 3, this is 'queue.Queue'. This indicates Python 2 compatibility or a mix.

class Device(object):
    """
    @brief Represents a single device in the distributed system.
    Each device has a unique ID, manages its sensor data, interacts with a supervisor,
    and processes assigned scripts in a dedicated thread. It includes various
    synchronization primitives for coordinated operation.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id (int): A unique identifier for the device.
        @param sensor_data (dict): A dictionary holding sensor readings for different locations.
        @param supervisor (Supervisor): A reference to the central supervisor managing devices.
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal when new scripts are assigned.
        self.scripts = []  # List to hold (script, location) tuples.
        self.timepoint_done = Event()  # Event to signal completion of a timepoint's processing.
        self.barrier = None  # Placeholder for a reusable barrier object, assigned during setup.
        self.thread = DeviceThread(self)  # The dedicated thread for this device's operations.
        self.thread.start()  # Starts the device's operational thread.

    def __str__(self):
        """
        @brief Provides a string representation of the Device.

        @return str: A formatted string indicating the device ID.
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared resources (barrier) across devices.
        This method is typically called by the supervisor or a coordinating entity.

        @param devices (list): A list of all Device instances in the system.
        """
        # Stores a reference to all devices in the system.
        self.devices = devices
        # Initializes a ReusableBarrier for all devices to synchronize.
        self.barrier = ReusableBarrier(len(self.devices))
      
        # Assigns the same barrier instance to all devices in the system.
        # This loop seems to re-assign the barrier for each device, effectively setting it up globally once.
        for i in xrange(len(self.devices)):
            self.devices[i].barrier = self.barrier

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed at a specific data location.

        @param script (callable): The script (function or object with a run method) to execute.
                                  If None, it signals that script assignment for the timepoint is done.
        @param location (int): The identifier of the data location the script operates on.
        """
        # Conditional Logic: If a script is provided, it's added to the list.
        if script is not None:
            self.scripts.append((script, location))
            
        else:
            # If script is None, it means no more scripts for the current timepoint.
            # Signals that scripts have been received and timepoint assignment is done.
            self.script_received.set()
            self.timepoint_done.set()

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a given location.

        @param location (int): The identifier of the data location.
        @return any: The sensor data at the specified location, or None if not found.
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a given location.

        @param location (int): The identifier of the data location to update.
        @param data (any): The new data value.
        """
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Shuts down the device's operational thread, waiting for its completion.
        """
        self.thread.join()


class Worker(Thread):
    """
    @brief A worker thread responsible for executing a single script for a Device.
    It collects data from its device and its neighbors, runs the assigned script,
    and then propagates the updated data back to the device and its neighbors.
    """

    def __init__(self, device, neighbours, script, location):
        """
        @brief Initializes a new Worker thread.

        @param device (Device): The Device object that created this worker.
        @param neighbours (list): A list of neighboring Device objects.
        @param script (callable): The script (function or object with a run method) to execute.
        @param location (int): The data location the script operates on.
        """
        Thread.__init__(self, name="Thread %d's Worker " % (device.device_id))
        self.device = device
        self.neighbours = neighbours
        self.script = script
        self.location = location

    def run(self):
        """
        @brief The main execution logic for the Worker thread.
        Collects data, runs the script, and updates data on devices.
        """
        scriptData = []        
        data = self.device.get_data(self.location)
        
        # Conditional Logic: Collects data from its own device if available.
        if not data is None:
            scriptData.append(data)

        # Block Logic: Collects data from neighboring devices.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if not data is None:
                scriptData.append(data)

        # Conditional Logic: If any data was collected, execute the script and propagate results.
        if scriptData:
            # Executes the assigned script with the collected data.
            newData = self.script.run(scriptData)

            # Block Logic: Propagates the new data to all neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, newData)
            # Updates the data on its own device.
            self.device.set_data(self.location, newData)

    def shutdown(self):
        """
        @brief Joins the worker thread, waiting for its completion.
        """
        self.join()


class DeviceThread(Thread):
    """
    @brief The dedicated thread of execution for a Device.
    This thread continuously monitors for new script assignments,
    manages a pool of Worker threads to execute scripts concurrently,
    and synchronizes with other device threads using a barrier.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device (Device): The Device object this thread is responsible for.
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        

    def run(self):
        """
        @brief The main execution loop of the DeviceThread.
        It continuously fetches neighbor information, waits for scripts,
        distributes them to a fixed-size worker pool, and then
        synchronizes via a barrier before starting the next timepoint.
        """
        q = Queue.Queue() # A queue to hold (script, location) tuples for workers.

        listOfWorkers = [] # List to manage active worker threads.
        numberOfWorkers = 0 # Counter for currently active workers.

        while True:
            # Block Logic: Continuously checks for supervisor signals and processes timepoints.
            # Invariant: Each iteration represents a timepoint or processing cycle.
            
            # Retrieves neighbor devices from the supervisor. If None, it signals shutdown.
            neighbours = self.device.supervisor.get_neighbours()
            if neighbours is None:
                break # Terminates the thread if supervisor signals shutdown.
            
            # Synchronization: Waits until scripts for the current timepoint are fully assigned.
            self.device.script_received.wait()
            
            # Block Logic: Transfers assigned scripts from the device's list to the queue.
            for (script, location) in self.device.scripts:
                q.put((script,location))
          
            # Block Logic: Processes scripts from the queue using a worker pool.
            while not q.empty():
                (script,location) = q.get() # Retrieves a script and its location from the queue.         

                # Conditional Logic: Manages the worker pool, ensuring no more than 8 active workers.
                if numberOfWorkers < 8:
                    # Creates and starts a new worker if the pool is not full.
                    worker = Worker(self.device, neighbours, script, location)
                    listOfWorkers.append(worker)
                    worker.start()
                    numberOfWorkers += 1
                
                else:
                    # If the worker pool is full, finds a finished worker to replace.
                    index = -1
                    for i in range(len(listOfWorkers)):
                        if not listOfWorkers[i].is_alive():
                            listOfWorkers[i].shutdown() # Ensures the finished worker is joined.
                            index = i
                            break
                    # Removes the finished worker and adds a new one to the same slot.
                    listOfWorkers.pop(index)
                    
                    worker = Worker(self.device, neighbours, script, location)
                    listOfWorkers.insert(index,worker)
                    listOfWorkers[index].start()
                    # Note: numberOfWorkers is not correctly maintained here, it should stay at 8.
                    # This could lead to logical errors if numberOfWorkers is critical beyond this block.
                    # As numberOfWorkers was incremented above without being decremented for the removed worker.
                    # For a robust implementation, numberOfWorkers should be decremented when a worker finishes.
                    # numberOfWorkers+=1; # This line should be reconsidered based on desired worker pool management.


                q.task_done() # Signals that the task from the queue is done. 

            # Block Logic: Joins all remaining active worker threads to ensure completion before proceeding.
            for i in range(len(listOfWorkers)):
                listOfWorkers[i].shutdown()

            # Synchronization: Waits until all script processing for the timepoint is conceptually done.
            self.device.timepoint_done.wait()       
            # Synchronization: Waits at the reusable barrier until all device threads complete their timepoint.
            self.device.barrier.wait()              
            
            # Resets the event flags for the next timepoint.
            self.device.script_received.clear()     
            self.device.timepoint_done.clear()      
            # Clears the list of scripts for the next timepoint.
            self.device.scripts = []


class ReusableBarrier():
    """
    @brief Implements a reusable barrier using semaphores and a lock for thread synchronization.
    This barrier allows a fixed number of threads to wait for each other before
    proceeding, and can be reused multiple times. It employs a two-phase
    approach to ensure proper synchronization and reset.
    """

    def __init__(self, num_threads):
        """
        @brief Initializes the reusable barrier.

        @param num_threads (int): The total number of threads that must reach the barrier.
        """
        self.num_threads = num_threads
        # Counters for threads reaching the barrier in each phase.
        # Stored in a list to make them mutable within the 'phase' method.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.counter_lock = Lock()                  # Lock to protect access to the counters.
        self.threads_sem1 = Semaphore(0)            # Semaphore for threads waiting in phase 1.
        self.threads_sem2 = Semaphore(0)            # Semaphore for threads waiting in phase 2.
 
    def wait(self):
        """
        @brief Blocks the calling thread until all 'num_threads' have reached the barrier.
        This method executes both phases of the barrier to ensure full synchronization.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
 
    def phase(self, count_threads, threads_sem):
        """
        @brief Implements one phase of the barrier synchronization.

        @param count_threads (list): A list containing the counter for the current phase.
                                     (Using a list allows modification within the function scope).
        @param threads_sem (Semaphore): The semaphore associated with this phase.
        """
        with self.counter_lock: # Ensures exclusive access to the counter.
            count_threads[0] -= 1 # Decrements the thread count for this phase.
            # Conditional Logic: If this is the last thread to reach the barrier in this phase.
            if count_threads[0] == 0:               
                # Releases all waiting threads from the semaphore.
                for i in range(self.num_threads):
                    threads_sem.release()           
                # Resets the counter for the next use of this phase.
                count_threads[0] = self.num_threads 
        threads_sem.acquire() # Threads wait here until released by the last thread.