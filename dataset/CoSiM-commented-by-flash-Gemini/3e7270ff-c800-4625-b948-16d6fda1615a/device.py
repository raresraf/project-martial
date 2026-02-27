


"""
@file device.py
@brief This module defines a simulated device environment utilizing a thread pool for concurrent script execution and a reusable barrier for synchronization.

@details It implements the `Device` class to represent individual simulation entities.
         The `DeviceThread` manages the device's operational loop, dynamically adding
         tasks to a `ThreadPool` for parallel processing of scripts. Shared `Lock`
         objects ensure data consistency across different data locations. A custom
         `ReusableBarrier` (likely from an external module) is used for global synchronization.
         This file also contains a standalone `ReusableBarrier` implementation and a `MyThread`
         example, possibly for testing or demonstration, though not directly integrated
         into the primary `Device` logic.
"""

from threading import Event, Thread, Lock , Condition
from queue import Worker, ThreadPool
from reusable_barrier_semaphore import ReusableBarrier

class Device(object):
    """
    @brief Represents a single simulated device in the environment.

    @details Each device manages its own sensor data, interacts with a central supervisor,
             and executes scripts. It coordinates its activities through a dedicated
             `DeviceThread`, which in turn dispatches tasks to an internal `ThreadPool`
             for concurrent script execution. Shared `Lock` objects ensure data
             consistency across different data locations.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        @brief Initializes a new Device instance.

        @param device_id An integer representing the unique identifier for the device.
        @param sensor_data A dictionary containing initial sensor data for the device.
        @param supervisor An object responsible for overseeing and coordinating devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        self.script_received = Event()  # Event to signal that all scripts for a timepoint are assigned.
        self.wait_neighbours = Event()  # Event to signal when neighbor information is ready.
        self.scripts = []               # List to store (script, location) tuples assigned to this device.
        self.neighbours = []            # List to store neighboring devices for data interaction.
        self.allDevices = []            # List of all devices in the simulation (set during setup).
        self.locks = []                 # List of locks for data locations.
        self.pool = ThreadPool(8)       # Internal ThreadPool for executing scripts concurrently.
        self.lock = Lock()              # General-purpose lock for internal use (e.g., protecting shared data).
        self.thread = DeviceThread(self) # The dedicated thread for this device's main loop.
        self.thread.start()             # Start the main device thread.

    def __str__(self):
        """
        @brief Returns a string representation of the Device.

        @return A string in the format "Device %d" % self.device_id.
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        @brief Sets up shared synchronization resources.

        @details This method is called once at the beginning of the simulation.
                 It initializes a global `ReusableBarrier` and a list of `Lock`
                 objects for a fixed number of data locations, making them
                 accessible to all devices.
        @param devices A list of all Device objects in the simulation.
        """
        
        self.allDevices = devices # Store references to all devices.
        self.barrier = ReusableBarrier(len(devices)) # Initialize the global reusable barrier.

        # Block Logic: Initialize a fixed number of locks for potential data locations.
        for i in range(0, 50): # Assuming 50 distinct data locations need locking.
            self.locks.append(Lock()) # Create a new lock for each location.

        pass # No further action is explicitly defined in this section.

    def assign_script(self, script, location):
        """
        @brief Assigns a script to be executed by the device at a specific data location.

        @details If a script is provided, it's added to the device's script queue
                 and also added as a task to the internal thread pool for execution.
                 If `script` is None, it signals the `script_received` event,
                 indicating that all scripts for the current timepoint have been
                 assigned to this device.
        @param script The script object to assign, or None.
        @param location The data location (e.g., sensor ID) where the script will operate.
        """
        
        if script is not None:
            self.scripts.append((script, location)) # Store the script and its target location.
            self.pool.add_task(self.executeScript,script,location) # Add the script execution as a task to the thread pool.
        else:
            self.script_received.set() # Signal that all script assignments for this timepoint are done.

    def get_data(self, location):
        """
        @brief Retrieves sensor data for a specified location from the device's internal state.

        @param location The data location (e.g., sensor ID).
        @return The sensor data for the location, or None if the location is not found.
        """
        
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        @brief Sets or updates sensor data for a specified location in the device's internal state.

        @param location The data location (e.g., sensor ID).
        @param data The new sensor data to set.
        """
        
        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        @brief Initiates the shutdown sequence for the device's dedicated thread.

        @details Joins the `DeviceThread`, ensuring all ongoing operations are completed
                 before the device fully shuts down.
        """
        
        self.thread.join() # Wait for the main device thread to finish its execution.

    def executeScript(self,script,location):
        """
        @brief Executes a single script for a specific data location.

        @details This method is executed by a worker thread from the `ThreadPool`.
                 It waits for neighbor information to be ready, gathers data from
                 neighboring devices (acquiring locks), gathers data from the current
                 device (acquiring locks), executes the script, and then updates
                 the sensor data on both the neighbors and the current device (acquiring locks).
        @param script The script object to execute.
        @param location The data location (e.g., sensor ID) where the script operates.
        """

        self.wait_neighbours.wait() # Wait until neighbor information is available.
        script_data = [] # List to accumulate data for the script.

        # Block Logic: Gather data from neighboring devices, acquiring locks for each location.
        if not self.neighbours is None:
            for device in self.neighbours:
                device.locks[location].acquire() # Acquire lock for neighbor's data location.
                data = device.get_data(location)
                device.locks[location].release() # Release lock.

                if data is not None:
                    script_data.append(data)

        # Block Logic: Gather data from the current device, acquiring a lock for its data location.
        self.locks[location].acquire() # Acquire lock for current device's data location.
        data = self.get_data(location)
        self.locks[location].release() # Release lock.

        if data is not None:
            script_data.append(data)

        # Precondition: Execute the script only if there is data available.
        if script_data != []:
            result = script.run(script_data) # Execute the assigned script with the collected data.

            # Block Logic: Update sensor data on neighboring devices, acquiring locks.
            if not self.neighbours is None:
                for device in self.neighbours:

                    device.locks[location].acquire() # Acquire lock for neighbor's data location.
                    device.set_data(location, result)
                    device.locks[location].release() # Release lock.

            # Block Logic: Update sensor data on the current device, acquiring a lock.
            self.locks[location].acquire() # Acquire lock for current device's data location.
            self.set_data(location, result)
            self.locks[location].release() # Release lock.



class DeviceThread(Thread):
    """
    @brief Manages the main operational loop for a Device object, coordinating script execution.

    @details This thread is responsible for the overall lifecycle of a device's operations
             within a timepoint. It fetches neighbor information from the supervisor,
             dispatches script execution tasks to a thread pool, and handles global
             synchronization using the shared `ReusableBarrier`.
    """

    def __init__(self, device):
        """
        @brief Initializes a new DeviceThread instance.

        @param device The `Device` object that this thread is responsible for.
        """
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device # Reference to the parent Device object.

    def run(self):
        """
        @brief The main execution loop for the DeviceThread.

        @details This loop continuously processes timepoints in the simulation.
                 It first clears synchronization events, fetches neighbor information,
                 signals its worker threads that neighbor data is ready, and if the
                 simulation is ongoing, adds assigned scripts to the thread pool.
                 It then waits for all scripts to be received and for the thread pool
                 to complete its tasks. Finally, it synchronizes globally with other devices.
                 The loop terminates when the supervisor signals the end of the simulation.
        """

        while True:
            self.device.script_received.clear()  # Reset event for next timepoint.
            self.device.wait_neighbours.clear()  # Reset event for next timepoint.

            self.device.neighbours = [] # Clear previous neighbor list.
            # Block Logic: Fetch updated neighbor information from the supervisor.
            self.device.neighbours = self.device.supervisor.get_neighbours()
            self.device.wait_neighbours.set() # Signal that neighbor information is available.

            # Precondition: If supervisor returns None, it signals the end of the simulation.
            if self.device.neighbours is None:
                self.device.pool.wait_completion() # Wait for any remaining tasks in the pool.
                self.device.pool.terminateWorkers() # Signal workers to terminate.
                self.device.pool.threadJoin()     # Wait for worker threads to join.
                return # Exit the main loop.

            # Block Logic: Add scripts to the thread pool for parallel execution.
            for (script, location) in self.device.scripts:
                self.device.pool.add_task(self.device.executeScript,script,location)

            self.device.script_received.wait() # Wait until all scripts for the timepoint have been assigned.
            self.device.pool.wait_completion() # Wait for all tasks in the thread pool to complete.

            # Block Logic: Synchronize globally with other devices after processing the timepoint.
            for dev in self.device.allDevices: # Assuming 'dev.barrier.wait()' means the barrier is shared across all devices.
                dev.barrier.wait() # Global synchronization barrier.


from Queue import Queue
from threading import Thread

class Worker(Thread):
    """
    @brief A worker thread that processes tasks from a shared queue within a ThreadPool.

    @details This class is part of the `ThreadPool` implementation. Each worker
             continuously retrieves functions and their arguments from a `Queue`,
             executes them, and signals completion. It can be gracefully terminated.
    """
    def __init__(self, tasks):
        """
        @brief Initializes a new Worker thread.

        @param tasks The shared `Queue` from which tasks (functions and their arguments) are pulled.
        """
        Thread.__init__(self)
        self.tasks = tasks # The queue of tasks.
        self.daemon = True # Set as daemon to allow the main program to exit without waiting for it.
        self.terminate_worker = False # Flag to signal the worker to terminate.
        self.start() # Start the worker thread automatically.

    def run(self):
        """
        @brief The main execution loop for the Worker.

        @details This loop continuously gets tasks from the `tasks` queue. If a special
                 `None` task is received, it signals termination. Otherwise, it executes
                 the received function with its arguments and then signals task completion.
        """
        while True:
            func, args, kargs = self.tasks.get() # Retrieve a task (function, args, kwargs) from the queue.
            if func == None: # Check for a special termination signal (None function).
                self.tasks.task_done() # Mark the termination task as done.
                break # Exit the worker thread's loop.
            try: func(*args, **kargs) # Execute the task function.
            except Exception, e: print e # Catch and print any exceptions during task execution.
            self.tasks.task_done() # Signal that the task has been processed.


class ThreadPool:
    """
    @brief Manages a pool of worker threads for executing tasks concurrently.

    @details This class provides an abstraction for distributing tasks to a fixed
             number of worker threads. It uses a shared `Queue` for tasks and
             allows for adding tasks, waiting for their completion, and gracefully
             terminating the worker threads.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes a new ThreadPool.

        @param num_threads The number of worker threads to create in the pool.
        """
        self.tasks = Queue(99999) # A bounded queue to hold tasks.
        self.workers = []        # List to hold references to worker threads.
        # Block Logic: Create and add worker threads to the pool.
        for _ in range(num_threads):
            self.workers.append(Worker(self.tasks)) # Each worker is initialized with the shared task queue.

    def add_task(self, func, *args, **kargs):
        """
        @brief Adds a new task to the thread pool.

        @details A task consists of a function and its arguments. The task is
                 placed in the shared queue for a worker thread to pick up.
        @param func The function to be executed.
        @param args Positional arguments for the function.
        @param kargs Keyword arguments for the function.
        """
        self.tasks.put((func, args, kargs)) # Add the task to the queue.

    def wait_completion(self):
        """
        @brief Waits for all tasks in the queue to be completed.

        @details This method blocks until all tasks previously put into the queue
                 have been processed by the worker threads.
        """
        self.tasks.join() # Block until all tasks in the queue are done.

    def terminateWorkers(self):
        """
        @brief Signals all worker threads to terminate gracefully.

        @details This is achieved by adding a special `None` task to the queue
                 for each worker, which they interpret as a termination signal.
        """
        for worker in self.workers:
            worker.tasks.put([None,None,None]) # Add a termination signal for each worker.
            worker.terminate_worker = True # Set termination flag (though not explicitly used in this worker's run).

    def threadJoin(self):
        """
        @brief Waits for all worker threads in the pool to complete their execution and join.
        """

        for worker in self.workers:
            worker.join() # Join each worker thread.
from threading import *

class ReusableBarrier():
    """
    @brief Implements a reusable barrier for synchronizing multiple threads.

    @details This is an internal implementation of a reusable barrier, distinct from
             the one imported from `reusable_barrier_semaphore.py`. It uses a
             two-phase mechanism to ensure all participating threads wait until
             every thread has reached a common synchronization point, and then
             releases them simultaneously. It is designed to be reusable
             for subsequent synchronization points without needing re-initialization.
    """
    def __init__(self, num_threads):
        """
        @brief Initializes a new ReusableBarrier instance.

        @param num_threads The total number of threads that will participate in the barrier.
        """
        self.num_threads = num_threads
        # Two counters are used to manage the two phases of the barrier, allowing reusability.
        # Stored in a list to allow modification within nested scopes.
        self.count_threads1 = [self.num_threads]
        self.count_threads2 = [self.num_threads]
        self.count_lock = Lock()                 # A lock to protect access to the thread counters.
        self.threads_sem1 = Semaphore(0)         # Semaphore for the first phase.
        self.threads_sem2 = Semaphore(0)         # Semaphore for the second phase.
    
    def wait(self):
        """
        @brief Blocks the calling thread until all threads have reached this barrier.

        @details This method orchestrates the two-phase synchronization, ensuring
                 all `num_threads` complete `phase1` and `phase2` before any
                 thread proceeds past the `wait` call.
        """
        self.phase(self.count_threads1, self.threads_sem1)
        self.phase(self.count_threads2, self.threads_sem2)
    
    def phase(self, count_threads, threads_sem):
        """
        @brief Executes a single phase of the barrier synchronization.

        @details Decrements a shared counter. When the counter reaches zero, it
                 releases all waiting threads using a semaphore. It then resets
                 the counter for the next phase.
        @param count_threads A list containing the counter for the current phase (mutable).
        @param threads_sem The semaphore associated with the current phase.
        """
        with self.count_lock:
            count_threads[0] -= 1
            if count_threads[0] == 0:            
                for i in range(self.num_threads):
                    threads_sem.release()        # Release all waiting threads.
                count_threads[0] = self.num_threads  # Reset counter for reusability.
        threads_sem.acquire()                    # Block until released by the last thread.


class MyThread(Thread):
    """
    @brief Example thread demonstrating the usage of the ReusableBarrier.

    @details This thread repeatedly waits on a shared barrier and then prints
             its ID and the current step, illustrating how the barrier
             synchronizes multiple threads. This class is an example and not
             directly integrated into the main Device simulation logic. It is
             likely used for testing or demonstrating the ReusableBarrier's functionality.
    """
    def __init__(self, tid, barrier):
        """
        @brief Initializes a new MyThread instance.

        @param tid The unique identifier for this thread.
        @param barrier A shared `ReusableBarrier` instance.
        """
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier
    
    def run(self):
        """
        @brief The main execution loop for the example MyThread.

        @details This loop calls `barrier.wait()` repeatedly to synchronize
                 with other `MyThread` instances and then prints a message
                 after each synchronization step.
        """
        for i in xrange(10):
            self.barrier.wait() # Wait for all threads to reach this point.
            print "I'm Thread " + str(self.tid) + " after barrier, in step " + str(i) + "\n",