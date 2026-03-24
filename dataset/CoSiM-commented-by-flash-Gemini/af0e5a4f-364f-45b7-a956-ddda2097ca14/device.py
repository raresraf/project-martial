"""
This module defines a simulation framework for distributed sensor data processing
with an integrated thread pool for concurrent script execution.

It includes:
- Device: Represents a simulated device managing sensor data, scripts, and synchronization.
- DeviceThread: A dedicated thread for each Device to manage simulation logic and task dispatch.
- ThreadPool: A generic thread pool implementation to manage worker threads for executing tasks concurrently.
- Worker: A worker thread class used by the ThreadPool.
"""


from threading import Thread, Condition, Semaphore
from barrier import Barrier
from threadpool import ThreadPool

class Device(object):
    """
    Represents a simulated computational device in a distributed system.
    Each device has a unique ID, sensor data, interacts with a supervisor,
    and can have scripts assigned for execution. It manages its own dedicated thread
    and synchronization mechanisms for data access and timepoint progression.
    """
    

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Initializes a new Device instance.

        Args:
            device_id (int): A unique identifier for the device.
            sensor_data (dict): A dictionary mapping data locations to initial sensor readings.
            supervisor (Supervisor): The supervisor object responsible for coordinating devices.
        """
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.data_semaphores = {loc : Semaphore(1) for loc in sensor_data}
        self.scripts = []

        self.new_script = False
        self.timepoint_end = False
        self.cond = Condition()

        self.barrier = None
        self.supervisor = supervisor
        self.thread = DeviceThread(self)
        # Starts the dedicated thread for this device.
        self.thread.start()

    def __str__(self):
        """
        Returns a string representation of the Device.

        Returns:
            str: A string in the format "Device {device_id}".
        """
        
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Configures the shared barrier for all devices.
        This method is designed to be called by a coordinating entity (e.g., supervisor).
        Only the device with device_id 0 initializes the global barrier.

        Args:
            devices (list): A list of all Device instances in the simulation.
        """
        
        # Block Logic: Only device with ID 0 acts as the coordinator to set up the global barrier.
        if self.device_id == 0:
            # Initializes a Barrier for all participating devices.
            self.barrier = Barrier(len(devices))
            # Distributes the created barrier to all other devices in the simulation.
            for neigh in devices:
                if neigh.device_id != self.device_id:
                    neigh.set_barrier(self.barrier)

    def set_barrier(self, barrier):
        """
        Sets the shared barrier for this device. This is typically called by
        the coordinating device (device_id 0) during setup.

        Args:
            barrier (Barrier): The shared barrier instance.
        """

        
        self.barrier = barrier

    def assign_script(self, script, location):
        """
        Assigns a script to be executed at a specific data location on this device.
        Notifies the DeviceThread about new scripts or timepoint termination.

        Args:
            script (object): The script object to be executed.
            location (int): The identifier for the data location where the script should run.
        """

        
        # Block Logic: Acquires a condition lock to safely modify shared state.
        with self.cond:
            # If a script is provided, it's added to the list and a flag is set.
            if script is not None:
                self.scripts.append((script, location))
                self.new_script = True
            # If no script is provided (None), it signals the end of a timepoint.
            else:
                self.timepoint_end = True
            # Notifies all waiting threads that the state has changed.
            self.cond.notifyAll()

    def timepoint_ended(self):
        """
        Waits for a signal indicating either a new script has been assigned
        or the current timepoint has ended. This is a blocking call.

        Returns:
            bool: True if the timepoint has ended, False if a new script has been assigned.
        """

        
        # Block Logic: Acquires the condition lock to safely check and modify state variables.
        with self.cond:
            # Waits until either a new script is assigned or the timepoint termination is signaled.
            while not self.new_script and \
                  not self.timepoint_end:
                self.cond.wait()

            # If a new script was assigned, reset the flag and indicate that the timepoint has not ended.
            if self.new_script:
                self.new_script = False
                return False
            # If the timepoint ended, reset the flag and potentially prepare for new scripts.
            else:
                self.timepoint_end = False
                # If there are still scripts, set new_script to True to allow the loop to continue next time.
                self.new_script = len(self.scripts) > 0
                return True

    def get_data(self, location):
        """
        Retrieves sensor data for a specific location, acquiring a semaphore lock
        to ensure exclusive access to the data before returning it.

        Args:
            location (int): The identifier for the data location.

        Returns:
            Any: The sensor data at the specified location, or None if the location is not found.
        """

        


        # Block Logic: Checks if the requested location exists in the sensor data.
        if location in self.sensor_data:
            # Acquires a semaphore specific to this location to prevent concurrent access
            # while retrieving data, ensuring data consistency.
            self.data_semaphores[location].acquire()
            return self.sensor_data[location]
        else:
            # If the location is not found, returns None.
            return None

    def set_data(self, location, data):
        """
        Sets sensor data for a specific location and releases the associated semaphore.
        Data is updated only if the location exists.

        Args:
            location (int): The identifier for the data location.
            data (Any): The new sensor data to set.
        """

        
        # Block Logic: Updates sensor data for the given location if it exists.
        if location in self.sensor_data:
            self.sensor_data[location] = data
            # Releases the semaphore for this location, allowing other threads to access it.
            self.data_semaphores[location].release()

    def shutdown(self):
        """
        Shuts down the device by joining its associated DeviceThread,
        ensuring the thread completes its execution gracefully.
        """

        
        self.thread.join()

class DeviceThread(Thread):
    """
    Represents a dedicated worker thread for a Device.
    This thread manages the overall simulation logic for its associated device,
    including querying supervisor for neighbors, coordinating with other threads
    via barriers, and dispatching script execution to a thread pool.
    """
    

    def __init__(self, device):
        """
        Initializes a new DeviceThread instance.

        Args:
            device (Device): The Device instance this thread is associated with.
        """

        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    @staticmethod
    def run_script(own_device, neighbours, script, location):
        """
        Executes a given script at a specific location, collecting data from
        the own device and its neighbors, and then propagating the results.

        Args:
            own_device (Device): The Device instance where the script is being run.
            neighbours (list): A list of neighboring Device instances.
            script (object): The script object to execute.
            location (int): The data location relevant to the script.
        """

        
        script_data = []

        # Block Logic: Gathers sensor data from neighboring devices for the specified location.
        for device in neighbours:
            if device is own_device:
                continue # Skip the own device if it's somehow in the neighbors list.
            data = device.get_data(location)
            if data is not None:
                script_data.append(data)

        # Block Logic: Gathers sensor data from the own device for the specified location.
        data = own_device.get_data(location)

        if data is not None:
            script_data.append(data)

        # Block Logic: If data was collected, executes the script and propagates the result.
        if script_data != []:
            # Executes the script with the collected sensor data.
            result = script.run(script_data)

            # Block Logic: Propagates the script's result back to neighboring devices.
            for device in neighbours:
                if device is not own_device:
                    device.set_data(location, result)

            # Block Logic: Updates the own device's sensor data with the script's result.
    def run(self):
        """
        The main execution loop for the DeviceThread.
        It continuously fetches neighbor information, processes assigned scripts
        using a thread pool, and synchronizes with other devices.
        """


        
        # Block Logic: Initializes a thread pool for concurrent script execution.
        pool_size = 8
        pool = ThreadPool(pool_size)

        while True:
            # Block Logic: Fetches the latest neighbor information from the supervisor.
            neighbours = self.device.supervisor.get_neighbours()
            # If no neighbors (e.g., simulation terminated), break the loop.
            if neighbours is None:
                break

            # Block Logic: Processes scripts for the current timepoint.
            # It continues as long as the timepoint has not officially ended (more scripts expected).
            offset = 0
            while not self.device.timepoint_ended():
                scripts = self.device.scripts[offset:]
                # Adds each new script as a task to the thread pool for execution.
                for (script, location) in scripts:
                    pool.add_task(DeviceThread.run_script, self.device,
                                  neighbours, script, location)

                # Updates the offset to track which scripts have already been dispatched.
                offset = len(scripts)

            # Block Logic: Waits for all tasks in the thread pool to complete before proceeding.
            pool.wait()

            # Block Logic: Synchronizes with other devices at a global barrier,
            # ensuring all devices complete their timepoint processing before moving to the next.
            self.device.barrier.wait()

        # Block Logic: Terminates the thread pool gracefully upon simulation completion.
        pool.terminate()


from Queue import Queue
from threading import Thread

class Worker(Thread):
    """
    A worker thread class used by the ThreadPool to execute tasks.
    It continuously fetches tasks from a queue and executes them.
    """
    

    def __init__(self, tasks):
        """
        Initializes a new Worker thread.

        Args:
            tasks (Queue): The queue from which the worker thread will retrieve tasks.
        """

        
        Thread.__init__(self)
        self.tasks = tasks

    def run(self):
        """
        The main loop for the worker thread.
        It continuously gets tasks from the queue, executes them, and marks them as done.
        Exits the loop if a ValueError is raised (used for termination).
        """
        while True:
            # Block Logic: Retrieves a task (function, arguments, keyword arguments) from the queue.
            func, args, kargs = self.tasks.get()
            try:
                # Block Logic: Executes the retrieved function with its arguments.
                func(*args, **kargs)
            except ValueError:
                # Block Logic: Catches ValueError, used as a signal to terminate the worker thread.
                return
            finally:
                # Block Logic: Marks the task as done in the queue, regardless of success or failure.
                self.tasks.task_done()

class ThreadPool(object):
    """
    A simple Thread Pool implementation that manages a fixed number of worker threads
    to execute tasks concurrently. Tasks are submitted to a queue and processed by
    available workers.
    """
    

    def __init__(self, num_threads):
        """
        Initializes the ThreadPool with a specified number of worker threads.

        Args:
            num_threads (int): The number of worker threads to create in the pool.
        """

        
        # Block Logic: Initializes a task queue with a maximum size defined by num_threads.
        self.tasks = Queue(num_threads)
        # Block Logic: Creates a list of Worker threads, each associated with the task queue.
        self.workers = [Worker(self.tasks) for _ in range(num_threads)]

        # Block Logic: Starts each worker thread, allowing them to begin processing tasks from the queue.
        for worker in self.workers:
            worker.start()

    def add_task(self, func, *args, **kargs):
        """
        Adds a new task to the queue for execution by a worker thread.

        Args:
            func (callable): The function to be executed as a task.
            *args: Positional arguments to pass to the function.
            **kargs: Keyword arguments to pass to the function.
        """

        
        self.tasks.put((func, args, kargs))

    def wait(self):
        """
        Blocks until all tasks in the queue have been processed by the worker threads.
        """

        
        self.tasks.join()

    def terminate(self):
        """
        Shuts down the thread pool gracefully.
        It waits for all current tasks to complete, then signals worker threads
        to terminate by adding a special "raising_dummy" task.
        """

        
        self.wait()

        # Block Logic: Defines a dummy function that raises a ValueError.
        # This function is used to signal worker threads to terminate.
        def raising_dummy():
            raise ValueError


        # Block Logic: Adds a "raising_dummy" task for each worker to the queue.
        # This causes each worker to terminate gracefully when it processes its dummy task.
        for _ in range(len(self.workers)):
            self.add_task(raising_dummy)
        # Block Logic: Waits for each worker thread to complete its execution and terminate.
        for worker in self.workers:
            worker.join()
