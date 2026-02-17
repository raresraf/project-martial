"""
@file device.py
@brief Simulates a distributed device network using a centralized task scheduler and a thread pool.
@details This script models a network of devices that process data in synchronized time steps.
Unlike models where each device spawns its own threads, this implementation uses a central
TaskScheduler that manages a fixed pool of persistent worker threads. Devices act as task
producers, adding tasks to a shared work pool. A barrier synchronizes the devices after
all tasks for a time step have been submitted.
"""

from threading import Event, Thread, Lock
# The ReusableBarrierSem implementation is assumed to be in a file named barrier.py
from barrier import ReusableBarrierSem
# The TaskScheduler and Task classes are defined later in this file, but could be in separate files.
from taskscheduler import TaskScheduler
from task import Task


def add_lock_for_location(lock_per_location, location):
    """Utility function to add a new lock for a given data location."""
    lock_per_location.append((location, Lock()))

def get_lock_for_location(lock_per_location, location):
    """Utility function to retrieve the lock associated with a given data location."""
    for (loc, lock) in lock_per_location:
        if loc == location:
            return lock
    return None


class Device(object):
    """
    Represents a device node in the simulation. It produces tasks based on scripts
    assigned by a supervisor and submits them to a central TaskScheduler.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor
        # Event to signal when a new script is assigned.
        self.script_received = Event()
        self.scripts = []
        # A shared barrier, provided after initialization.
        self.barrier = None
        # A shared task scheduler, provided after initialization.
        self.taskscheduler = None
        # Event to signal when the supervisor has finished assigning work for a timepoint.
        self.timepoint_done = Event()
        self.thread = DeviceThread(self)
        self.thread.start()

    def __str__(self):
        

        return "Device %d" % self.device_id

    def share_barrier(self, barrier):
        """Assigns the shared barrier object to this device."""
        self.barrier = barrier

    def share_taskscheduler(self, taskscheduler):
        """Assigns the shared TaskScheduler object to this device."""
        self.taskscheduler = taskscheduler

    def setup_devices(self, devices):
        """
        Performs global setup for the simulation. Executed by device 0.
        It creates and distributes the shared barrier and task scheduler to all devices.
        """
        
        # Block invariant: This setup is performed only once by the primary device (device 0).
        if self.device_id == 0:

            lock_per_location = []

            
            # Aggregate all unique data locations from all devices and create a lock for each.
            for device in devices:
                for location in device.sensor_data:
                    lock = get_lock_for_location(lock_per_location, location)
                    if lock is None:
                        add_lock_for_location(lock_per_location, location)

            # Initialize the shared synchronization and scheduling objects.
            self.barrier = ReusableBarrierSem(len(devices))
            self.taskscheduler = TaskScheduler(lock_per_location)

            
            # Distribute the shared objects to all other devices.
            for device in devices:


                device.share_taskscheduler(self.taskscheduler)
                device.share_barrier(self.barrier)

    def assign_script(self, script, location):
        """
        Assigns a script to be executed by this device for the current time step.
        Called by the supervisor.
        """
        if script is not None:
            self.scripts.append((script, location))
            self.script_received.set()
        else:
            # A None script is the supervisor's signal that all assignments for this time step are done.
            self.timepoint_done.set()

    def get_data(self, location):
        """Retrieves sensor data for a given location."""
        if location in self.sensor_data:
            return self.sensor_data[location]
        return None

    def set_data(self, location, data):
        """Updates sensor data for a given location."""
        self.sensor_data[location] = data

    def shutdown(self):
        """Joins the main device thread to shut down the device."""
        self.thread.join()


class DeviceThread(Thread):
    """
    The main control loop for a device. Its primary responsibility is to receive scripts,
    create tasks, and submit them to the central TaskScheduler at each time step.
    """

    def __init__(self, device):
        
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device

    def run(self):
        
        # Busy-wait until the shared barrier has been initialized and assigned.
        while self.device.barrier is None:
            pass

        # Main simulation loop. Each iteration represents a discrete time step.
        while True:
            
            neighbours = self.device.supervisor.get_neighbours()

            
            # Pre-condition: Check for the simulation termination signal.
            if neighbours is None:
                # If this is the primary device, it's responsible for shutting down the scheduler.
                if self.device.device_id == 0:
                    
                    self.device.taskscheduler.finish = True
                    self.device.taskscheduler.wait_workers()
                break

            
            # Wait for the supervisor to finish assigning all scripts for this time step.
            self.device.timepoint_done.wait()
            self.device.timepoint_done.clear()

            
            # Block Logic: Create Task objects from scripts and submit them to the scheduler's work pool.
            for (script, location) in self.device.scripts:
                new_task = Task(self.device, script, location, neighbours)
                self.device.taskscheduler.add_task(new_task)

            
            # --- BARRIER SYNCHRONIZATION ---
            # All device threads wait here. This ensures that all tasks for the current time step
            # have been submitted to the scheduler before any device moves to the next time step.
            self.device.barrier.wait()


class Task(object):
    """
    A data class representing a single unit of work to be performed by a Worker thread.
    It encapsulates the logic (script) and the context (device, location, neighbours).
    """

    def __init__(self, device, script, location, neighbours):
        

        self.device = device
        self.script = script
        self.location = location
        self.neighbours = neighbours

    def execute(self):
        """
        The core execution logic. It gathers data, runs the script, and distributes the result.
        This method is called by a Worker thread.
        """

        script_data = []

        
        # Aggregate data from all neighboring devices for the specified location.
        for device in self.neighbours:
            data = device.get_data(self.location)
            if data is not None:
                script_data.append(data)

		
        # Include the local device's data.
        data = self.device.get_data(self.location)
        if data is not None:
            script_data.append(data)

        if script_data != []:
            
            # Run the computational part of the script.
            result = self.script.run(script_data)

            
            # Distribute the result to all neighboring devices.
            for device in self.neighbours:
                device.set_data(self.location, result)

            
            # Update the local device's data as well.
            self.device.set_data(self.location, result)


class TaskScheduler(object):
    """
    Manages a thread pool of persistent Worker threads and a central queue of tasks.
    It's responsible for distributing work to the workers.
    """

    def __init__(self, lock_per_location):
        

        self.nr_threads = 16
        self.lock_per_location = lock_per_location
        # The central work pool (queue) of tasks to be executed.
        self.workpool = []
        self.workpool_lock = Lock()
        self.workers_list = []
        # A flag to signal workers to terminate.
        self.finish = False

        self.start_workers()

    def add_task(self, new_task):
        """Adds a task to the work pool in a thread-safe manner."""
        with self.workpool_lock:
            self.workpool.append(new_task)

    def get_task(self):
        """Retrieves a task from the work pool in a thread-safe manner."""
        self.workpool_lock.acquire()
        if self.workpool != []:
            ret = self.workpool.pop()
        else:
            ret = None
        self.workpool_lock.release()
        return ret

    def start_workers(self):
        """Initializes and starts the pool of worker threads."""
        tid = 0
        while tid < self.nr_threads:
            thread = Worker(self)
            self.workers_list.append(thread)
            tid += 1

        for worker in self.workers_list:
            worker.start()

    def wait_workers(self):
        """Waits for all worker threads in the pool to complete."""
        for worker in self.workers_list:
            worker.join()

    def get_lock_per_location(self, location):
        """Retrieves the lock for a specific data location."""
        for (loc, lock) in self.lock_per_location:
            if loc == location:
                return lock
        return None


class Worker(Thread):
    """
    A persistent worker thread. It continuously fetches tasks from the TaskScheduler's
    work pool and executes them.
    """

    def __init__(self, taskscheduler):
        

        Thread.__init__(self)
        self.taskscheduler = taskscheduler

    def run(self):
        """
        The main loop for a worker. It fetches and executes tasks until a finish signal is received.
        """
        # The worker thread runs until the 'finish' flag is set by the main device.
        while True:
            if self.taskscheduler.finish == True:
                break

            # Inner loop to continuously poll for new tasks. This is a busy-wait loop.
            while True:
                task = self.taskscheduler.get_task()
                if task is None:
                    break
                # Acquire the specific lock for the task's data location before execution.
                with self.taskscheduler.get_lock_per_location(task.location):
                    task.execute()